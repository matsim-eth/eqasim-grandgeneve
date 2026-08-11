import unicodedata
import pandas as pd
import geopandas as gpd

import data.gtfs.utils as gtfs
from data.gtfs.cleaned import get_input_files
from data.od.thonon_boat_zones import QUERY_DATE

"""
Detects GTFS trips on the CGN boat lines modelled in data.od.thonon_boat_zones
(Lausanne<->Thonon, Lausanne<->Evian, Nyon<->Yvoire), on QUERY_DATE: a feed
lists a separate trip_id per schedule season (e.g. N3 runs far more
crossings in summer), so counting all trip_id rows without picking one day
overcounts actual daily crossings. Also exposes each line endpoint's stop
coordinates, for synthesis.population.spatial.primary.locations /
synthesis.population.trips to place and schedule boat commuters.
"""

# GTFS route_type 4 is the standard "Ferry" code; the Swiss extended
# hierarchy (opentransportdata.swiss, used by merged_gtfs.zip) instead tags
# every water transport route as 1000.
WATER_ROUTE_TYPES = {"4", "1000"}

LINES_OF_INTEREST = {
    "Lausanne <-> Thonon" : ("Lausanne", "Thonon"),
    "Lausanne <-> Evian"  : ("Lausanne", "Evian"),
    "Nyon <-> Yvoire"     : ("Nyon", "Yvoire"),
}


def configure(context):
    context.config("data_path")
    context.config("gtfs_path", "gtfs_idf")
    context.config("gtfs_files", None)


def remove_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn")


def find_line(origin_stop, destination_stop, lines = LINES_OF_INTEREST):
    """Which of `lines` has a trip's own (origin, destination) stop names as
    its two endpoints, in either direction; returns (line, origin_port,
    destination_port) with origin_port/destination_port the canonical short
    names in the matched order, or (None, None, None)."""
    origin, destination = remove_accents(origin_stop).lower(), remove_accents(destination_stop).lower()

    for line, (a, b) in lines.items():
        a_norm, b_norm = remove_accents(a).lower(), remove_accents(b).lower()
        if a_norm in origin and b_norm in destination:
            return line, a, b
        if b_norm in origin and a_norm in destination:
            return line, b, a

    return None, None, None


WEEKDAY_COLUMNS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def active_service_ids(feed, date):
    """service_ids running on `date` (a "YYYY-MM-DD" string), from calendar.txt's weekday/date-range plus calendar_dates.txt's exceptions."""
    gtfs_date = date.replace("-", "")
    weekday_column = WEEKDAY_COLUMNS[pd.Timestamp(date).weekday()]

    active = set()
    if "calendar" in feed:
        df_calendar = feed["calendar"]
        running = df_calendar[
            (df_calendar[weekday_column].astype(str) == "1")
            & (df_calendar["start_date"].astype(str) <= gtfs_date)
            & (df_calendar["end_date"].astype(str) >= gtfs_date)
        ]
        active |= set(running["service_id"])

    if "calendar_dates" in feed:
        df_exceptions = feed["calendar_dates"]
        df_exceptions = df_exceptions[df_exceptions["date"].astype(str) == gtfs_date]
        exception_type = df_exceptions["exception_type"].astype(str)
        active -= set(df_exceptions[exception_type == "2"]["service_id"])
        active |= set(df_exceptions[exception_type == "1"]["service_id"])

    return active


def find_boat_trips(feed, path):
    df_routes = feed["routes"]
    df_stops = feed["stops"].set_index("stop_id")
    df_stop_times = feed["stop_times"]

    water_routes = df_routes[df_routes["route_type"].astype(str).isin(WATER_ROUTE_TYPES)]
    water_trips = feed["trips"][feed["trips"]["route_id"].isin(water_routes["route_id"])]

    running_services = active_service_ids(feed, QUERY_DATE)
    water_trips = water_trips[water_trips["service_id"].isin(running_services)]

    water_stop_times = df_stop_times[df_stop_times["trip_id"].isin(water_trips["trip_id"])]

    print(f"Found {len(water_routes)} water-transport routes ({len(water_trips)} trips running on {QUERY_DATE}) in {path}")

    routes_by_id = water_routes.set_index("route_id")
    route_by_trip = water_trips.set_index("trip_id")["route_id"]

    rows = []
    for trip_id, trip_stop_times in water_stop_times.groupby("trip_id"):
        trip_stop_times = trip_stop_times.sort_values("stop_sequence")
        first, last = trip_stop_times.iloc[0], trip_stop_times.iloc[-1]

        if first["stop_id"] not in df_stops.index or last["stop_id"] not in df_stops.index:
            continue

        origin, destination = df_stops.loc[first["stop_id"]], df_stops.loc[last["stop_id"]]
        origin_stop, destination_stop = origin["stop_name"], destination["stop_name"]

        line, origin_port, destination_port = find_line(origin_stop, destination_stop)
        if line is None:
            continue

        route_id = route_by_trip[trip_id]
        route = routes_by_id.loc[route_id]

        rows.append({
            "line"            : line,
            "route_id"        : route_id,
            "route_short_name": route.get("route_short_name"),
            "trip_id"         : trip_id,
            "origin_stop"     : origin_stop,
            "destination_stop": destination_stop,
            "origin_port"     : origin_port,
            "destination_port": destination_port,
            "origin_lat"      : origin["stop_lat"],
            "origin_lon"      : origin["stop_lon"],
            "destination_lat" : destination["stop_lat"],
            "destination_lon" : destination["stop_lon"],
            "departure_time"  : first["departure_time"],
            "arrival_time"    : last["arrival_time"],
        })

    return pd.DataFrame(rows)


def extract_ports(df_boat_trips):
    """One row per canonical port (Thonon, Evian, Yvoire, Lausanne, Nyon)
    with its stop coordinates, reprojected to EPSG:2154 to match the rest of
    the synthesis pipeline."""
    origins = df_boat_trips[["origin_port", "origin_lat", "origin_lon"]].rename(
        columns = { "origin_port": "port", "origin_lat": "lat", "origin_lon": "lon" }
    )
    destinations = df_boat_trips[["destination_port", "destination_lat", "destination_lon"]].rename(
        columns = { "destination_port": "port", "destination_lat": "lat", "destination_lon": "lon" }
    )
    df_ports = pd.concat([origins, destinations], ignore_index = True).drop_duplicates("port")

    df_ports = gpd.GeoDataFrame(
        df_ports[["port"]],
        geometry = gpd.points_from_xy(df_ports["lon"], df_ports["lat"]),
        crs = "EPSG:4326"
    ).to_crs("EPSG:2154")

    return df_ports.reset_index(drop = True)


def execute(context):
    gtfs_files = context.config("gtfs_files")
    if not gtfs_files:
        gtfs_files = get_input_files("{}/{}".format(context.config("data_path"), context.config("gtfs_path")))

    df_boat_trips = pd.concat([
        find_boat_trips(gtfs.read_feed(path), path) for path in gtfs_files
    ], ignore_index = True)

    print(f"Detected {len(df_boat_trips)} boat trips on the {df_boat_trips['line'].nunique() if len(df_boat_trips) else 0} lines of interest:")
    if len(df_boat_trips):
        print(df_boat_trips.groupby("line", observed = True)["trip_id"].nunique().to_string())

    return { "trips": df_boat_trips, "ports": extract_ports(df_boat_trips) }
