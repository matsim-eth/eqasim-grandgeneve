import numpy as np
import pandas as pd

"""
Overrides synthesis.population.trips for boat commuters (see
synthesis.population.spatial.primary.locations.select_boat_users) with a
GTFS-scheduled home->work->home chain. Passthrough when
generate_outbound_flows is off.
"""

HOME_TO_WORK_WINDOW = (6 * 3600 + 30 * 60, 9 * 3600)
WORK_TO_HOME_WINDOW = (16 * 3600, 18 * 3600 + 30 * 60)

TRIP_COLUMNS = [
    "person_id", "trip_index", "departure_time", "arrival_time",
    "preceding_purpose", "following_purpose", "is_first_trip", "is_last_trip",
    "trip_duration", "activity_duration", "mode"
]


def configure(context):
    context.stage("synthesis.population.trips")
    context.config("random_seed")

    context.config("generate_outbound_flows")
    if context.config("generate_outbound_flows"):
        context.stage("synthesis.population.spatial.primary.locations")
        context.stage("data.gtfs.boat_trips")


def gtfs_time_to_seconds(value):
    hours, minutes, seconds = (int(part) for part in str(value).split(":"))
    return hours * 3600 + minutes * 60 + seconds


def select_departure(df_schedule, line, origin_port, window, random):
    """A random (departure_time, arrival_time), in seconds, among df_schedule
    rows on `line` departing `origin_port` within `window`; None if none match."""
    candidates = df_schedule[(df_schedule["line"] == line) & (df_schedule["origin_port"] == origin_port)]
    if len(candidates) == 0:
        return None

    seconds = candidates["departure_time"].map(gtfs_time_to_seconds)
    candidates = candidates[(seconds >= window[0]) & (seconds < window[1])]
    if len(candidates) == 0:
        return None

    row = candidates.iloc[random.integers(len(candidates))]
    return gtfs_time_to_seconds(row["departure_time"]), gtfs_time_to_seconds(row["arrival_time"])


def build_boat_chain(df_boat_users, df_schedule, random_seed):
    """Builds a synthetic home->work->home chain for each selected boat
    commuter, using an actual GTFS departure drawn from their assigned line
    (06:30-09:00 outbound, 16:00-18:30 return). Commuters with no scheduled
    departure in the relevant window fall back to their normal HTS-derived
    chain."""
    random = np.random.default_rng(random_seed)

    rows = []
    resolved_person_ids = []

    for row in df_boat_users.itertuples():
        outbound = select_departure(df_schedule, row.line, row.home_port, HOME_TO_WORK_WINDOW, random)
        inbound  = select_departure(df_schedule, row.line, row.work_port, WORK_TO_HOME_WINDOW, random)

        if outbound is None or inbound is None:
            continue

        departure_1, arrival_1 = outbound
        departure_2, arrival_2 = inbound

        rows.append(dict(
            person_id         = row.person_id,
            trip_index        = 0,
            departure_time    = float(departure_1) - 120, # 2 minutes before boat departure 
            arrival_time      = float(arrival_1),
            preceding_purpose = "home", 
            following_purpose = "work",
            is_first_trip     = True, 
            is_last_trip      = False,
            trip_duration     = float(arrival_1 - departure_1) + 120,
            activity_duration = float(departure_2 - 30*60 - arrival_1), # matches trip 1's actual (buffered) departure_time
            mode              = "pt"
        ))

        rows.append(dict(
            person_id         = row.person_id, 
            trip_index        = 1,
            departure_time    = float(departure_2) - 30*60, # 30 minutes before to try and not miss the boat connection 
            arrival_time      = float(arrival_2),
            preceding_purpose = "work", 
            following_purpose = "home",
            is_first_trip     = False, 
            is_last_trip      = True,
            trip_duration     = float(arrival_2 - departure_2) + 30*60,
            activity_duration = np.nan,
            mode              = "pt"
        ))

        resolved_person_ids.append(row.person_id)

    n_fallback = len(df_boat_users) - len(resolved_person_ids)

    print("Thonon -> CH boat commute mode: %d boat commuters given a custom home->work->home chain (%d fell back to their normal chain due to missing GTFS boat trips)" % (
        len(resolved_person_ids), n_fallback
    ))

    return pd.DataFrame(rows, columns = TRIP_COLUMNS), resolved_person_ids


def execute(context):
    df_trips = context.stage("synthesis.population.trips")

    if not context.config("generate_outbound_flows"):
        return df_trips

    df_boat_users = context.stage("synthesis.population.spatial.primary.locations")[2]
    if len(df_boat_users) == 0:
        return df_trips

    df_schedule = context.stage("data.gtfs.boat_trips")["trips"]
    df_boat_chain, resolved_person_ids = build_boat_chain(df_boat_users, df_schedule, context.config("random_seed"))

    if len(resolved_person_ids) == 0:
        return df_trips

    df_trips = df_trips.copy()
    df_trips["mode"] = df_trips["mode"].astype(str)
    df_trips = df_trips[~df_trips["person_id"].isin(resolved_person_ids)]

    df_trips = pd.concat([df_trips, df_boat_chain], ignore_index = True)
    df_trips = df_trips.sort_values(by = ["person_id", "trip_index"]).reset_index(drop = True)

    df_trips["mode"] = df_trips["mode"].astype("category")
    return df_trips
