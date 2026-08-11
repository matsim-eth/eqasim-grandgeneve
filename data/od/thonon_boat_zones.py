import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

"""
Boat and land travel times per commune of the Arrondissement de
Thonon-les-Bains, for the Lausanne area and the district de Nyon. Ported
from Documents/Tests/Cross border commuters/boat_zones.py. Consumed by
data.od.cleaned (via the advantage_min derived below) to reassign a modelled
share of car/pt work commuters to "boat". recompute_thonon_boat_advantage
picks the source: a precomputed CSV from data_path (default), or a live
rederivation from transport.opendata.ch + OSRM. The Lausanne/Nyon
destination-group municipality lists are always baked in
(LAUSANNE_AREA_CH_IDS / NYON_DISTRICT_CH_IDS) rather than recomputed -- see
ggs_vs_mobpro.py's find_neighboring_municipalities /
find_district_municipalities if they ever need regenerating.
"""

TIMETABLE_API = "https://transport.opendata.ch/v1"
OSRM_TABLE    = "https://router.project-osrm.org/table/v1/driving/"

TIMEZONE = ZoneInfo("Europe/Zurich")

# A term-time Monday, so neither school holidays nor the weekend timetable applies.
QUERY_DATE = "2026-09-14"

# Averaged over a morning's worth of departures rather than one arbitrary
# moment, since the boat's sparse schedule can flip a commune's classification
# depending on whether a sailing has just gone.
HOME_DEPARTURES = ["06:30", "07:00", "07:30", "08:00", "08:30"]

# Departure windows queried per (access point, destination); together they
# cover the morning and well past it.
WINDOW_TIMES           = ["05:30", "07:00", "08:30", "10:00"]
CONNECTIONS_PER_WINDOW = 10

# OSRM's free-flow road times flatter long drives towards the Geneva border,
# i.e. the "land" alternative; raise this to penalise long drives.
CONGESTION_FACTOR = 1.0

# Longest drive anyone is assumed to make to reach a port or station.
MAX_DRIVE_MINUTES = 40

# The three CGN cross-border landing stages on the French shore, plus the
# railway stations offering the land route through Annemasse/Geneva.
ACCESS_POINTS = {
    "Évian (port)"      : ("Evian-les-Bains (F) (lac)",  "port"),
    "Thonon (port)"     : ("Thonon-les-Bains (F) (lac)", "port"),
    "Yvoire (port)"     : ("Yvoire (F) (lac)",           "port"),
    "Évian (rail)"      : ("Evian-les-Bains",            "rail"),
    "Thonon (rail)"     : ("Thonon-les-Bains",           "rail"),
    "Perrignier"        : ("Perrignier",                 "rail"),
    "Bons-en-Chablais"  : ("Bons-en-Chablais",           "rail"),
    "Machilly"          : ("Machilly",                   "rail"),
    "Annemasse"         : ("Annemasse",                  "rail"),
    "Genève"            : ("Genève",                     "rail"),
    "St-Gingolph (rail)": ("St-Gingolph (Suisse)",       "rail"),
}

# Genève is excluded from classification (though still fetched): a commuter
# who has already driven across the border is more likely to drive the rest
# of the way than to park there, and it dominates the result otherwise.
EXCLUDED_ACCESS_POINTS = {"Genève"}

# Queried against the timetable API; each stands in for its whole destination group.
DESTINATIONS = ["Lausanne", "Nyon"]

# CGN's cross-border commuter lines appear as BATN1/2/3 in the timetable.
BOAT_CATEGORY = "BAT"

# The 70 communes of the Arrondissement de Thonon-les-Bains
# (https://fr.wikipedia.org/wiki/Arrondissement_de_Thonon-les-Bains).
ARRONDISSEMENT_DE_THONON = {
    "Abondance": "74001", "Allinges": "74005", "Anthy-sur-Léman": "74013",
    "Armoy": "74020", "Ballaison": "74025", "La Baume": "74030",
    "Bellevaux": "74032", "Bernex": "74033", "Le Biot": "74034",
    "Boëge": "74037", "Bogève": "74038", "Bonnevaux": "74041",
    "Bons-en-Chablais": "74043", "Brenthonne": "74048", "Burdignin": "74050",
    "Cervens": "74053", "Champanges": "74057", "La Chapelle-d'Abondance": "74058",
    "Châtel": "74063", "Chens-sur-Léman": "74070", "Chevenoz": "74073",
    "La Côte-d'Arbroz": "74091", "Douvaine": "74105", "Draillant": "74106",
    "Essert-Romand": "74114", "Évian-les-Bains": "74119", "Excenevex": "74121",
    "Fessy": "74126", "Féternes": "74127", "La Forclaz": "74129",
    "Les Gets": "74134", "Habère-Lullin": "74139", "Habère-Poche": "74140",
    "Larringes": "74146", "Loisin": "74150", "Lugrin": "74154",
    "Lullin": "74155", "Lully": "74156", "Lyaud": "74157",
    "Margencel": "74163", "Marin": "74166", "Massongy": "74171",
    "Maxilly-sur-Léman": "74172", "Meillerie": "74175", "Messery": "74180",
    "Montriond": "74188", "Morzine": "74191", "Nernier": "74199",
    "Neuvecelle": "74200", "Novel": "74203", "Orcier": "74206",
    "Perrignier": "74210", "Publier": "74218", "Reyvroz": "74222",
    "Saint-André-de-Boëge": "74226", "Saint-Gingolph": "74237",
    "Saint-Jean-d'Aulps": "74238", "Saint-Paul-en-Chablais": "74249",
    "Saxel": "74261", "Sciez": "74263", "Seytroux": "74271",
    "Thollon-les-Mémises": "74279", "Thonon-les-Bains": "74281",
    "Vacheresse": "74286", "Vailly": "74287", "Veigy-Foncenex": "74293",
    "La Vernaz": "74295", "Villard": "74301", "Vinzier": "74308",
    "Yvoire": "74315",
}

LAUSANNE_GROUP = "Lausanne area"
NYON_GROUP     = "District de Nyon"

# Lausanne plus every municipality touching it (swissBOUNDARIES3D), as of 2026.
LAUSANNE_AREA_CH_IDS = [
    "CH05586", "CH05648", "CH05590", "CH05611", "CH05587", "CH05523",
    "CH05514", "CH05635", "CH05583", "CH05582", "CH05584", "CH05792",
    "CH05527", "CH05591", "CH05516", "CH05592", "CH05515", "CH05589",
    "CH05585", "CH05627",
]

# Every municipality of the district de Nyon, VD (BEZIRKSNUM 2228), as of 2026.
NYON_DISTRICT_CH_IDS = [
    "CH05429", "CH05430", "CH05434", "CH05701", "CH05702", "CH05703",
    "CH05704", "CH05705", "CH05706", "CH05707", "CH05708", "CH05709",
    "CH05710", "CH05711", "CH05712", "CH05713", "CH05714", "CH05715",
    "CH05716", "CH05717", "CH05718", "CH05719", "CH05720", "CH05721",
    "CH05722", "CH05723", "CH05724", "CH05725", "CH05726", "CH05727",
    "CH05728", "CH05729", "CH05730", "CH05731", "CH05732", "CH05852",
    "CH05853", "CH05854", "CH05855", "CH05856", "CH05857", "CH05858",
    "CH05859", "CH05860", "CH05861", "CH05862", "CH05863",
]

DESTINATION_GROUPS = {
    LAUSANNE_GROUP: LAUSANNE_AREA_CH_IDS,
    NYON_GROUP    : NYON_DISTRICT_CH_IDS,
}

# Destination group -> column prefix used for its travel times (own DESTINATIONS entry, lowercased).
ZONE_COLUMN_KEY = {
    LAUSANNE_GROUP: "lausanne",
    NYON_GROUP    : "nyon",
}

TRAVEL_TIME_COLUMNS = ["code_insee"] + [
    f"{key}_{mode}_min" for key in ZONE_COLUMN_KEY.values() for mode in ("boat", "land")
]
ADVANTAGE_COLUMNS = ["code_insee"] + [f"{key}_advantage_min" for key in ZONE_COLUMN_KEY.values()]


def configure(context):
    context.config("data_path")
    context.config("thonon_boat_travel_times_path", "cross_border/thonon_boat_travel_times.csv")
    context.config("recompute_thonon_boat_advantage", False)

    if context.config("recompute_thonon_boat_advantage"):
        context.stage("data.spatial.municipalities")


# Both APIs are free public services with rate limits, so requests are
# spaced out and a 429 is waited out rather than hammered.
MIN_REQUEST_INTERVAL = 1.2
_last_request_at = [0.0]


def _get(url, params, attempts = 4):
    """Query a public API, throttled, retrying on rate limits and hiccups."""
    for attempt in range(attempts):
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at[0])
        if wait > 0:
            time.sleep(wait)

        try:
            response = requests.get(url, params = params, timeout = 45)
            _last_request_at[0] = time.monotonic()

            if response.status_code == 429:
                backoff = int(response.headers.get("Retry-After", 0)) or 20 * (attempt + 1)
                print(f"INFO:   rate limited, waiting {backoff}s")
                time.sleep(backoff)
                continue

            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            _last_request_at[0] = time.monotonic()
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))

    raise RuntimeError(f"gave up on {url} after {attempts} attempts (rate limited)")


def resolve_access_points():
    """Look up each access point's timetable id and coordinates."""
    resolved = {}
    for name, (query, kind) in ACCESS_POINTS.items():
        stations = _get(f"{TIMETABLE_API}/locations", {"query": query, "type": "station"}).get("stations", [])
        stations = [s for s in stations if s.get("id") and s["coordinate"]["x"]]
        if not stations:
            raise ValueError(f"access point {name!r}: no station found for query {query!r}")
        station = stations[0]
        resolved[name] = {
            "kind" : kind,
            "stop" : station["name"],
            "lat"  : station["coordinate"]["x"],
            "lon"  : station["coordinate"]["y"],
        }
    return resolved


def fetch_departures(access_points):
    """For every (access point, destination), collect the morning's departures."""
    departures = {}
    for name, info in access_points.items():
        for destination in DESTINATIONS:
            found = {}
            for window in WINDOW_TIMES:
                payload = _get(f"{TIMETABLE_API}/connections", {
                    "from" : info["stop"],
                    "to"   : destination,
                    "date" : QUERY_DATE,
                    "time" : window,
                    "limit": CONNECTIONS_PER_WINDOW,
                })
                for connection in payload.get("connections", []):
                    departure = connection["from"].get("departureTimestamp")
                    arrival   = connection["to"].get("arrivalTimestamp")
                    if not (departure and arrival):
                        continue
                    categories = [s["journey"]["category"] for s in connection["sections"] if s.get("journey")]
                    found[departure] = {
                        "departure" : departure,
                        "arrival"   : arrival,
                        "uses_boat" : any(c.startswith(BOAT_CATEGORY) for c in categories),
                        "route"     : " > ".join(categories),
                    }
            departures[f"{name}|{destination}"] = sorted(found.values(), key = lambda d: d["departure"])
            print(f"INFO:   {name:20s} -> {destination:9s} {len(found):2d} departures")
    return departures


def fetch_drive_times(commune_points, access_points):
    """Road travel time (seconds) from every commune to every access point, as a single OSRM matrix request."""
    origins      = [(p.y, p.x) for p in commune_points.values()]
    destinations = [(a["lat"], a["lon"]) for a in access_points.values()]

    coordinates = ";".join(f"{lon},{lat}" for lat, lon in origins + destinations)
    sources      = ";".join(str(i) for i in range(len(origins)))
    targets      = ";".join(str(i + len(origins)) for i in range(len(destinations)))

    payload = _get(OSRM_TABLE + coordinates, {
        "sources"     : sources,
        "destinations": targets,
        "annotations" : "duration",
    })
    if payload.get("code") != "Ok":
        raise ValueError(f"OSRM refused the matrix request: {payload.get('code')} {payload.get('message', '')}")

    return {
        commune: dict(zip(access_points, row))
        for commune, row in zip(commune_points, payload["durations"])
    }


def best_by_mode(commune, destination, cache, home_departure_ts):
    """Earliest arrival at `destination` by boat, and without one, leaving `commune` at home_departure_ts."""
    best = {}
    for access_point, drive_seconds in cache["drive_times"][commune].items():
        if drive_seconds is None or access_point in EXCLUDED_ACCESS_POINTS:
            continue
        drive_seconds = drive_seconds * CONGESTION_FACTOR
        if drive_seconds > MAX_DRIVE_MINUTES * 60:
            continue
        arrive_at_access = home_departure_ts + drive_seconds

        for option in cache["departures"].get(f"{access_point}|{destination}", []):
            if option["departure"] < arrive_at_access:
                continue
            mode      = "boat" if option["uses_boat"] else "land"
            candidate = {
                "arrival"      : option["arrival"],
                "total_minutes": round((option["arrival"] - home_departure_ts) / 60),
            }
            if mode not in best or candidate["arrival"] < best[mode]["arrival"]:
                best[mode] = candidate

    return best


def classify_communes(cache):
    """One row per commune with <key>_boat_min / <key>_land_min (door-to-door minutes, averaged over HOME_DEPARTURES) for key in "lausanne"/"nyon"."""
    departure_timestamps = {
        t: datetime.fromisoformat(f"{QUERY_DATE}T{t}:00").replace(tzinfo = TIMEZONE).timestamp()
        for t in HOME_DEPARTURES
    }

    rows = []
    for commune in cache["drive_times"]:
        row = {"commune": commune, "code_insee": ARRONDISSEMENT_DE_THONON[commune]}

        for destination in DESTINATIONS:
            key = destination.lower()

            by_departure = {t: best_by_mode(commune, destination, cache, ts)
                            for t, ts in departure_timestamps.items()}

            times = {"boat": [], "land": []}
            for t in HOME_DEPARTURES:
                for mode, candidate in by_departure[t].items():
                    times[mode].append(candidate["total_minutes"])

            row[f"{key}_boat_min"] = round(sum(times["boat"]) / len(times["boat"]), 1) if times["boat"] else None
            row[f"{key}_land_min"] = round(sum(times["land"]) / len(times["land"]), 1) if times["land"] else None

        rows.append(row)

    return pd.DataFrame(rows)


def add_advantage(detail):
    """Adds <key>_advantage_min (positive = boat faster) from <key>_boat_min / <key>_land_min."""
    detail = detail.copy()
    for key in ZONE_COLUMN_KEY.values():
        detail[f"{key}_advantage_min"] = detail[f"{key}_land_min"] - detail[f"{key}_boat_min"]
    return detail


def build_commune_points(context):
    df_municipalities = context.stage("data.spatial.municipalities")
    thonon_codes = set(ARRONDISSEMENT_DE_THONON.values())

    by_code = df_municipalities[df_municipalities["commune_id"].isin(thonon_codes)]
    by_code = by_code.set_index("commune_id").to_crs("EPSG:4326")

    return {
        name: by_code.loc[code, "geometry"].representative_point()
        for name, code in ARRONDISSEMENT_DE_THONON.items()
        if code in by_code.index
    }


def execute(context):
    if not context.config("recompute_thonon_boat_advantage"):
        path = "%s/%s" % (context.config("data_path"), context.config("thonon_boat_travel_times_path"))
        detail = pd.read_csv(path, dtype = {"code_insee": str})
        print(f"Read Thonon boat/land travel times for {len(detail)} communes from {path}")
    else:
        print("Recomputing the Thonon boat/land travel times from live APIs (transport.opendata.ch, OSRM) ...")

        commune_points = build_commune_points(context)
        missing = set(ARRONDISSEMENT_DE_THONON.values()) - {
            code for name, code in ARRONDISSEMENT_DE_THONON.items() if name in commune_points
        }
        if missing:
            print(f"WARNING: {len(missing)} communes of the arrondissement have no polygon in data.spatial.municipalities: {sorted(missing)}")

        print("INFO: resolving access points")
        access_points = resolve_access_points()

        print("INFO: fetching timetables")
        departures = fetch_departures(access_points)

        print("INFO: fetching road times")
        drive_times = fetch_drive_times(commune_points, access_points)

        detail = classify_communes({"departures": departures, "drive_times": drive_times})

    detail = add_advantage(detail)

    unreachable = detail[detail[[f"{key}_advantage_min" for key in ZONE_COLUMN_KEY.values()]].isna().all(axis = 1)]
    if len(unreachable):
        print(f"WARNING: {len(unreachable)} communes have no reachable itinerary to either destination: "
              f"{', '.join(unreachable['commune'])}")

    return detail[ADVANTAGE_COLUMNS]
