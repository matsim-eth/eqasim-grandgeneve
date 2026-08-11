import numpy as np
import pandas as pd
import geopandas as gpd
from .candidates import EDUCATION_MAPPING
import data.od.thonon_boat_zones as thonon_boat_zones
import data.gtfs.boat_trips as boat_trips

# Logistic curve turning the boat's time advantage into a mode-switch
# probability. BOAT_CAR_PARK_AND_RIDE_SHARE is corrected per destination
# zone using census mode shares (data.od.cleaned, see corridor_mode_shares)
# rather than applied directly to each candidate's own HTS-matched mode:
# the local HTS survey sample for this corridor is thin enough that mode
# ended up almost perfectly (and spuriously) collinear with destination zone.
BOAT_P_MAX        = 0.80
BOAT_MIDPOINT_MIN = 15
BOAT_SLOPE_MIN    = 10
BOAT_CAR_PARK_AND_RIDE_SHARE = 0.20

# Work destination group -> the canonical port name used for it in
# data.gtfs.boat_trips.LINES_OF_INTEREST.
WORK_PORT_BY_ZONE = { "lausanne": "Lausanne", "nyon": "Nyon" }

BOAT_USER_COLUMNS = ["person_id", "home_port", "work_port", "line"]


def configure(context):
    context.stage("synthesis.population.spatial.primary.candidates")
    context.stage("synthesis.population.spatial.commute_distance")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.locations.work")
    context.stage("synthesis.locations.education")
    context.stage("synthesis.population.trips")

    context.config("education_location_source", "bpe")
    context.config("random_seed")

    context.config("generate_outbound_flows")
    if context.config("generate_outbound_flows"):
        context.stage("data.od.thonon_boat_zones")
        context.stage("data.gtfs.boat_trips")
        context.stage("data.spatial.municipalities")
        context.stage("data.od.cleaned")


def define_distance_ordering(df_persons, df_candidates, progress):
    indices = []

    f_available = np.ones((len(df_candidates),), dtype = bool)
    costs = np.ones((len(df_candidates),)) * np.inf

    commute_coordinates = np.vstack([
        df_candidates["geometry"].x.values,
        df_candidates["geometry"].y.values
    ]).T

    for home_coordinate, commute_distance in zip(df_persons["home_location"], df_persons["commute_distance"]):
        home_coordinate = np.array([home_coordinate.x, home_coordinate.y])
        distances = np.sqrt(np.sum((commute_coordinates[f_available] - home_coordinate)**2, axis = 1))
        costs[f_available] = np.abs(distances - commute_distance)

        selected_index = np.argmin(costs)
        indices.append(selected_index)
        f_available[selected_index] = False
        costs[selected_index] = np.inf

        progress.update()

    assert len(set(indices)) == len(df_candidates)

    return indices


define_ordering = define_distance_ordering


def process_municipality(context, origin_id):
    # Load data
    df_candidates, df_persons = context.data("df_candidates"), context.data("df_persons")

    # Find relevant records
    df_persons = df_persons[df_persons["commune_id"] == origin_id][[
        "person_id", "home_location", "commute_distance"
    ]].copy()
    df_candidates = df_candidates[df_candidates["origin_id"] == origin_id]

    # From previous step, this should be equal!
    assert len(df_persons) == len(df_candidates)

    indices = define_ordering(df_persons, df_candidates, context.progress)
    df_candidates = df_candidates.iloc[indices]

    df_candidates["person_id"] = df_persons["person_id"].values
    df_candidates = df_candidates.rename(columns = dict(destination_id = "commune_id"))

    return df_candidates[["person_id", "commune_id", "location_id", "geometry"]]


def process(context, purpose, df_persons, df_candidates):
    unique_ids = df_candidates["origin_id"].unique()

    df_result = []

    with context.progress(label = "Distributing %s destinations" % purpose, total = len(df_persons)) as progress:
        with context.parallel(dict(df_persons = df_persons, df_candidates = df_candidates)) as parallel:
            for df_partial in parallel.imap_unordered(process_municipality, unique_ids):
                df_result.append(df_partial)

    return pd.concat(df_result).sort_index()


def boat_probability(advantage_minutes, p_max = BOAT_P_MAX, midpoint = BOAT_MIDPOINT_MIN, slope = BOAT_SLOPE_MIN):
    """Share of car/pt work commuters taking the boat, given how many minutes faster it is."""
    advantage = pd.Series(advantage_minutes, dtype = float)
    p = p_max / (1 + np.exp(-(advantage - midpoint) / slope))
    return p.where(advantage.notna(), 0.0)


def assign_ports(selected, zone_key, home_commune, df_municipalities, df_ports):
    """For each selected boat user, picks the concrete line and home/work
    ports: Nyon-area workers always use Yvoire via "Nyon <-> Yvoire";
    Lausanne-area workers pick whichever of Thonon/Evian is geographically
    closer to their home commune, via "Lausanne <-> Thonon" / "Lausanne <->
    Evian"."""
    work_port = zone_key.loc[selected].map(WORK_PORT_BY_ZONE)
    home_port = pd.Series(index = selected, dtype = object)
    line = pd.Series(index = selected, dtype = object)

    nyon_persons = selected[(work_port == "Nyon").values]
    home_port.loc[nyon_persons] = "Yvoire"
    line.loc[nyon_persons] = "Nyon <-> Yvoire"

    lausanne_persons = selected[(work_port == "Lausanne").values]
    if len(lausanne_persons) > 0:
        municipality_geometry = df_municipalities.set_index("commune_id")["geometry"]
        home_points = gpd.GeoSeries(
            home_commune.loc[lausanne_persons].astype(str).map(municipality_geometry).apply(lambda g: g.representative_point()).values,
            index = lausanne_persons, crs = df_municipalities.crs
        )

        port_geometry = df_ports.set_index("port")["geometry"]
        distance_to_thonon = home_points.distance(port_geometry["Thonon"])
        distance_to_evian  = home_points.distance(port_geometry["Evian"])

        thonon_persons = lausanne_persons[(distance_to_thonon <= distance_to_evian).values]
        evian_persons  = lausanne_persons[(distance_to_thonon > distance_to_evian).values]

        home_port.loc[thonon_persons] = "Thonon"
        line.loc[thonon_persons]      = "Lausanne <-> Thonon"
        home_port.loc[evian_persons]  = "Evian"
        line.loc[evian_persons]       = "Lausanne <-> Evian"

    return pd.DataFrame({
        "person_id": selected,
        "home_port": home_port.loc[selected].values,
        "work_port": work_port.loc[selected].values,
        "line": line.loc[selected].values,
    })


def corridor_mode_shares(df_od_work):
    """Census (MOBPRO) car share among car/pt work commuters from the
    Arrondissement de Thonon to each destination zone, indexed by
    thonon_boat_zones.ZONE_COLUMN_KEY values ("lausanne", "nyon"). Used to
    correct BOAT_CAR_PARK_AND_RIDE_SHARE for this corridor: the local HTS
    survey sample is too thin for this niche subpopulation and turns out to
    badly misrepresent the true car/pt split, per zone, relative to the much
    larger census sample."""
    thonon_codes = set(thonon_boat_zones.ARRONDISSEMENT_DE_THONON.values())
    destination_to_group = {
        ch_id: group
        for group, ch_ids in thonon_boat_zones.DESTINATION_GROUPS.items()
        for ch_id in ch_ids
    }

    df = df_od_work[df_od_work["origin_id"].astype(str).isin(thonon_codes)].copy()
    df["zone_key"] = df["destination_id"].astype(str).map(destination_to_group).map(thonon_boat_zones.ZONE_COLUMN_KEY)
    df = df[df["zone_key"].notna() & df["commute_mode"].isin(["car", "pt"])]

    totals = df.groupby(["zone_key", "commute_mode"], observed = True)["weight"].sum().unstack(fill_value = 0.0)
    return totals["car"] / (totals["car"] + totals["pt"])


def select_boat_users(df_persons, df_work, df_trips, advantage, df_od_work, df_municipalities, df_ports, random_seed):
    """Selects, among agents living in the Arrondissement de Thonon and
    assigned a workplace in the Lausanne area / district de Nyon
    (Switzerland), a modelled share of car/pt work commuters as boat users,
    and picks each one's departure/arrival port and line. Nothing else is
    changed here (e.g. trip mode) -- that is built downstream in
    synthesis.population.trips, using this selection plus the actual GTFS
    boat schedule."""
    home_commune = df_persons.set_index("person_id")["commune_id"]
    work_commune = df_work.drop_duplicates("person_id").set_index("person_id")["commune_id"]

    destination_to_group = {
        ch_id: group
        for group, ch_ids in thonon_boat_zones.DESTINATION_GROUPS.items()
        for ch_id in ch_ids
    }
    zone_key = work_commune.astype(str).map(destination_to_group).map(thonon_boat_zones.ZONE_COLUMN_KEY)

    thonon_codes = set(thonon_boat_zones.ARRONDISSEMENT_DE_THONON.values())
    in_scope = home_commune.reindex(work_commune.index).astype(str).isin(thonon_codes) & zone_key.notna()
    scoped_persons = in_scope[in_scope].index

    empty = pd.DataFrame(columns = BOAT_USER_COLUMNS)

    if len(scoped_persons) == 0:
        print("Thonon -> CH boat commute mode: 0 agents in scope")
        return empty

    is_work_leg = (df_trips["following_purpose"] == "work") | (df_trips["preceding_purpose"] == "work")
    df_work_trips = df_trips[is_work_leg & df_trips["person_id"].isin(scoped_persons)]
    mode_by_person = df_work_trips.drop_duplicates("person_id").set_index("person_id")["mode"].astype(str)

    candidate_persons = mode_by_person[mode_by_person.isin(["car", "pt"])].index

    if len(candidate_persons) == 0:
        print("Thonon -> CH boat commute mode: 0 / 0 work commuters selected as boat users (%d agents in scope)" % len(scoped_persons))
        return empty

    advantage_long = advantage.melt(
        id_vars = ["code_insee"],
        value_vars = [f"{key}_advantage_min" for key in thonon_boat_zones.ZONE_COLUMN_KEY.values()],
        var_name = "column", value_name = "advantage_min"
    )
    advantage_long["zone_key"] = advantage_long["column"].str.replace("_advantage_min", "", regex = False)
    advantage_map = advantage_long.set_index(["code_insee", "zone_key"])["advantage_min"]

    lookup_keys = pd.MultiIndex.from_arrays([
        home_commune.loc[candidate_persons].astype(str), zone_key.loc[candidate_persons]
    ])
    advantage_min = pd.Series(advantage_map.reindex(lookup_keys).values, index = candidate_persons)

    p_pt = boat_probability(advantage_min)

    car_share = corridor_mode_shares(df_od_work)
    effective_multiplier = car_share * BOAT_CAR_PARK_AND_RIDE_SHARE + (1 - car_share) * 1.0
    p = p_pt * zone_key.loc[candidate_persons].map(effective_multiplier)

    random = np.random.default_rng(random_seed)
    draw = random.random(size = len(candidate_persons))
    selected = candidate_persons[draw < p.values]

    print("Thonon -> CH boat commute mode: %d / %d work commuters selected as boat users (%d agents in scope)" % (
        len(selected), len(candidate_persons), len(scoped_persons)
    ))

    if len(selected) == 0:
        return empty

    return assign_ports(selected, zone_key, home_commune, df_municipalities, df_ports)


def execute(context):
    data = context.stage("synthesis.population.spatial.primary.candidates")
    df_persons = data["persons"]

    # Separate data set
    df_work = df_persons[df_persons["has_work_trip"]]
    df_education = df_persons[df_persons["has_education_trip"]]

    # Attach home locations
    df_home = context.stage("synthesis.population.spatial.home.locations")

    df_work = pd.merge(df_work, df_home[["household_id", "geometry"]].rename(columns = {
        "geometry": "home_location"
    }), how = "left", on = "household_id")

    df_education = pd.merge(df_education, df_home[["household_id", "geometry"]].rename(columns = {
        "geometry": "home_location"
    }), how = "left", on = "household_id")

    # Attach commute distances
    df_commute_distance = context.stage("synthesis.population.spatial.commute_distance")

    df_work = pd.merge(df_work, df_commute_distance["work"], how = "left", on = "person_id")
    df_education = pd.merge(df_education, df_commute_distance["education"], how = "left", on = "person_id")

    # Attach geometry
    df_locations = context.stage("synthesis.locations.work")[["location_id", "geometry"]]
    df_work_candidates = data["work_candidates"]
    df_work_candidates = pd.merge(df_work_candidates, df_locations, how = "left", on = "location_id")
    df_work_candidates = gpd.GeoDataFrame(df_work_candidates)

    df_locations = context.stage("synthesis.locations.education")[["education_type", "location_id", "geometry"]]
    df_education_candidates = data["education_candidates"]
    df_education_candidates = pd.merge(df_education_candidates, df_locations, how = "left", on = "location_id")
    df_education_candidates = gpd.GeoDataFrame(df_education_candidates)

    # Assign destinations
    df_work = process(context, "work", df_work, df_work_candidates)

    if context.config("education_location_source") == "bpe":
        df_education = process(context, "education", df_education, df_education_candidates)

    else :
        education = []
        for prefix, education_type in EDUCATION_MAPPING.items():
            education.append(process(context, prefix,df_education[df_education["age_range"]==prefix],df_education_candidates[df_education_candidates["education_type"].isin(education_type)]))
        df_education = pd.concat(education).sort_index()

    boat_user_ids = pd.DataFrame(columns = BOAT_USER_COLUMNS)
    if context.config("generate_outbound_flows"):
        advantage = context.stage("data.od.thonon_boat_zones")
        df_trips = context.stage("synthesis.population.trips")
        df_ports = context.stage("data.gtfs.boat_trips")["ports"]
        df_municipalities = context.stage("data.spatial.municipalities")
        df_od_work, _ = context.stage("data.od.cleaned")
        boat_user_ids = select_boat_users(
            df_persons, df_work, df_trips, advantage, df_od_work, df_municipalities, df_ports, context.config("random_seed")
        )

    return df_work, df_education, boat_user_ids
