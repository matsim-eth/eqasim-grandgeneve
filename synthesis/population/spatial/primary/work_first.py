import numpy as np
import pandas as pd

"""
Attaches a work commune and a workplace to every employed census agent,
before statistical matching (used with work_location_method =
"before_matching", see synthesis.population.matched and
synthesis.population.spatial.primary.candidates).

The work commune is drawn from the MOBPRO destinations (data.od.cleaned) of
the agent's home commune, its census commute mode (TRANS) and its census place
of work (place_of_work, from ILT: home commune, elsewhere in France, abroad).
When MOBPRO has no record for that combination, the draw falls back to the
home commune and place of work, then to the home commune and commute mode,
then to the home commune only, and finally to the home commune itself.
Destinations outside the modelled area are drawn as well (MOBPRO workers of
data.od.workers_by_origin minus those of data.od.cleaned): these workers get
no work commune nor workplace (work_outside), and are made cross-perimeter
agents without trips in synthesis.population.enriched.

The workplace is then drawn among the work locations of that commune,
weighted by employees. It gives the attributes used for matching: whether the
workplace is in Switzerland and the class of the crow-fly home-workplace
distance.
"""

# Census commute_mode -> data.od.cleaned commute_mode (motorcycles count as
# car there); employed agents without a mode work from home
CENSUS_TO_MOBPRO_MODE = { "walk": "walk", "bike": "bike", "motorcycle": "car", "car": "car", "pt": "pt" }
NO_TRANSPORT = "no transport"

DESTINATION_KEYS = [
    ["origin_id", "place_of_work", "mobpro_mode"],
    ["origin_id", "place_of_work"],
    ["origin_id", "mobpro_mode"],
    ["origin_id"],
]

# Upper bounds (m) of the home-workplace distance classes (< 5 km, 5-10 km,
# 10-20 km, > 20 km: 34%, 20%, 20% and 17% of the employed in EDGT 74);
# class -1 = not employed, class -2 = works at home (census: no transport,
# HTS: usual workplace at 0 m), kept apart because these workers travel to
# work on far fewer days (44% vs about 80% in EDGT 74)
WORK_DISTANCE_BOUNDARIES = [5e3, 10e3, 20e3]

CH_PREFIX = "CH"
OUTSIDE = "outside"

COMMUTE_MODE_CLASSES = { "car": "car", "motorcycle": "car", "pt": "pt", "walk": "active", "bike": "active" }


def configure(context):
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.locations.work")
    context.stage("data.od.cleaned")
    context.stage("data.od.workers_by_origin")

    context.config("random_seed")


def work_distance_class(distance, employed, at_home):
    distance_class = np.digitize(np.nan_to_num(distance), WORK_DISTANCE_BOUNDARIES)
    distance_class = np.where(at_home, -2, distance_class)
    return np.where(employed, distance_class, -1)


def sample_destinations(random, df_persons, df_od, keys):
    """Multinomial draw of a destination_id for each person, per group of
    `keys` (same procedure as synthesis.population.spatial.primary.candidates).
    Persons whose group has no MOBPRO weight are left as NaN."""
    destinations = pd.Series(np.nan, index = df_persons.index, dtype = object)
    df_od = df_od[df_od["weight"] > 0]
    od_groups = df_od.groupby(keys).indices

    for key, person_indices in df_persons.groupby(keys).indices.items():
        od_indices = od_groups.get(key)
        if od_indices is None:
            continue

        weights = df_od["weight"].values[od_indices].astype(float)
        counts = random.multinomial(len(person_indices), weights / weights.sum())
        drawn = np.repeat(df_od["destination_id"].values[od_indices], counts)
        random.shuffle(drawn)
        destinations.iloc[person_indices] = drawn

    return destinations


def assign_communes(random, df_persons, df_od, destination_keys, label):
    """destination_id for each person, by the first destination_keys that
    have OD records, then the home commune (also used by
    synthesis.population.spatial.primary.education_first)."""
    destination_id = pd.Series(np.nan, index = df_persons.index, dtype = object)

    for keys in destination_keys:
        f_missing = destination_id.isna()
        if not f_missing.any():
            break

        df_cells = df_od.groupby(["origin_id", "destination_id"] + keys[1:], as_index = False)["weight"].sum()
        destination_id.loc[f_missing] = sample_destinations(random, df_persons[f_missing].reset_index(drop = True), df_cells, keys).values
        print("%s: %d / %d persons drawn by %s" % (label, (f_missing & destination_id.notna()).sum(), len(df_persons), keys))

    f_missing = destination_id.isna()
    destination_id.loc[f_missing] = df_persons.loc[f_missing, "origin_id"]
    print("%s: %d persons without any OD record for their home commune, staying there" % (label, f_missing.sum()))

    return destination_id


def add_outside_destinations(df_od, df_all, keys):
    """Adds, per cell of keys (origin_id first), the OD weight of
    destinations outside the modelled area (df_all: all destinations, minus
    those of df_od) as an extra OUTSIDE destination."""
    df_outside = df_all.merge(df_od.groupby(keys, as_index = False)["weight"].sum(),
        on = keys, how = "left", suffixes = ("", "_inside"))
    df_outside["weight"] = (df_outside["weight"] - df_outside["weight_inside"].fillna(0.0)).clip(lower = 0.0)
    df_outside = df_outside[df_outside["origin_id"].isin(set(df_od["origin_id"])) & (df_outside["weight"] > 0.1)]
    df_outside["destination_id"] = OUTSIDE
    return pd.concat([df_od, df_outside[df_od.columns.intersection(df_outside.columns)]], ignore_index = True)


def assign_locations(random, destination_id, df_locations):
    """A location per person in its destination commune, by location weight."""
    location_id = pd.Series(np.nan, index = destination_id.index, dtype = object)
    location_groups = df_locations.groupby("commune_id").indices

    for commune_id, person_indices in destination_id.groupby(destination_id).indices.items():
        indices = location_groups[commune_id]
        weights = df_locations["weight"].values[indices].astype(float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(indices)) / len(indices)
        location_id.iloc[person_indices] = df_locations["location_id"].values[random.choice(indices, size = len(person_indices), p = weights)]

    return location_id


def commute_mode_class(commute_mode, employed):
    """Census commute mode as matched with the HTS usual commute mode
    (data.hts.edgt_74.adisp_merge.merge): car (incl. motorcycles), pt,
    active (walk, bike), home (no transport) or none (not employed)."""
    mode_class = pd.Series(commute_mode).astype(object).map(COMMUTE_MODE_CLASSES).fillna("home").values
    return np.where(employed, mode_class, "none")


def execute(context):
    random = np.random.default_rng(context.config("random_seed"))

    # Employed agents and their home
    df_workers = context.stage("synthesis.population.sampled")
    df_workers = df_workers[df_workers["employed"]][["person_id", "household_id", "commute_mode", "place_of_work"]]

    df_homes = context.stage("synthesis.population.spatial.home.locations")
    df_homes = pd.DataFrame({
        "household_id": df_homes["household_id"].values,
        "origin_id": df_homes["commune_id"].astype(str).values,
        "home_x": df_homes.geometry.x.values, "home_y": df_homes.geometry.y.values })

    df_workers = df_workers.merge(df_homes, on = "household_id").reset_index(drop = True)
    df_workers["mobpro_mode"] = df_workers["commute_mode"].astype(object).map(CENSUS_TO_MOBPRO_MODE).fillna(NO_TRANSPORT)
    df_workers["place_of_work"] = df_workers["place_of_work"].astype(object).fillna("france")

    # MOBPRO destinations by home commune, commute mode and place of work
    df_od, _ = context.stage("data.od.cleaned")
    df_od = df_od.astype({ "origin_id": str, "destination_id": str, "commute_mode": str }).rename(columns = { "commute_mode": "mobpro_mode" })
    df_od = df_od[df_od["origin_id"].isin(set(df_workers["origin_id"]))]
    df_od["place_of_work"] = np.where(df_od["destination_id"] == df_od["origin_id"], "home_commune",
        np.where(df_od["destination_id"].str.startswith(CH_PREFIX), "abroad", "france"))

    # Workers of each cell working outside the modelled area, as an extra destination
    df_all = context.stage("data.od.workers_by_origin").astype({ "origin_id": str, "commute_mode": str }).rename(columns = { "commute_mode": "mobpro_mode" })
    df_od = add_outside_destinations(df_od, df_all, ["origin_id", "place_of_work", "mobpro_mode"])

    df_workers["work_commune_id"] = assign_communes(random, df_workers, df_od, DESTINATION_KEYS, "Work-first")
    df_workers["work_outside"] = df_workers["work_commune_id"] == OUTSIDE

    # Workplace within the commune
    df_locations = context.stage("synthesis.locations.work")
    df_locations = pd.DataFrame({
        "location_id": df_locations["location_id"].values, "commune_id": df_locations["commune_id"].astype(str).values,
        "weight": df_locations["employees"].values, "x": df_locations.geometry.x.values, "y": df_locations.geometry.y.values })

    f_unknown = ~df_workers["work_commune_id"].isin(set(df_locations["commune_id"])) & ~df_workers["work_outside"]
    if f_unknown.any():
        print("Work-first: %d workers drawn to communes without work location, working in their home commune" % f_unknown.sum())
        df_workers.loc[f_unknown, "work_commune_id"] = df_workers.loc[f_unknown, "origin_id"]

    f_inside = ~df_workers["work_outside"]
    df_workers["work_location_id"] = pd.Series(np.nan, index = df_workers.index, dtype = object)
    df_workers.loc[f_inside, "work_location_id"] = assign_locations(random, df_workers.loc[f_inside, "work_commune_id"].reset_index(drop = True), df_locations).values
    df_workers.loc[~f_inside, "work_commune_id"] = np.nan

    df_xy = df_locations.set_index("location_id")[["x", "y"]].reindex(df_workers["work_location_id"].values)
    df_workers["work_distance"] = np.sqrt((df_xy["x"].values - df_workers["home_x"].values) ** 2 + (df_xy["y"].values - df_workers["home_y"].values) ** 2)

    # Outside the modelled area: abroad (other Swiss cantons, other countries)
    # counts as cross-border, and the distance is taken as the longest class
    df_workers["work_cross_border"] = np.where(f_inside, df_workers["work_commune_id"].fillna("").str.startswith(CH_PREFIX),
        df_workers["place_of_work"] == "abroad")
    df_workers["work_distance_class"] = work_distance_class(
        np.where(f_inside, df_workers["work_distance"], np.inf), True, df_workers["mobpro_mode"] == NO_TRANSPORT)

    print("Work-first: %d workers, %.1f%% working in Switzerland, %.1f%% outside the modelled area" % (
        len(df_workers), 100 * (df_workers["work_cross_border"] & f_inside).mean(), 100 * df_workers["work_outside"].mean()))

    return df_workers[["person_id", "work_commune_id", "work_location_id", "work_distance", "work_cross_border", "work_distance_class", "work_outside"]]
