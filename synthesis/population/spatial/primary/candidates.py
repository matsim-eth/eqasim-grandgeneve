import pandas as pd
import numpy as np


def configure(context):
    context.stage("data.od.weighted")

    context.stage("synthesis.locations.education")
    context.stage("synthesis.locations.work")

    context.stage("synthesis.population.spatial.home.zones")
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.trips")

    context.config("output_path")
    context.config("random_seed")
    context.config("education_location_source", "bpe")

    context.config("work_location_method", "after_matching")
    if context.config("work_location_method") == "before_matching":
        context.stage("synthesis.population.spatial.primary.work_first")

    context.config("education_location_method", "after_matching")
    if context.config("education_location_method") == "before_matching":
        context.stage("synthesis.population.spatial.primary.education_first")
        context.stage("data.od.cleaned")


EDUCATION_MAPPING = {
    "primary_school": ["C1"],
    "middle_school": ["C2"],
    "high_school": ["C3"],
    "higher_education": ["C4", "C5", "C6"]}


def sample_destination_municipalities(context, arguments):
    # Load data
    origin_id, count, random_seed = arguments
    df_od = context.data("df_od")

    # Prepare state
    random = np.random.default_rng(random_seed)
    df_od = df_od[df_od["origin_id"] == origin_id].copy()

    # Sample destinations
    weights = df_od["weight"].values.astype(np.float64) # conversion for multinomial
    weights = weights / np.sum(weights)

    df_od["count"] = random.multinomial(count, weights)
    df_od = df_od[df_od["count"] > 0]

    context.progress.update()
    return df_od[["origin_id", "destination_id", "count"]]


def sample_locations(context, arguments):
    # Load data
    destination_id, random_seed = arguments
    df_locations, df_flow = context.data("df_locations"), context.data("df_flow")

    # Prepare state
    random = np.random.default_rng(random_seed)
    df_locations = df_locations[df_locations["commune_id"] == destination_id]
    
    # Determine demand
    df_flow = df_flow[df_flow["destination_id"] == destination_id]
    count = df_flow["count"].sum()

    # Sample destinations
    weight = np.ones((len(df_locations),)) / len(df_locations)

    if "weight" in df_locations:
        weight = df_locations["weight"].values / df_locations["weight"].sum()
    
    location_counts = random.multinomial(count, weight)
    location_ids = df_locations["location_id"].values
    location_ids = np.repeat(location_ids, location_counts)

    # Shuffle, as otherwise it is likely that *all* copies 
    # of the first location id go to the first origin, and so on
    # note that directly shuffling location_ids gives warning in current pandas
    sorter = np.arange(len(location_ids))
    random.shuffle(sorter)
    location_ids = location_ids[sorter]

    # Construct a data set for all commutes to this zone
    origin_id = np.repeat(df_flow["origin_id"].values, df_flow["count"].values)

    df_result = pd.DataFrame.from_records(dict(
        origin_id = origin_id,
        location_id = location_ids
    ))
    df_result["destination_id"] = destination_id

    return df_result


def sample_locations2(df_locations, df_flow, arguments):
    # Load data
    destination_id, random_seed = arguments

    # Prepare state
    random = np.random.default_rng(random_seed)
    df_locations = df_locations[df_locations["commune_id"] == destination_id]
    
    # Determine demand
    df_flow = df_flow[df_flow["destination_id"] == destination_id]
    count = df_flow["count"].sum()

    # Sample destinations
    weight = np.ones((len(df_locations),)) / len(df_locations)

    if "weight" in df_locations:
        weight = df_locations["weight"].values / df_locations["weight"].sum()
    
    location_counts = random.multinomial(count, weight)
    location_ids = df_locations["location_id"].values
    location_ids = np.repeat(location_ids, location_counts)

    # Shuffle, as otherwise it is likely that *all* copies 
    # of the first location id go to the first origin, and so on
    random.shuffle(location_ids)

    # Construct a data set for all commutes to this zone
    origin_id = np.repeat(df_flow["origin_id"].values, df_flow["count"].values)

    df_result = pd.DataFrame.from_records(dict(
        origin_id = origin_id,
        location_id = location_ids
    ))
    df_result["destination_id"] = destination_id

    return df_result


def process(context, purpose, random, df_persons, df_od, df_locations,step_name):
    df_persons = df_persons[df_persons["has_%s_trip" % purpose]]

    # Sample commute flows based on population
    df_demand = df_persons.groupby("commune_id",observed=False).size().reset_index(name = "count")
    df_demand["random_seed"] = random.integers(0, int(1e6), len(df_demand))
    df_demand = df_demand[["commune_id", "count", "random_seed"]]
    df_demand = df_demand[df_demand["count"] > 0]

    df_flow = []

    with context.progress(label = "Sampling %s municipalities" % step_name, total = len(df_demand)) as progress:
        with context.parallel(dict(df_od = df_od)) as parallel:
            for df_partial in parallel.imap_unordered(sample_destination_municipalities, df_demand.itertuples(index = False, name = None)):
                df_flow.append(df_partial)

    df_flow = pd.concat(df_flow).sort_values(["origin_id", "destination_id"])

    # Sample destinations based on the obtained flows
    unique_ids = df_flow["destination_id"].unique()
    random_seeds = random.integers(0, int(1e6), len(unique_ids))

    df_result = []

    with context.progress(label = "Sampling %s destinations" % purpose, total = len(df_demand)) as progress:
        with context.parallel(dict(df_locations = df_locations, df_flow = df_flow)) as parallel:
            for unique_id, random_seed in zip(unique_ids, random_seeds):
                try:
                    sample_locations2(df_locations, df_flow, (unique_id, random_seed))
                except ValueError as e:
                    print(f"Failed for id={unique_id}, seed={random_seed}: {e}")
                    break
            for df_partial in parallel.imap_unordered(sample_locations, zip(unique_ids, random_seeds)):
                df_result.append(df_partial)

    df_result = pd.concat(df_result).sort_values(["origin_id", "destination_id"])

    return df_result[["origin_id", "destination_id", "location_id"]]


def draw_missing_destinations(random, df_persons, df_od, df_locations, df_od_by_age = None):
    """destination_id, drawn from the OD by home commune (commune_id), for
    the persons that have none yet; only destinations with a location. With
    df_od_by_age (OD with an age_range column), by home commune and age range
    when it has records for them, by home commune from df_od otherwise."""
    f_missing = df_persons["destination_id"].isna()
    destinations_with_location = set(df_locations["commune_id"].astype(str))

    def prepare(df):
        df = df.astype({ "origin_id": str, "destination_id": str })
        return df[df["destination_id"].isin(destinations_with_location) & (df["weight"] > 0)]

    od_groups = prepare(df_od).groupby("origin_id")
    od_age_groups = None if df_od_by_age is None else prepare(df_od_by_age).astype({ "age_range": str }).groupby(["origin_id", "age_range"])
    person_keys = ["commune_id"] if df_od_by_age is None else ["commune_id", "age_range"]

    for key, df_origin in df_persons[f_missing].groupby(person_keys):
        if od_age_groups is not None and key in od_age_groups.groups:
            df_cell = od_age_groups.get_group(key)
        else:
            df_cell = od_groups.get_group(key[0])
        weights = df_cell.groupby("destination_id")["weight"].sum()
        destinations = np.repeat(weights.index.values, random.multinomial(len(df_origin), weights.values / weights.sum()))
        random.shuffle(destinations)
        df_persons.loc[df_origin.index, "destination_id"] = destinations

    return f_missing


def process_work_first(context, random, df_persons, df_od, df_locations):
    """Work candidates with work_location_method = "before_matching": persons
    with a work trip keep the work commune of
    synthesis.population.spatial.primary.work_first; the others (not employed
    in the census, but matched with a working HTS respondent) get one drawn
    from MOBPRO by home commune. Candidate locations are then sampled per
    (home commune, work commune) like in process, and matched to the persons
    by commute distance in synthesis.population.spatial.primary.locations.
    Returns the candidates and the work commune (destination_id) of each
    person."""
    df_persons = df_persons[df_persons["has_work_trip"]][["person_id", "commune_id"]].copy()
    df_persons["commune_id"] = df_persons["commune_id"].astype(str)

    df_work_first = context.stage("synthesis.population.spatial.primary.work_first")
    df_persons = pd.merge(df_persons, df_work_first[["person_id", "work_commune_id"]].rename(
        columns = { "work_commune_id": "destination_id" }), on = "person_id", how = "left")

    f_missing = draw_missing_destinations(random, df_persons, df_od, df_locations)
    print("Work candidates: %d persons with a work trip, %d of them without census workplace (destination drawn by home commune)" % (
        len(df_persons), f_missing.sum()))

    df_flow = df_persons.rename(columns = { "commune_id": "origin_id" }).groupby(["origin_id", "destination_id"]).size().reset_index(name = "count")
    df_flow = df_flow.sort_values(["origin_id", "destination_id"])

    unique_ids = df_flow["destination_id"].unique()
    random_seeds = random.integers(0, int(1e6), len(unique_ids))

    df_result = []

    with context.progress(label = "Sampling work destinations", total = len(unique_ids)) as progress:
        with context.parallel(dict(df_locations = df_locations, df_flow = df_flow)) as parallel:
            for df_partial in parallel.imap_unordered(sample_locations, zip(unique_ids, random_seeds)):
                df_result.append(df_partial)

    df_result = pd.concat(df_result).sort_values(["origin_id", "destination_id"])

    return df_result[["origin_id", "destination_id", "location_id"]], df_persons.set_index("person_id")["destination_id"]


def process_education_first(context, random, df_persons, df_od, df_locations):
    """Education locations with education_location_method =
    "before_matching", one per person (person_id, origin_id, destination_id,
    location_id): persons with an education trip keep the place of study of
    synthesis.population.spatial.primary.education_first, already matched
    with the HTS respondent by distance class; the others (not students in
    the census, but matched with an HTS respondent having an education trip)
    get a commune drawn from MOBSCO by home commune and age range, and a
    location fitting their age range in it."""
    # Imported here, as education_first imports EDUCATION_MAPPING from this module
    import synthesis.population.spatial.primary.education_first as education_first
    import synthesis.population.spatial.primary.work_first as work_first

    df_persons = df_persons[df_persons["has_education_trip"]][["person_id", "commune_id", "age_range"]].copy()
    df_persons["commune_id"] = df_persons["commune_id"].astype(str)

    df_education_first = context.stage("synthesis.population.spatial.primary.education_first")
    df_persons = pd.merge(df_persons, df_education_first[["person_id", "education_commune_id", "education_location_id"]].rename(
        columns = { "education_commune_id": "destination_id", "education_location_id": "location_id" }), on = "person_id", how = "left")

    _, df_od_by_age = context.stage("data.od.cleaned")
    df_persons["age_range"] = df_persons["age_range"].astype(str)
    f_missing = draw_missing_destinations(random, df_persons, df_od, df_locations, df_od_by_age)

    df_locations = df_locations.assign(commune_id = df_locations["commune_id"].astype(str), education_type = df_locations["education_type"].astype(str))
    for range_name in EDUCATION_MAPPING.keys():
        f_range = f_missing & (df_persons["age_range"] == range_name)
        if f_range.any():
            df_persons.loc[f_range, "location_id"] = work_first.assign_locations(
                random, df_persons.loc[f_range, "destination_id"].reset_index(drop = True),
                education_first.typed_locations(df_locations, range_name)).values

    print("Education candidates: %d persons with an education trip, %d of them without census place of study (drawn by home commune)" % (
        len(df_persons), f_missing.sum()))

    df_persons = df_persons.rename(columns = { "commune_id": "origin_id" })
    return df_persons[["person_id", "origin_id", "destination_id", "location_id"]], df_persons.set_index("person_id")["destination_id"]


def execute(context):
    # Prepare population data
    df_persons = context.stage("synthesis.population.enriched")[["person_id", "household_id", "age_range"]].copy()
    df_trips = context.stage("synthesis.population.trips")

    df_persons["has_work_trip"] = df_persons["person_id"].isin(df_trips[
        (df_trips["following_purpose"] == "work") | (df_trips["preceding_purpose"] == "work")
    ]["person_id"])
    
    df_persons["has_education_trip"] = df_persons["person_id"].isin(df_trips[
        (df_trips["following_purpose"] == "education") | (df_trips["preceding_purpose"] == "education")
    ]["person_id"])

    df_homes = context.stage("synthesis.population.spatial.home.zones")
    df_persons = pd.merge(df_persons, df_homes, on = "household_id")

    # Prepare spatial data
    df_work_od, df_education_od = context.stage("data.od.weighted")

    # Sampling
    random = np.random.default_rng(context.config("random_seed"))

    df_locations = context.stage("synthesis.locations.work")
    df_locations["weight"] = df_locations["employees"]

    if context.config("work_location_method") == "before_matching":
        df_work, work_destination = process_work_first(context, random, df_persons, df_work_od, df_locations)
        df_persons["work_destination_id"] = df_persons["person_id"].map(work_destination)
    else:
        df_work = process(context, "work", random, df_persons,
            df_work_od, df_locations, "work"
        )

    df_locations = context.stage("synthesis.locations.education")
    if context.config("education_location_method") == "before_matching":
        df_education, education_destination = process_education_first(context, random, df_persons, df_education_od, df_locations)
        df_persons["education_destination_id"] = df_persons["person_id"].map(education_destination)
    elif context.config("education_location_source") == 'bpe':
        df_education = process(context, "education", random, df_persons, df_education_od, df_locations,"education")
    else :
        df_education = []
        for prefix, education_type in EDUCATION_MAPPING.items():
            df_education.append(
                process(context, "education", random,
                    df_persons[df_persons["age_range"]==prefix],
                    df_education_od[df_education_od["age_range"]==prefix],df_locations[df_locations["education_type"].isin(education_type)],prefix)
            )
        df_education = pd.concat(df_education)

    person_columns = ["person_id", "household_id", "age_range", "commune_id", "has_work_trip", "has_education_trip"]
    for column in ["work_destination_id", "education_destination_id"]:
        if column in df_persons:
            person_columns.append(column)

    return dict(
        work_candidates = df_work,
        education_candidates = df_education,
        persons = df_persons[df_persons["has_work_trip"] | df_persons["has_education_trip"]][person_columns]
    )
