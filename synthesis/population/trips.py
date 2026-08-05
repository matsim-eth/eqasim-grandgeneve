import numpy as np
import pandas as pd

"""
This stage duplicates trips and attaches them to the synthetic population.
"""

def configure(context):
    context.stage("synthesis.population.matched")
    context.stage("synthesis.population.enriched")
    context.config("random_seed")
    context.config("with_motorcycles", False)

    hts = context.config("hts")
    context.stage("data.hts.selected", alias = "hts")

def execute(context):
    # Load data
    df_trips = context.stage("hts")[2]

    # Duplicate with synthetic persons
    df_matching = context.stage("synthesis.population.matched")
    df_trips = df_trips.rename(columns = { "person_id": "hts_id" })
    df_trips = pd.merge(df_matching, df_trips, on = "hts_id")

    df_enriched = context.stage("synthesis.population.enriched")

    # Cross-perimeter agents get a single "cross_perimeter" activity and no
    # trips (see synthesis.population.activities), so their HTS-derived trip
    # chain is dropped here rather than generated and removed downstream.
    if "is_crossperim_person" in df_enriched:
        cross_perimeter_ids = df_enriched.loc[df_enriched["is_crossperim_person"], "person_id"]
        df_trips = df_trips[~df_trips["person_id"].isin(cross_perimeter_ids)]

    if context.config("with_motorcycles"):
        df_population = df_enriched[["person_id", "use_motorcycle"]]
        df_trips["mode"] = df_trips["mode"].cat.add_categories("motorcycle")
        df_trips = pd.merge(df_population, df_trips, on = "person_id")

    df_trips = df_trips.sort_values(by = ["person_id", "trip_id"])

    # Define trip index
    df_count = df_trips.groupby("person_id").size().reset_index(name = "count")
    df_trips["trip_index"] = np.hstack([np.arange(count) for count in df_count["count"].values])
    df_trips = df_trips.sort_values(by = ["person_id", "trip_index"])

    # Diversify departure times
    random = np.random.default_rng(context.config("random_seed"))
    counts = df_trips[["person_id"]].groupby("person_id").size().reset_index(name = "count")["count"].values

    interval = df_trips[["person_id", "departure_time"]].groupby("person_id").min().reset_index()["departure_time"].values
    interval = np.minimum(1800.0, interval) # If first departure time is just 5min after midnight, we only add a deviation of 5min

    offset = random.random(size = (len(counts), )) * interval * 2.0 - interval
    offset = np.repeat(offset, counts)

    df_trips["departure_time"] += offset
    df_trips["arrival_time"] += offset
    df_trips["departure_time"] = np.round(df_trips["departure_time"])
    df_trips["arrival_time"] = np.round(df_trips["arrival_time"])

    assert (df_trips["departure_time"] >= 0.0).all()
    assert (df_trips["arrival_time"] >= 0.0).all()

    if "use_motorcycle" in df_trips:
        df_trips.loc[df_trips["use_motorcycle"], "mode"] = "motorcycle"

    return df_trips[[
        "person_id", "trip_index",
        "departure_time", "arrival_time",
        "preceding_purpose", "following_purpose",
        "is_first_trip", "is_last_trip",
        "trip_duration", "activity_duration",
        "mode"
    ]]
