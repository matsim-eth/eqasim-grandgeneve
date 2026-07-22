import pandas as pd
import numpy as np

"""
Transforms the synthetic trip table into a synthetic activity table.
"""

def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.trips")

def execute(context):
    df_activities = context.stage("synthesis.population.trips")

    # Add trip count
    counts = df_activities.groupby("person_id").size().reset_index(name = "trip_count")["trip_count"].values
    df_activities["trip_count"] = np.hstack([[count] * count for count in counts])

    # Shift times and types of trips to arrive at activities
    df_activities["purpose"] = df_activities["preceding_purpose"]
    df_activities["end_time"] = df_activities["departure_time"]

    df_activities["start_time"] = df_activities.shift(1)["arrival_time"]
    df_activities.loc[df_activities["is_first_trip"], "start_time"] = np.nan

    df_activities["is_first"] = df_activities["is_first_trip"]
    df_activities["is_last"] = False

    df_activities["activity_index"] = df_activities["trip_index"]

    # Add missing end activity
    df_last = df_activities[df_activities["is_last_trip"]].copy()
    df_last["purpose"] = df_activities["following_purpose"]

    df_last["start_time"] = df_activities["arrival_time"]
    df_last["end_time"] = np.nan

    df_last["is_first"] = False
    df_last["is_last"] = True

    df_last["activity_index"] = df_last["trip_count"]
    df_last["trip_index"] = -1

    df_activities = pd.concat([
        df_activities[["person_id", "activity_index", "trip_index", "purpose", "start_time", "end_time", "is_first", "is_last"]],
        df_last[["person_id", "activity_index", "trip_index", "purpose", "start_time", "end_time", "is_first", "is_last"]]
    ]).sort_values(by = ["person_id", "activity_index"])

    # Add activities for people without trips
    df_enriched = context.stage("synthesis.population.enriched")
    has_crossperim = "is_crossperim_person" in df_enriched

    columns = ["person_id"] + (["is_crossperim_person"] if has_crossperim else [])
    df_missing = df_enriched[~df_enriched["person_id"].isin(df_activities["person_id"])][columns].copy()

    df_missing["activity_index"] = 0
    df_missing["trip_index"] = -1
    df_missing["purpose"] = "home"

    if has_crossperim:
        # Cross-perimeter agents (see synthesis.population.trips, which drops
        # their trips) get a single "cross_perimeter" activity instead of
        # "home", and no trips at all.
        df_missing.loc[df_missing["is_crossperim_person"], "purpose"] = "cross_perimeter"
        df_missing = df_missing.drop(columns = ["is_crossperim_person"])

    df_missing["start_time"] = np.nan
    df_missing["end_time"] = np.nan
    df_missing["is_first"] = True
    df_missing["is_last"] = True

    # "cross_perimeter" is not among the HTS-derived purpose categories on
    # df_activities, so cast through plain strings rather than reusing its
    # categorical dtype directly (which would silently turn it into NaN).
    df_activities["purpose"] = df_activities["purpose"].astype(str)
    df_missing["purpose"] = df_missing["purpose"].astype(str)
    df_activities = pd.concat([df_activities, df_missing])
    df_activities["purpose"] = df_activities["purpose"].astype("category")

    # Some cleanup
    df_activities["duration"] = df_activities["end_time"] - df_activities["start_time"]

    return df_activities
