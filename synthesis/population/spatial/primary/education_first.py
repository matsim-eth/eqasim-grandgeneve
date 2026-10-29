import numpy as np
import pandas as pd

import synthesis.population.spatial.primary.work_first as work_first
from synthesis.population.spatial.primary.candidates import EDUCATION_MAPPING

"""
Attaches a place of study (commune and education location) to every census
student, before statistical matching (used with education_location_method =
"before_matching", see synthesis.population.matched and
synthesis.population.spatial.primary.candidates), like
synthesis.population.spatial.primary.work_first does for workers.

The commune is drawn from the MOBSCO destinations (data.od.cleaned) of the
student's home commune, its age range and its census place of study
(place_of_study, from ILETUD: home commune, elsewhere in France, abroad),
falling back to fewer of these keys when MOBSCO has no record. Destinations
outside the modelled area are drawn as well (data.od.students_by_origin minus
data.od.cleaned): these students get no place of study (education_outside)
and become cross-perimeter agents without trips in
synthesis.population.enriched.

The location is then drawn among the education locations of that commune
fitting the age range (any education location of the commune when there is
none), which gives the matching attribute: the class of the crow-fly
home-school distance, or abroad.
"""

DESTINATION_KEYS = [
    ["origin_id", "place_of_study", "age_range"],
    ["origin_id", "age_range"],
    ["origin_id", "place_of_study"],
    ["origin_id"],
]

# Upper bounds (m) of the home-place of study distance classes (< 1 km,
# 1-3 km, 3-10 km, > 10 km: 24%, 38%, 26% and 12% of the students in EDGT 74);
# class -1 = not a student, EDUCATION_ABROAD_CLASS = studies abroad (5% in
# EDGT 74, mostly in Geneva)
EDUCATION_DISTANCE_BOUNDARIES = [1e3, 3e3, 10e3]
EDUCATION_ABROAD_CLASS = 4


def configure(context):
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.locations.education")
    context.stage("data.od.cleaned")
    context.stage("data.od.students_by_origin")

    context.config("random_seed")


def education_distance_class(distance, studies, abroad):
    distance_class = np.digitize(np.nan_to_num(distance), EDUCATION_DISTANCE_BOUNDARIES)
    distance_class = np.where(abroad, EDUCATION_ABROAD_CLASS, distance_class)
    return np.where(studies, distance_class, -1)


def age_range(age):
    """Census age -> MOBSCO age range, as in synthesis.population.enriched."""
    return np.select([age <= 10, age <= 14, age <= 17], ["primary_school", "middle_school", "high_school"], "higher_education")


def typed_locations(df_locations, age_range):
    """Education locations for an age range: those of its types, and all
    those of the communes without any of its types."""
    f_typed = df_locations["education_type"].isin(EDUCATION_MAPPING[age_range])
    f_untyped_communes = ~df_locations["commune_id"].isin(set(df_locations.loc[f_typed, "commune_id"]))
    return df_locations[f_typed | f_untyped_communes]


def execute(context):
    random = np.random.default_rng(context.config("random_seed"))

    # Students and their home
    df_students = context.stage("synthesis.population.sampled")
    df_students = df_students[df_students["studies"]][["person_id", "household_id", "age", "place_of_study"]]

    df_homes = context.stage("synthesis.population.spatial.home.locations")
    df_homes = pd.DataFrame({
        "household_id": df_homes["household_id"].values,
        "origin_id": df_homes["commune_id"].astype(str).values,
        "home_x": df_homes.geometry.x.values, "home_y": df_homes.geometry.y.values })

    df_students = df_students.merge(df_homes, on = "household_id").reset_index(drop = True)
    df_students["age_range"] = age_range(df_students["age"].values)
    df_students["place_of_study"] = df_students["place_of_study"].astype(object).fillna("france")

    # Education locations
    df_locations = context.stage("synthesis.locations.education")
    df_locations = pd.DataFrame({
        "location_id": df_locations["location_id"].values, "commune_id": df_locations["commune_id"].astype(str).values,
        "education_type": df_locations["education_type"].astype(str).values,
        "weight": df_locations["weight"].values if "weight" in df_locations else 1.0,
        "x": df_locations.geometry.x.values, "y": df_locations.geometry.y.values })

    # MOBSCO destinations with an education location, by home commune, age
    # range and place of study, plus those outside the modelled area
    _, df_od = context.stage("data.od.cleaned")
    df_od = df_od.astype({ "origin_id": str, "destination_id": str, "age_range": str })
    df_od = df_od[df_od["origin_id"].isin(set(df_students["origin_id"]))]
    df_od["place_of_study"] = np.where(df_od["destination_id"] == df_od["origin_id"], "home_commune",
        np.where(df_od["destination_id"].str.startswith(work_first.CH_PREFIX), "abroad", "france"))

    df_all = context.stage("data.od.students_by_origin").astype({ "origin_id": str, "age_range": str })
    df_od = work_first.add_outside_destinations(df_od, df_all, ["origin_id", "place_of_study", "age_range"])
    df_od = df_od[df_od["destination_id"].isin(set(df_locations["commune_id"]) | { work_first.OUTSIDE })]

    df_students["education_commune_id"] = work_first.assign_communes(random, df_students, df_od, DESTINATION_KEYS, "Education-first")
    df_students["education_outside"] = df_students["education_commune_id"] == work_first.OUTSIDE

    # Location within the commune, of a type fitting the age range
    df_students["education_location_id"] = pd.Series(np.nan, index = df_students.index, dtype = object)

    for range_name in EDUCATION_MAPPING.keys():
        df_range = typed_locations(df_locations, range_name)
        f_range = (df_students["age_range"] == range_name) & ~df_students["education_outside"]

        # Home commune (fallback) without any education location: nearest one
        f_unknown = f_range & ~df_students["education_commune_id"].isin(set(df_range["commune_id"]))
        for index in df_students.index[f_unknown]:
            nearest = np.argmin((df_range["x"].values - df_students.at[index, "home_x"]) ** 2 + (df_range["y"].values - df_students.at[index, "home_y"]) ** 2)
            df_students.at[index, "education_commune_id"] = df_range["commune_id"].values[nearest]
        if f_unknown.any():
            print("Education-first: %d %s students without education location in their commune, nearest commune with one" % (f_unknown.sum(), range_name))

        df_students.loc[f_range, "education_location_id"] = work_first.assign_locations(
            random, df_students.loc[f_range, "education_commune_id"].reset_index(drop = True), df_range).values

    df_students.loc[df_students["education_outside"], "education_commune_id"] = np.nan

    df_xy = df_locations.drop_duplicates("location_id").set_index("location_id")[["x", "y"]].reindex(df_students["education_location_id"].values)
    df_students["education_distance"] = np.sqrt((df_xy["x"].values - df_students["home_x"].values) ** 2 + (df_xy["y"].values - df_students["home_y"].values) ** 2)

    # Outside the modelled area: abroad (other Swiss cantons, other countries)
    # or the longest distance class
    f_inside = ~df_students["education_outside"]
    abroad = np.where(f_inside, df_students["education_commune_id"].fillna("").str.startswith(work_first.CH_PREFIX),
        df_students["place_of_study"] == "abroad")
    df_students["education_distance_class"] = education_distance_class(
        np.where(f_inside, df_students["education_distance"], np.inf), True, abroad)

    print("Education-first: %d students, %.1f%% studying in Switzerland, %.1f%% outside the modelled area" % (
        len(df_students), 100 * (abroad & f_inside).mean(), 100 * df_students["education_outside"].mean()))

    return df_students[["person_id", "education_commune_id", "education_location_id", "education_distance", "education_distance_class", "education_outside"]]
