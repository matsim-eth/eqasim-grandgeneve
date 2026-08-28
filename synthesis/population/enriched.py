import numpy as np
import pandas as pd

import data.hts.edgt_74.adisp_merge.merge as edgt74_merge

"""
This stage fuses census data with HTS data.
"""

def configure(context):
    context.config("with_motorcycles", False)

    context.stage("synthesis.population.matched")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.income.selected")
    context.config("extra_enriched_attributes", [])

    context.stage("data.hts.selected", alias = "hts")

    if context.config("hts") == "edgt_74":
        context.config("edgt74_version", default = "adisp")

        if context.config("edgt74_version") == "adisp":
            # Inputs needed to compute synthesis.population.cross_perimeter's
            # features from the real synthesized population (home location,
            # census area) rather than by reusing the HTS respondent's own
            # values.
            context.stage("data.hts.edgt_74.adisp_merge.zones")
            context.stage("data.spatial.ch.cantons")
            context.stage("synthesis.population.spatial.home.locations")
            context.stage("synthesis.population.cross_perimeter")

            context.config("random_seed")


def execute(context):
    is_edgt74_adisp = context.config("hts") == "edgt_74" and context.config("edgt74_version") == "adisp"

    # Select population columns
    population_columns = [
        "person_id", "household_id",
        "census_person_id", "census_household_id",
        "age", "sex", "employed", "studies",
        "number_of_cars", "number_of_motorcycles", "number_of_vehicles", "use_motorcycle",
        "household_size", "consumption_units",
        "socioprofessional_class"
    ]

    if is_edgt74_adisp:
        # edgt_area is assigned spatially (data.spatial.iris) from the real
        # home IRIS, not learnt from the HTS
        population_columns += ["edgt_area"]

    df_population = context.stage("synthesis.population.sampled")[population_columns]

    # Attach matching information
    df_matching   = context.stage("synthesis.population.matched")
    df_population = pd.merge(df_population, df_matching, on = "person_id")

    initial_size          = len(df_population)
    initial_person_ids    = len(df_population["person_id"].unique())
    initial_household_ids = len(df_population["household_id"].unique())

    # Attach person and household attributes from HTS
    df_hts_households, df_hts_persons, _ = context.stage("hts")
    df_hts_persons    = df_hts_persons.rename(columns = { "person_id": "hts_id", "household_id": "hts_household_id" })
    df_hts_households = df_hts_households.rename(columns = { "household_id": "hts_household_id" })

    columns    = ["hts_id", "hts_household_id", "has_license", "has_pt_subscription", "is_passenger"]

    if is_edgt74_adisp:
        # Highest education attained (P8): no census equivalent, so unlike
        # is_annemasse/log_dist_perimeter below it can only come from the
        # matched HTS respondent.
        columns += ["P8"]

    extra_cols = context.config("extra_enriched_attributes")

    assert isinstance(extra_cols, list), "`extra_enriched_attributes` parameter must be a list"
    columns += extra_cols

    df_population = pd.merge(df_population, df_hts_persons[columns], on="hts_id")
    df_population = pd.merge(df_population, df_hts_households[["hts_household_id", "number_of_bikes"]], on = "hts_household_id")

    # Statistical matching is not guaranteed to respect age when pairing a
    # synthetic person with an HTS respondent (age_class buckets are coarse,
    # and sparse combinations can fall back to unconstrained matching), so a
    # minor can inherit an adult donor's license. Enforce it directly.
    underage_selector = df_population["age"] < 18
    n_underage_licensed = int((underage_selector & df_population["has_license"]).sum())
    print(f"Identified {n_underage_licensed} agents under 18 years but having a driving license.")
    print("This is due to statistical matching - those agents were not matched using the age variable.")
    print("Fixing this to ensure consistency of the results.")
    df_population.loc[underage_selector, "has_license"] = False

    # Attach income
    df_income     = context.stage("synthesis.population.income.selected")
    df_population = pd.merge(df_population, df_income[["household_id", "household_income"]], on = "household_id")

    if is_edgt74_adisp:
        # Distance from the real (synthesized) home location to the survey
        # perimeter border, instead of reusing the HTS respondent's own value
        df_zones, _, _ = context.stage("data.hts.edgt_74.adisp_merge.zones")
        df_cantons     = context.stage("data.spatial.ch.cantons")
        df_homes       = context.stage("synthesis.population.spatial.home.locations")

        perimeter = edgt74_merge.build_perimeter_geometry(df_zones, df_cantons)
        df_homes  = df_homes.to_crs(df_zones.crs)

        df_distance = pd.DataFrame({
            "household_id": df_homes["household_id"].values,
            "distance_to_perimeter_border": df_homes.geometry.distance(perimeter.boundary).values
        })

        df_population = pd.merge(df_population, df_distance, on = "household_id")

    # Check consistency
    final_size          = len(df_population)
    final_person_ids    = len(df_population["person_id"].unique())
    final_household_ids = len(df_population["household_id"].unique())

    print(initial_size, final_size)

    assert initial_size == final_size
    assert initial_person_ids == final_person_ids
    assert initial_household_ids == final_household_ids

    # Add car availability
    df_number_of_cars = df_population[["household_id", "number_of_cars"]].drop_duplicates("household_id")
    df_number_of_licenses = df_population[["household_id", "has_license"]].groupby("household_id").sum().reset_index().rename(columns = { "has_license": "number_of_licenses" })
    df_car_availability = pd.merge(df_number_of_cars, df_number_of_licenses)

    df_car_availability["car_availability"] = "all"
    df_car_availability.loc[df_car_availability["number_of_cars"] < df_car_availability["number_of_licenses"], "car_availability"] = "some"
    df_car_availability.loc[df_car_availability["number_of_cars"] == 0, "car_availability"] = "none"
    df_car_availability["car_availability"] = df_car_availability["car_availability"].astype("category")

    df_population = pd.merge(df_population, df_car_availability[["household_id", "car_availability"]])

    # Handle motorcycle use if needed (remove use_motorcycle)
    if not context.config("with_motorcycles"):
        df_population.drop(columns=["use_motorcycle"])

    # Add bike availability
    df_population["bike_availability"] = "all"
    df_population.loc[df_population["number_of_bikes"] < df_population["household_size"], "bike_availability"] = "some"
    df_population.loc[df_population["number_of_bikes"] == 0, "bike_availability"] = "none"
    df_population["bike_availability"] = df_population["bike_availability"].astype("category")
    
    # Add age range for education
    df_population["age_range"] = "higher_education"
    df_population.loc[df_population["age"]<=10,"age_range"] = "primary_school"
    df_population.loc[df_population["age"].between(11,14),"age_range"] = "middle_school"
    df_population.loc[df_population["age"].between(15,17),"age_range"] = "high_school"
    df_population["age_range"] = df_population["age_range"].astype("category")

    if is_edgt74_adisp:
        # synthesis.population.cross_perimeter features: is_annemasse and
        # log_dist_perimeter come from the real synthesized population above,
        # has_driving_license and educ_ord have no census equivalent and are
        # carried over from the matched HTS respondent.
        df_population["is_annemasse"]        = (df_population["edgt_area"] == "annemasse").astype(int)
        df_population["log_dist_perimeter"]  = np.log1p(df_population["distance_to_perimeter_border"])
        df_population["has_driving_license"] = df_population["has_license"].astype(int)
        df_population["educ_ord"]            = pd.to_numeric(df_population["P8"], errors = "coerce").map(edgt74_merge.EDUCATION_ORDINAL_MAP)

        # Score each synthesized person with the fitted cross-perimeter logit
        # (synthesis.population.cross_perimeter) and draw is_crossperim_person
        # from the resulting probability, rather than carrying over the HTS
        # respondent's own observed value.
        _, _, params = context.stage("synthesis.population.cross_perimeter")
        features = [feature for feature in params.index if feature != "const"]

        logit       = params["const"] + df_population[features].values.astype(float) @ params[features].values
        probability = 1.0 / (1.0 + np.exp(-logit))

        random = np.random.default_rng(context.config("random_seed"))
        df_population["is_crossperim_person"] = random.random(len(df_population)) < probability

    return df_population
