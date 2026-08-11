import pandas as pd
import geopandas as gpd
from data.sirene.density import impute_parallel as impute_sirene
from data.spatial.population_density import impute_parallel as impute_population
from data.spatial.ovgk import impute_parallel as impute_ovgk

EMPLOYEES_DENSITY_RADIUS = 500
POPULATION_DENSITY_RADIUS = 500

LOOP_DISTANCE_THRESHOLD = 20.0

def configure(context):
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")
    context.stage("synthesis.population.spatial.secondary.locations")

    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.trips_boat")

    context.stage("data.spatial.iris")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")
    context.stage("data.sirene.density")
    context.stage("data.spatial.population_density")
    context.stage("data.spatial.ovgk")

    context.config("processes", volatile = True)

    context.config("generate_outbound_flows")
    if context.config("generate_outbound_flows"):
        context.stage("data.gtfs.boat_trips")

def tag_short_assigned_trips(df_locations, df_trips, threshold = LOOP_DISTANCE_THRESHOLD):
    """Some trips only turn out to be very short once secondary locations
    have actually been assigned (e.g. a sampled destination snapping to a
    facility close to the origin), even if they weren't already tagged as
    "_loop" by the HTS/survey stage. Catch those here the same way
    data.hts.edgt_74.adisp_merge.merge.tag_short_trip_loop_mode does for the
    survey data: by suffixing their mode with "_loop"."""
    df_distances = df_locations[["person_id", "activity_index", "geometry"]].rename(
        columns = { "activity_index": "trip_index" }
    ).sort_values(by = ["person_id", "trip_index"])
    df_distances["euclidean_distance"] = df_distances["geometry"].distance(df_distances["geometry"].shift(-1))

    df_trips = df_trips.merge(
        df_distances[["person_id", "trip_index", "euclidean_distance"]],
        on = ["person_id", "trip_index"], how = "left"
    )

    df_trips["mode"] = df_trips["mode"].astype(str)
    already_tagged = df_trips["mode"].str.endswith("_loop")
    newly_short = (df_trips["euclidean_distance"] < threshold) & ~already_tagged

    n = len(df_trips)
    print("Trips newly tagged as _loop after secondary location assignment (euclidean distance < %.0f m): %d / %d (%.2f%%)" % (
        threshold, newly_short.sum(), n, 100 * newly_short.sum() / n
    ))

    df_trips.loc[newly_short, "mode"] += "_loop"
    df_trips["mode"] = df_trips["mode"].astype("category")

    return df_trips.drop(columns = ["euclidean_distance"])

def execute(context):
    df_home = context.stage("synthesis.population.spatial.home.locations")
    df_work, df_education, df_boat_users = context.stage("synthesis.population.spatial.primary.locations")
    df_secondary = context.stage("synthesis.population.spatial.secondary.locations")[0]

    df_persons = context.stage("synthesis.population.sampled")[["person_id", "household_id"]]
    df_locations = context.stage("synthesis.population.activities")[["person_id", "activity_index", "purpose"]]

    # Home locations
    # ("cross_perimeter" activities also happen at the agent's home: these
    # agents are dropped from the trip chain entirely - see
    # synthesis.population.trips / synthesis.population.activities - so
    # there is no secondary-location candidate to place them at.)
    df_home_locations = df_locations[df_locations["purpose"].isin(("home", "cross_perimeter"))]
    df_home_locations = pd.merge(df_home_locations, df_persons, on = "person_id")
    df_home_locations = pd.merge(df_home_locations, df_home[["household_id", "geometry"]], on = "household_id")
    df_home_locations["location_id"] = -1
    df_home_locations = df_home_locations[["person_id", "activity_index", "location_id", "geometry"]]

    # Boat commuters' "home" activity is relocated to their departure
    # harbour (see synthesis.population.trips_boat) rather than their real
    # dwelling; "work" keeps its normally-assigned location.
    if context.config("generate_outbound_flows") and len(df_boat_users) > 0:
        df_ports = context.stage("data.gtfs.boat_trips")["ports"]
        home_port_geometry = df_boat_users.set_index("person_id")["home_port"].map(
            df_ports.set_index("port")["geometry"]
        )

        is_boat_home = df_home_locations["person_id"].isin(home_port_geometry.index)
        df_home_locations.loc[is_boat_home, "geometry"] = df_home_locations.loc[is_boat_home, "person_id"].map(home_port_geometry).values

    # Work locations
    df_work_locations = df_locations[df_locations["purpose"] == "work"]
    df_work_locations = pd.merge(df_work_locations, df_work[["person_id", "location_id", "geometry"]], on = "person_id")
    df_work_locations = df_work_locations[["person_id", "activity_index", "location_id", "geometry"]]
    assert not df_work_locations["geometry"].isna().any()

    # Education locations
    df_education_locations = df_locations[df_locations["purpose"] == "education"]
    df_education_locations = pd.merge(df_education_locations, df_education[["person_id", "location_id", "geometry"]], on = "person_id")
    df_education_locations = df_education_locations[["person_id", "activity_index", "location_id", "geometry"]]
    assert not df_education_locations["geometry"].isna().any()

    # Secondary locations
    df_secondary_locations = df_locations[~df_locations["purpose"].isin(("home", "cross_perimeter", "work", "education"))].copy()
    df_secondary_locations = pd.merge(df_secondary_locations, df_secondary[[
        "person_id", "activity_index", "location_id", "geometry"
    ]], on = ["person_id", "activity_index"], how = "left")
    df_secondary_locations = df_secondary_locations[["person_id", "activity_index", "location_id", "geometry"]]
    assert not df_secondary_locations["geometry"].isna().any()

    # Validation
    initial_count = len(df_locations)
    df_locations = pd.concat([df_home_locations, df_work_locations, df_education_locations, df_secondary_locations])

    df_locations = df_locations.sort_values(by = ["person_id", "activity_index"])
    final_count = len(df_locations)

    assert initial_count == final_count

    assert not df_locations["geometry"].isna().any()
    df_locations = gpd.GeoDataFrame(df_locations, crs = df_home.crs)

    # add municipalities
    df_iris = context.stage("data.spatial.iris")
    df_iris = gpd.GeoDataFrame(df_iris, crs = df_home.crs)

    df_locations = gpd.sjoin(df_locations, df_iris, how = "left")
    df_locations = df_locations.drop(columns = ["index_right"])

    # Add municipality types
    # (municipality_types' own commune_id uses a different, FR/CH-prefixed
    # format from the IRIS-derived one above, so it is renamed here and only
    # used below as a fallback for locations outside of the IRIS zoning)
    df_muntypes = context.stage("data.spatial.municipality_types")[["municipality_type", "commune_id", "geometry"]]
    df_muntypes = df_muntypes.rename(columns = { "commune_id": "muntype_commune_id" })
    df_muntypes = gpd.GeoDataFrame(df_muntypes, crs = df_home.crs)

    df_locations = gpd.sjoin(df_locations, df_muntypes, how = "left")
    df_locations = df_locations.drop(columns = ["index_right"])

    # IRIS only covers France, so locations outside of it (e.g. in
    # Switzerland, when generate_outbound_flows is enabled) get no
    # iris_id / commune_id / departement_id / region_id from the join above.
    # Backfill them using the municipality_types zoning, which also covers
    # Switzerland, following the same "CH" convention used for cross-border
    # zones elsewhere in the pipeline (see data.spatial.codes).
    for column in ["iris_id", "commune_id", "departement_id", "region_id"]:
        df_locations[column] = df_locations[column].astype(object)

    missing = df_locations["commune_id"].isna()
    df_locations.loc[missing, "commune_id"] = df_locations.loc[missing, "muntype_commune_id"]
    df_locations.loc[missing, "iris_id"] = "CH"
    df_locations.loc[missing, "departement_id"] = "CH"
    df_locations.loc[missing, "region_id"] = "CH"

    df_locations = df_locations.drop(columns = ["muntype_commune_id"])

    print(df_locations.municipality_type.value_counts(dropna = False))

    # Attach SIRENE-based densities to activities
    df_locations["x"] = df_locations.geometry.x
    df_locations["y"] = df_locations.geometry.y
    threads = max(1, min(context.config("processes"), 8)) # avoid too many threads for this step as it can cause memory issues

    df_locations = impute_sirene(context, df_locations, x = "x", y = "y", chunk_size = 10_000,
        radius = EMPLOYEES_DENSITY_RADIUS, point_type = "activity", measure = "employees",
        output_column = "employee_density", n_jobs = threads)

    df_locations = impute_sirene(context, df_locations, x = "x", y = "y", chunk_size = 10_000,
        radius = EMPLOYEES_DENSITY_RADIUS, point_type = "activity", measure = "companies",
        output_column = "companies_density", n_jobs = threads)

    df_locations = impute_population(context, context.stage("data.spatial.population_density"), df_locations,
        x = "x", y = "y", chunk_size = 5000, n_jobs = threads,
        radius = POPULATION_DENSITY_RADIUS, point_type = "activity")

    df_locations = impute_ovgk(context, df_locations, x = "x", y = "y", chunk_size = 5000,
        point_type = "activity", output_column = "ovgk", n_jobs = threads)

    print(df_locations.columns)

    df_trips = tag_short_assigned_trips(df_locations, context.stage("synthesis.population.trips_boat"))

    return df_locations, df_trips
