import numpy as np
import pandas as pd
from shapely.ops import unary_union

import data.hts.edgt_74.adisp_merge.zones as zones

"""
This stage merges the reweighted Annemasse and Annecy EDGT (Adisp) surveys
into a single edgt_74 data set: it tags each record's origin survey, makes
sure household/person/trip ids are unique across the two surveys, and
resolves each trip's origin/destination zone code (carried through as
origin_zone_id/destination_zone_id by the two surveys' filtered stages) to
the unified zoning system built in data.hts.edgt_74.adisp_merge.zones.
"""

# P8 (highest education attained) collapsed to a 4-level ordinal: "currently
# in school" / no schooling / primary bucket together at the bottom since
# none of them denote a completed secondary-or-higher level.
#
# Lives here (rather than in synthesis.population.cross_perimeter, which pulls
# in heavy ML deps) so that synthesis.population.enriched can reuse it too
# when assigning educ_ord from the matched HTS respondent.
EDUCATION_ORDINAL_MAP = {
    0: 0, 1: 0, 9: 0,                 # currently in school, primary, no schooling
    2: 1, 3: 1, 4: 1, 7: 1,           # lower/upper secondary, apprenticeship (primary/secondary)
    5: 2,                             # higher ed. <=bac+2
    6: 3, 8: 3,                       # higher ed. bac+3+, apprenticeship (higher ed.)
}


def configure(context):
    context.stage("data.hts.edgt_74.adisp_annemasse.reweighted")
    context.stage("data.hts.edgt_74.adisp_annecy.reweighted")
    context.stage("data.hts.edgt_74.adisp_merge.zones")
    context.stage("data.spatial.ch.cantons")


def deduplicate_ids(base_households, base_persons, base_trips, other_households, other_persons, other_trips):

    base_households  = base_households.copy()
    base_persons     = base_persons.copy()
    base_trips       = base_trips.copy()
    other_households = other_households.copy()
    other_persons    = other_persons.copy()
    other_trips      = other_trips.copy()

    household_offset = base_households["household_id"].max() + 1
    person_offset     = base_persons["person_id"].max() + 1
    trip_offset       = base_trips["trip_id"].max() + 1

    base_households["household_id_old"] = base_households["household_id"]

    base_persons["household_id_old"] = base_persons["household_id"]
    base_persons["person_id_old"]    = base_persons["person_id"]

    base_trips["person_id_old"] = base_trips["person_id"]
    base_trips["trip_id_old"]   = base_trips["trip_id"]

    other_households["household_id_old"] = other_households["household_id"]
    other_households["household_id"]    += household_offset

    other_persons["household_id_old"] = other_persons["household_id"]
    other_persons["person_id_old"]    = other_persons["person_id"]
    other_persons["household_id"]    += household_offset
    other_persons["person_id"]       += person_offset

    other_trips["person_id_old"] = other_trips["person_id"]
    other_trips["trip_id_old"]   = other_trips["trip_id"]
    other_trips["person_id"]    += person_offset
    other_trips["trip_id"]      += trip_offset

    return base_households, base_persons, base_trips, other_households, other_persons, other_trips


def resolve_trip_zone_ids(df_trips, df_zones, df_points, code_to_dtir):

    code_to_zone_id, dtir_to_zone_id, dtir_to_zone_ids, legacy_to_zone_ids = zones.build_zone_code_lookups(df_zones)
    point_to_zone_id = zones.build_point_to_zone_id(df_points)
    rng = np.random.default_rng(0)

    for column in ["origin_zone_id", "destination_zone_id"]:
        # D3/D7 arrive zero-padded to 8 digits (the raw survey CSVs' own
        # convention, preserved through cleaned/filtered/reweighted since
        # those stages read them as plain strings), but the unified zoning's
        # zone_code and the other lookup keys are unpadded; strip the
        # padding so codes line up.
        codes = df_trips[column].astype(str).str.strip().str.lstrip("0")
        codes = codes.where(codes != "", "0")

        resolved = zones.resolve_zone_id(
            codes, code_to_zone_id, code_to_dtir,
            dtir_to_zone_id, dtir_to_zone_ids, legacy_to_zone_ids,
            point_to_zone_id, rng
        )

        n = len(resolved)
        print(f"{column}: {resolved.notna().sum()} / {n} trips resolved to a unified zone ({100 * resolved.notna().sum() / n:.1f}%)")

        df_trips[column] = resolved

    return df_trips


def find_cross_perimeter_trips(df_trips, df_zones):

    df_zones = df_zones[["zone_id", "d30_name"]].copy()

    df_trips = df_trips.merge(df_zones.rename(columns = {"zone_id": "origin_zone_id", "d30_name": "origin_zone_perimeter"}),
                                               on = "origin_zone_id", how = "left")
    df_trips = df_trips.merge(df_zones.rename(columns = {"zone_id": "destination_zone_id", "d30_name": "destination_zone_perimeter"}),
                                               on = "destination_zone_id", how = "left")
    
    outside_perimeter_zones = ["Reste Suisse", "Reste France",
                               "Reste Auvergne Rhône-Alpes",
                               "Communes des EPCI AU Lyon et Bourg-en-Bresse",
                               "Reste département de l'AIN"]
    
    df_trips.loc[:, "origin_outside_perimeter"]      = df_trips["origin_zone_perimeter"].isin(outside_perimeter_zones)
    df_trips.loc[:, "destination_outside_perimeter"] = df_trips["destination_zone_perimeter"].isin(outside_perimeter_zones)

    df_trips.loc[:, "is_crossperim_trip"] = df_trips["origin_outside_perimeter"] | df_trips["destination_outside_perimeter"]

    for column in ["destination_zone_perimeter", "origin_zone_perimeter", "origin_outside_perimeter", "destination_outside_perimeter"]:
        del df_trips[column]

    return df_trips


def find_cross_perimeter_persons(df_persons, df_trips):
    df_persons["is_crossperim_person"] = df_persons["person_id"].isin(
        df_trips[df_trips["is_crossperim_trip"]]["person_id"].unique()
    )

    return df_persons


INTERNAL_ZONE_SOURCES = ["annemasse_zf", "annecy_zf"]


def resolve_residence_zone_ids(df_persons, df_zones, df_points, code_to_dtir):
    """Resolve residence_zone_id from each person's home zone code (ZFP,
    carried through as-is by the two surveys' filtered stages), the same way
    resolve_trip_zone_ids resolves trip endpoints."""
    code_to_zone_id, dtir_to_zone_id, dtir_to_zone_ids, legacy_to_zone_ids = zones.build_zone_code_lookups(df_zones)
    point_to_zone_id = zones.build_point_to_zone_id(df_points)
    rng = np.random.default_rng(0)

    codes = df_persons["ZFP"].astype(str).str.strip().str.lstrip("0")
    codes = codes.where(codes != "", "0")

    resolved = zones.resolve_zone_id(
        codes, code_to_zone_id, code_to_dtir,
        dtir_to_zone_id, dtir_to_zone_ids, legacy_to_zone_ids,
        point_to_zone_id, rng
    )

    n = len(resolved)
    print(f"ZFP: {resolved.notna().sum()} / {n} persons resolved to a unified zone ({100 * resolved.notna().sum() / n:.1f}%)")

    df_persons["residence_zone_id"] = resolved

    return df_persons


def build_perimeter_geometry(df_zones, df_cantons):
    """Union of the two surveys' own zoning (source in annemasse_zf/annecy_zf,
    i.e. excluding the coarser "external"/foreign catch-all zones) and the
    neighbouring Swiss cantons (already filtered to the relevant ones by
    data.spatial.ch.cantons via the swiss_cantons/cantons config): the whole
    region a person has to leave to be considered a cross-perimeter person."""
    internal_zones = df_zones[df_zones["source"].isin(INTERNAL_ZONE_SOURCES)]
    df_cantons = df_cantons.to_crs(df_zones.crs)

    geometries = list(internal_zones.geometry.make_valid()) + list(df_cantons.geometry.make_valid())
    return unary_union(geometries)


def find_distance_to_perimeter(df_persons, df_zones, df_cantons):
    """Adds distance_to_perimeter_border (metres): each person's home zone
    centroid distance to the perimeter boundary."""
    perimeter = build_perimeter_geometry(df_zones, df_cantons)

    residence_zone_ids = df_persons["residence_zone_id"].dropna().unique()
    residence_zones = df_zones[df_zones["zone_id"].isin(residence_zone_ids)].set_index("zone_id")
    centroids = residence_zones.geometry.centroid

    distance_by_zone = centroids.distance(perimeter.boundary)
    df_persons["distance_to_perimeter_border"] = df_persons["residence_zone_id"].map(distance_by_zone)

    return df_persons


def execute(context):
    annemasse_households, annemasse_persons, annemasse_trips = context.stage("data.hts.edgt_74.adisp_annemasse.reweighted")
    annecy_households, annecy_persons, annecy_trips           = context.stage("data.hts.edgt_74.adisp_annecy.reweighted")

    annemasse_households, annemasse_persons, annemasse_trips, annecy_households, annecy_persons, annecy_trips = deduplicate_ids(
        annemasse_households, annemasse_persons, annemasse_trips,
        annecy_households, annecy_persons, annecy_trips
    )

    annemasse_households["edgt_area"] = "annemasse"
    annemasse_persons["edgt_area"]    = "annemasse"
    annemasse_trips["edgt_area"]      = "annemasse"
    annecy_households["edgt_area"]    = "annecy"
    annecy_persons["edgt_area"]       = "annecy"
    annecy_trips["edgt_area"]         = "annecy"

    df_households = pd.concat([annemasse_households, annecy_households])
    df_persons    = pd.concat([annemasse_persons, annecy_persons])
    df_trips      = pd.concat([annemasse_trips, annecy_trips])

    assert df_households["household_id"].is_unique
    assert df_persons["person_id"].is_unique
    assert df_trips["trip_id"].is_unique

    df_zones, df_points, code_to_dtir = context.stage("data.hts.edgt_74.adisp_merge.zones")
    df_cantons = context.stage("data.spatial.ch.cantons")

    df_trips   = resolve_trip_zone_ids(df_trips, df_zones, df_points, code_to_dtir)
    df_trips   = find_cross_perimeter_trips(df_trips, df_zones)
    df_persons = find_cross_perimeter_persons(df_persons, df_trips)

    df_persons = resolve_residence_zone_ids(df_persons, df_zones, df_points, code_to_dtir)
    df_persons = find_distance_to_perimeter(df_persons, df_zones, df_cantons)

    context_path = context.path()
    df_persons[[
        "edgt_household_id", "edgt_person_id", "is_crossperim_person",
        "residence_zone_id", "distance_to_perimeter_border",
    ]].to_csv(f"{context_path}/edgt_persons_all.csv", index = False)

    return df_households, df_persons, df_trips
