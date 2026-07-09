import numpy as np
import pandas as pd

import data.hts.edgt_74.adisp_merge.zones as zones

"""
This stage merges the reweighted Annemasse and Annecy EDGT (Adisp) surveys
into a single edgt_74 data set: it tags each record's origin survey, makes
sure household/person/trip ids are unique across the two surveys, and
resolves each trip's origin/destination zone code (carried through as
origin_zone_id/destination_zone_id by the two surveys' filtered stages) to
the unified zoning system built in data.hts.edgt_74.adisp_merge.zones.
"""


def configure(context):
    context.stage("data.hts.edgt_74.adisp_annemasse.reweighted")
    context.stage("data.hts.edgt_74.adisp_annecy.reweighted")
    context.stage("data.hts.edgt_74.adisp_merge.zones")


def deduplicate_ids(base_households, base_persons, base_trips, other_households, other_persons, other_trips):
    """Offset other_*'s household_id/person_id/trip_id so none of them collide
    with base_*'s, since both surveys number these independently starting
    from their own raw survey codes/0."""
    other_households = other_households.copy()
    other_persons    = other_persons.copy()
    other_trips      = other_trips.copy()

    household_offset = base_households["household_id"].max() + 1
    person_offset     = base_persons["person_id"].max() + 1
    trip_offset       = base_trips["trip_id"].max() + 1

    other_households["household_id"] += household_offset

    other_persons["household_id"] += household_offset
    other_persons["person_id"]     += person_offset

    other_trips["person_id"] += person_offset
    other_trips["trip_id"]   += trip_offset

    return other_households, other_persons, other_trips


def resolve_trip_zone_ids(df_trips, df_zones, df_points, code_to_dtir):
    """Resolve origin_zone_id/destination_zone_id in place, from the raw
    survey zone codes they hold coming in (D3/D7, renamed by the surveys'
    filtered stages) to the unified zoning's zone_id."""
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


def execute(context):
    annemasse_households, annemasse_persons, annemasse_trips = context.stage("data.hts.edgt_74.adisp_annemasse.reweighted")
    annecy_households, annecy_persons, annecy_trips           = context.stage("data.hts.edgt_74.adisp_annecy.reweighted")

    annecy_households, annecy_persons, annecy_trips = deduplicate_ids(
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
    df_trips = resolve_trip_zone_ids(df_trips, df_zones, df_points, code_to_dtir)

    return df_households, df_persons, df_trips
