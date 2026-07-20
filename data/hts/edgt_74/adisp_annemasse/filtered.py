import data.hts.hts as hts

"""
This stage filters out observations which live or work outside of the area.
"""

def configure(context):
    context.stage("data.hts.edgt_74.adisp_annemasse.cleaned")
    
    context.stage("data.spatial.codes")
    
    context.config("filter_hts", True)


def execute(context):
    filter_edgt = context.config("filter_hts")
    df_codes    = context.stage("data.spatial.codes").copy()
    df_codes    = df_codes[df_codes["iris_id"] != "CH"]

    df_households, df_persons, df_trips = context.stage("data.hts.edgt_74.adisp_annemasse.cleaned")

    if filter_edgt : 
        # Filter for non-residents
        requested_departments = df_codes["departement_id"].unique()
        f = df_persons["departement_id"].astype(str).isin(requested_departments)
        df_persons = df_persons[f]

        # Filter for people going outside of the area
        remove_ids = set()

        remove_ids |= set(df_trips[
            ~df_trips["origin_departement_id"].astype(str).isin(requested_departments) | ~df_trips["destination_departement_id"].astype(str).isin(requested_departments)
        ]["person_id"].unique())

        df_persons = df_persons[~df_persons["person_id"].isin(remove_ids)]

        # Only keep trips and households that still have a person
        df_trips = df_trips[df_trips["person_id"].isin(df_persons["person_id"].unique())]
        df_households = df_households[df_households["household_id"].isin(df_persons["household_id"])]

    # Keep the raw origin/destination zone codes (D3/D7), renamed, so the
    # unified zoning system can be resolved from them in data.hts.edgt_74.adisp_merge.merge.
    df_trips = df_trips.rename(columns = { "D3": "origin_zone_id", "D7": "destination_zone_id" })

    # Extra raw survey columns kept around for the cross-perimeter model
    # (data.hts.edgt_74.adisp_merge), on top of the generic hts.PERSON_COLUMNS.
    CROSSPERIM_MODEL_COLUMNS = [
        "P3", "P8", "P14", "P16",
        "P19", "P20", "P21", "P22", "P23", "P24",
        "P25", "P26",
    ]

    # Finish up
    df_households = df_households[hts.HOUSEHOLD_COLUMNS + ["edgt_household_id", "M6"]]
    df_persons = df_persons[hts.PERSON_COLUMNS + ["edgt_person_id", "edgt_household_id", "ZFP"] + CROSSPERIM_MODEL_COLUMNS]
    df_trips = df_trips[hts.TRIP_COLUMNS + ["routed_distance", "euclidean_distance", "origin_zone_id", "destination_zone_id", "edgt_person_id", "edgt_household_id", "edgt_trip_id"]]

    hts.check(df_households, df_persons, df_trips)
    return df_households, df_persons, df_trips
