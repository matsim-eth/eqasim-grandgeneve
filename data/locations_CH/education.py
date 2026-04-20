import numpy as np

def configure(context):
    context.stage("data.locations_CH.statent.destinations")
    context.stage("synthesis.locations.education_FR")


def execute(context):
    df_educplaces = context.stage("data.locations_CH.statent.destinations")[[
        "destination_id", "number_employees", "geometry", "municipality_id"
    ]].copy()

    # Use minimum number of employees as weight
    df_educplaces["employees"] = df_educplaces["number_employees"]
    df_educplaces["fake"]      = False

    # Get minimum ID to use here
    df_educ_fr       = context.stage("synthesis.locations.education_FR")
    max_educplace_id = df_educ_fr["location_id"].str.split("edu_").str[-1].astype(int).max() + 1

    # Add work identifier
    df_educplaces["location_id"] = np.arange(max_educplace_id, max_educplace_id + len(df_educplaces))
    df_educplaces["location_id"] = "edu_" + df_educplaces["location_id"].astype(str)

    # Communes?
    df_educplaces["commune_id"]     = df_educplaces["municipality_id"]
    df_educplaces["weight"]         = df_educplaces["employees"]
    df_educplaces["education_type"] = "C0"

    return df_educplaces[["location_id", "education_type", "commune_id", "weight", "fake", "geometry"]]
