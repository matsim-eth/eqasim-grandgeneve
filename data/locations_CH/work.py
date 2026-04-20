import numpy as np

def configure(context):
    context.stage("data.locations_CH.statent.destinations")
    context.stage("synthesis.locations.work_FR")


def execute(context):
    df_workplaces = context.stage("data.locations_CH.statent.destinations")[[
        "destination_id", "number_employees", "geometry", "municipality_id"
    ]].copy()

    # Use minimum number of employees as weight
    df_workplaces["employees"] = df_workplaces["number_employees"]
    df_workplaces["fake"] = False

    # Get minimum ID to use here
    df_work_fr       = context.stage("synthesis.locations.work_FR")
    max_workplace_id = df_work_fr["location_id"].str.split("work_").str[-1].astype(int).max() + 1

    # Add work identifier
    df_workplaces["location_id"] = np.arange(max_workplace_id, max_workplace_id + len(df_workplaces))
    df_workplaces["location_id"] = "work_" + df_workplaces["location_id"].astype(str)

    # Communes?
    df_workplaces["commune_id"] = df_workplaces["municipality_id"]

    return df_workplaces[["location_id", "commune_id", "employees", "fake", "geometry"]]
