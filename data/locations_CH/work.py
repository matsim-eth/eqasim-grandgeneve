def configure(context):
    context.stage("data.locations_CH.statent.destinations")


def execute(context):
    df_workplaces = context.stage("data.locations_CH.statent.destinations")[[
        "destination_id", "number_employees", "geometry", "municipality_id"
    ]].copy()

    # Use minimum number of employees as weight
    df_workplaces["employees"] = df_workplaces["number_employees"]
    df_workplaces["fake"] = False

    # destination_id is already canonical and disjoint from French ids -
    # no offset against work_FR's ids is needed.
    df_workplaces["location_id"] = df_workplaces["destination_id"]

    # Communes?
    df_workplaces["commune_id"] = df_workplaces["municipality_id"]

    return df_workplaces[["location_id", "commune_id", "employees", "fake", "geometry"]]
