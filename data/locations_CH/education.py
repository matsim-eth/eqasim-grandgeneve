def configure(context):
    context.stage("data.locations_CH.statent.destinations")


def execute(context):
    df_educplaces = context.stage("data.locations_CH.statent.destinations")[[
        "destination_id", "number_employees", "geometry", "municipality_id"
    ]].copy()

    # Use minimum number of employees as weight
    df_educplaces["employees"] = df_educplaces["number_employees"]
    df_educplaces["fake"]      = False

    # destination_id is already canonical and disjoint from French ids -
    # no offset against education_FR's ids is needed.
    df_educplaces["location_id"] = df_educplaces["destination_id"]

    # Communes?
    df_educplaces["commune_id"]     = df_educplaces["municipality_id"]
    df_educplaces["weight"]         = df_educplaces["employees"]
    df_educplaces["education_type"] = "C0"

    return df_educplaces[["location_id", "education_type", "commune_id", "weight", "fake", "geometry"]]
