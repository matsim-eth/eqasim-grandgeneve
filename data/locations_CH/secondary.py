def configure(context):
    context.stage("data.locations_CH.statent.destinations")


def execute(context):
    df_secplaces = context.stage("data.locations_CH.statent.destinations")[[
        "destination_id",
        "offers_leisure", "offers_other", "offers_shop",
        "geometry", "number_employees"
    ]].copy()

    # No need to select places offering secondary activities, all do as offers_other is always true

    # Use minimum number of employees as weight
    df_secplaces["employees"] = df_secplaces["number_employees"]
    df_secplaces["fake"] = False

    # destination_id is already canonical and disjoint from French ids -
    # no offset against secondary_FR's ids is needed.
    df_secplaces["enterprise_id"] = df_secplaces["destination_id"]
    df_secplaces["location_id"]   = df_secplaces["destination_id"]

    # Communes?
    df_secplaces["commune_id"] = "01001"

    # activity type
    df_secplaces["activity_type"] = "other"
    df_secplaces.loc[df_secplaces["offers_leisure"], "activity_type"] = "leisure"
    df_secplaces.loc[df_secplaces["offers_shop"],    "activity_type"] = "shop"

    return df_secplaces[["enterprise_id", "activity_type", "commune_id",  "geometry", "offers_leisure", "offers_shop", "offers_other", "location_id"]]
