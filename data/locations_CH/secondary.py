import numpy as np

def configure(context):
    context.stage("data.locations_CH.statent.destinations")
    context.stage("synthesis.locations.secondary_FR")


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

    # Get minimum ID to use here
    df_sec_fr       = context.stage("synthesis.locations.secondary_FR")
    max_secplace_id = df_sec_fr["location_id"].str.split("sec_").str[-1].astype(int).max() + 1
    print(max_secplace_id)

    # Add work identifier
    df_secplaces["enterprise_id"] = np.arange(max_secplace_id, max_secplace_id + len(df_secplaces))
    df_secplaces["location_id"] = "sec_" + df_secplaces["enterprise_id"].astype(str)

    # Communes?
    df_secplaces["commune_id"] = "01001"

    # activity type
    df_secplaces["activity_type"] = "other"
    df_secplaces.loc[df_secplaces["offers_leisure"], "activity_type"] = "leisure"
    df_secplaces.loc[df_secplaces["offers_shop"],    "activity_type"] = "shop"

    return df_secplaces[["enterprise_id", "activity_type", "commune_id",  "geometry", "offers_leisure", "offers_shop", "offers_other", "location_id"]]
