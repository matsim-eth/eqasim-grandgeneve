def configure(context):
    context.stage("data.bpe.cleaned")
    context.stage("data.spatial.municipalities")


def execute(context):
    df_locations = context.stage("data.bpe.cleaned")[[
        "enterprise_id", "activity_type", "commune_id", "geometry"
    ]].copy()

    # enterprise_id is already the canonical id - reuse it directly.
    df_locations["destination_id"] = df_locations["enterprise_id"]
    df_locations["location_id"]    = df_locations["enterprise_id"]

    # Attach attributes for activity types
    df_locations["offers_leisure"] = df_locations["activity_type"] == "leisure"
    df_locations["offers_shop"] = df_locations["activity_type"] == "shop"
    df_locations["offers_other"] = ~(df_locations["offers_leisure"] | df_locations["offers_shop"])

    return df_locations
