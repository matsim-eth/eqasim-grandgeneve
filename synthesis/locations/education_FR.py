import pandas as pd
import geopandas as gpd
import data.id_utils

def configure(context):
    context.stage("data.spatial.municipalities")

    if context.config("education_location_source","bpe") == "addresses":
        context.stage("data.external.education", alias = "location_source")
    else:
        context.stage("data.bpe.cleaned", alias = "location_source")


def fake_education(missing_communes, c, df_locations, df_zones):
    # Fake education destinations as the centroid of zones that have no other destinations
    print(
        "Adding fake education locations for %d municipalities"
        % (len(missing_communes))
    )

    df_added = []

    for commune_id in sorted(missing_communes):
        centroid = df_zones[df_zones["commune_id"] == commune_id][
            "geometry"
        ].centroid.iloc[0]

        # No real facility here, so fall back to a commune+level-keyed id
        # (base62 when numeric; Corsica's "2A"/"2B" codes are kept as-is).
        commune_key = data.id_utils.to_base62(int(commune_id)) if str(commune_id).isdigit() else str(commune_id)
        df_added.append({
            "commune_id": commune_id, "geometry": centroid,
            "location_id": "FR_EDU_CENTROID_%s_%s" % (commune_key, c),
        })

    df_added = gpd.GeoDataFrame(
        pd.DataFrame.from_records(df_added), crs=df_locations.crs
    )
    df_added["fake"] = True
    df_added["education_type"] = c
    df_added["weight"] = 1

    return df_added


def execute(context):
    df_locations = context.stage("location_source")

    df_locations = df_locations[df_locations["activity_type"] == "education"]
    id_column = "enterprise_id" if "enterprise_id" in df_locations.columns else "location_id"
    df_locations = df_locations[[id_column, "education_type", "commune_id","weight", "geometry"]].copy()
    df_locations = df_locations.rename(columns = {id_column: "location_id"})
    df_locations["fake"] = False

    # Add education destinations to the centroid of zones that have no other destinations
    df_zones = context.stage("data.spatial.municipalities")

    required_communes = set(df_zones["commune_id"].unique())  

    if context.config("education_location_source") != 'bpe' :
 
        # Add education destinations in function of level education
        for c in ["C1", "C2", "C3"]:
            missing_communes = required_communes - set(df_locations[df_locations["education_type"].str.startswith(c)]["commune_id"].unique())

            if len(missing_communes) > 0:
                df_locations = pd.concat([df_locations,fake_education(missing_communes, c, df_locations, df_zones)])
        
        # Add education destinations for last level education
        missing_communes = required_communes - set(df_locations[~(df_locations["education_type"].str.startswith(("C1", "C2", "C3")))]["commune_id"].unique())

        if len(missing_communes) > 0:

           df_locations = pd.concat([df_locations,fake_education(missing_communes, "C4", df_locations, df_zones)])
   
    else :

        missing_communes = required_communes - set(df_locations["commune_id"].unique())
        if len(missing_communes) > 0:

            df_locations = pd.concat([df_locations,fake_education(missing_communes, "C0", df_locations, df_zones)])
    
    df_locations["education_type"] = df_locations["education_type"].str[:2].astype("category")


    return df_locations[["location_id", "education_type", "commune_id", "weight", "fake", "geometry"]]
