import pandas as pd
import geopandas as gpd
import data.id_utils

def configure(context):
    context.stage("data.bpe.cleaned")
    context.stage("data.spatial.municipalities")

    context.config("data_path")
    context.config("education_file", "education/education_addresses.geojson")

def execute(context):
    df_locations = context.stage("data.bpe.cleaned")[[
         "enterprise_id", "activity_type", "education_type", "commune_id","weight", "geometry"
    ]]

    df_locations = df_locations[df_locations["activity_type"] == "education"]
    df_locations = df_locations[["enterprise_id", "activity_type","education_type", "commune_id","weight", "geometry"]].copy()
    df_locations = df_locations.rename(columns = {"enterprise_id": "location_id"})
    df_locations["fake"] = False

    df_zones = context.stage("data.spatial.municipalities")
    required_communes = set(df_zones["commune_id"].unique())


    df_education = gpd.read_file("{}/{}".format(context.config("data_path"), context.config("education_file")))[["education_type", "commune_id","weight", "geometry"]]
    df_education["fake"] = False
    df_education = df_education.to_crs("2154")
    df_education["activity_type"] = "education"

    # No stable registry id here, so fall back to a key built from the row's
    # own attributes (base62-encoded where numeric).
    commune_key = df_education["commune_id"].apply(
        lambda c: data.id_utils.to_base62(int(c)) if str(c).isdigit() else str(c))

    df_education["location_id"] = (
        "FR_EDU_ADDR_" + commune_key + "_" +
        df_education["education_type"].astype(str) + "_" +
        df_education.geometry.x.round().astype(int).apply(data.id_utils.to_base62) + "_" +
        df_education.geometry.y.round().astype(int).apply(data.id_utils.to_base62)
    )

    list_type = set(df_education["education_type"].unique())
    df_locations = pd.concat([df_locations[~(df_locations["education_type"].str.startswith(tuple(list_type)))],df_education[df_education["commune_id"].isin(required_communes)]])

    return df_locations
