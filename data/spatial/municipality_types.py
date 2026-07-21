import os
import pandas as pd
import geopandas as gpd

def configure(context):
    context.config("data_path")

    context.stage("data.spatial.municipalities")

    if context.config("generate_outbound_flows", default = False):
        context.stage("data.spatial.ch.municipality_types")


def execute(context):
    data_path = context.config("data_path")
    file_path = f"{data_path}/municipality_type_2025/municipality_type_2025.csv"

    df = pd.read_csv(file_path, sep = ";", skiprows = 2)
    df.columns = ["commune_id", "commune_name", "municipality_type"]

    df["municipality_type"] = df["municipality_type"].map({1: "urban",
                                                           2: "suburban",
                                                           3: "rural"})

    df_municipalities = context.stage("data.spatial.municipalities")

    df = df.loc[df["commune_id"].isin(df_municipalities["commune_id"].unique())]
    df = df.merge(df_municipalities[["commune_id", "geometry"]], on = "commune_id", how = "left")
    
    df.loc[:, "commune_id"] = "FR" + df["commune_id"].astype(str).str.zfill(5)

    gdf = gpd.GeoDataFrame(df, geometry = "geometry", crs = "EPSG:2154")

    if context.config("generate_outbound_flows"):
        swiss_muntpyes = context.stage("data.spatial.ch.municipality_types")[["municipality_id", "municipality_type", "geometry"]]
        swiss_muntpyes.columns = ["commune_id", "municipality_type", "geometry"]
        
        ch_gdf = gpd.GeoDataFrame(swiss_muntpyes, geometry = "geometry", crs = "EPSG:2056")
        ch_gdf = ch_gdf.to_crs("EPSG:2154")

        gdf = pd.concat([gdf, ch_gdf])

    del gdf["commune_name"]

    return gdf


FILE = "municipality_type_2025/municipality_type_2025.csv"

def validate(context):
    data_path = context.config("data_path")
    file_path = f"{data_path}/{FILE}"

    if not os.path.exists(file_path):
        raise RuntimeError("Municipality type data is not available")

    return os.path.getsize(file_path)