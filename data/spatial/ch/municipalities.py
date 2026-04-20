import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def configure(context):
    context.config("swiss_municipalities")


def execute(context):
    data_path = context.config("swiss_municipalities")

    df = gpd.read_file(data_path, engine="pyogrio", encoding = "utf8").to_crs("epsg:2056")
    df.crs = "epsg:2056"

    df.loc[:, "municipality_id"]   = df["BFS_NUMMER"]
    df.loc[:, "municipality_name"] = df["NAME"]
    df.loc[:, "canton_id"]         = df["KANTONSNUM"]

    return df[["municipality_id", "municipality_name", "geometry", "canton_id"]]
