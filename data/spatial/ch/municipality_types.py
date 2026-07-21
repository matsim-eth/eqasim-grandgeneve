import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from sklearn.neighbors import KDTree


def configure(context):
    context.config("swiss_municipality_types")
    context.stage("data.spatial.ch.municipalities")


def execute(context):
    # Load data
    data_path = context.config("swiss_municipality_types")

    df_types = pd.read_excel(data_path,
                             names    = ["municipality_id", "TYP"],
                             usecols  = [0, 6],
                             skiprows = 2,
                             nrows    = 2132,
                             )
    
    df_municipalities = context.stage("data.spatial.ch.municipalities")

    # Rewrite classification based on the official division    
    df_types.loc[df_types["TYP"] == 1, "municipality_type"] = "urbancore"
    df_types.loc[df_types["TYP"] == 2, "municipality_type"] = "urban"
    df_types.loc[df_types["TYP"] == 3, "municipality_type"] = "urban"
    df_types.loc[df_types["TYP"] == 4, "municipality_type"] = "suburban"
    df_types.loc[df_types["TYP"] == 5, "municipality_type"] = "suburban"
    df_types.loc[df_types["TYP"] == 6, "municipality_type"] = "rural"
    df_types.loc[df_types["TYP"] == 0, "municipality_type"] = "rural"

    df_types["municipality_type"] = df_types["municipality_type"].astype("category")
    
    df_types = df_types[["municipality_id", "municipality_type"]]

    # Match by municipality_id
    df_existing = pd.merge(df_municipalities, df_types, on="municipality_id")
    df_existing["imputed_municipality_type"] = False
    df_existing = df_existing[["municipality_id", "municipality_type", "imputed_municipality_type", "geometry"]]

    # Some ids are missing (because they are special zones)
    df_missing = gpd.GeoDataFrame(df_municipalities[
                                      ~df_municipalities["municipality_id"].isin(df_existing["municipality_id"])
                                  ])
    df_missing.crs = df_municipalities.crs
    df_missing = df_missing[["municipality_id", "geometry"]]

    coordinates = np.vstack([df_existing["geometry"].centroid.x, df_existing["geometry"].centroid.y]).T
    kd_tree = KDTree(coordinates)

    coordinates = np.vstack([df_missing["geometry"].centroid.x, df_missing["geometry"].centroid.y]).T
    indices = kd_tree.query(coordinates, return_distance=False).flatten()

    df_missing.loc[:, "municipality_type"] = df_existing.iloc[indices]["municipality_type"].values
    df_missing.loc[:, "imputed_municipality_type"] = True
    df_missing = df_missing[["municipality_id", "municipality_type", "imputed_municipality_type", "geometry"]]

    df_mapping = pd.concat((df_existing, df_missing))

    assert (len(df_mapping) == len(df_municipalities))
    assert (set(np.unique(df_mapping["municipality_id"])) == set(np.unique(df_municipalities["municipality_id"])))

    df_mapping = pd.DataFrame(df_mapping[["municipality_id", "municipality_type", "imputed_municipality_type", "geometry"]])
    df_mapping["municipality_type"] = df_mapping["municipality_type"].astype("category")
    df_mapping["geometry"] = shapely.force_2d(df_mapping["geometry"])

    df_mapping.loc[:, "municipality_id"] = "CH" + df_mapping["municipality_id"].astype(str).str.zfill(5)

    return df_mapping