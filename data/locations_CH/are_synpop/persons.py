import os
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.validation import make_valid


def configure(context):
    context.config("data_path")
    context.config("are_synpop", "other_locations/CH/are_synpop/2754_SynPop2022_Data_v1.0.zip")

    context.config("swiss_cantons", None)
    context.config("cantons", None)


def execute(context):
    data_path  = context.config("data_path")
    are_synpop = context.config("are_synpop")

    df = pd.read_csv("{}/{}".format(data_path, are_synpop), sep = ";")
    df = df[["person_id", "xcoord", "ycoord"]].rename(columns = { "xcoord": "x", "ycoord": "y" })

    df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df["x"], df["y"], crs = "epsg:2056"))

    # Select only locations in perimeter (same convention as data.locations_CH.statent.statent)
    perimeter = context.config("swiss_cantons")
    cantons   = context.config("cantons")

    if isinstance(perimeter, str):
        if perimeter.endswith(".shp") or perimeter.endswith(".gpkg"):
            target_region = gpd.read_file(perimeter, engine = "pyogrio", encoding = "utf8").to_crs("epsg:2056")
            if "KANTON" in perimeter or "CANTON" in perimeter:
                if isinstance(cantons, list):
                    target_region = target_region[target_region["NAME"].isin(cantons)]
                elif isinstance(cantons, str):
                    target_region = target_region[target_region["NAME"] == cantons]
            target_region = target_region.geometry.apply(make_valid).buffer(0).unary_union
        else:
            raise ValueError("Unsupported file format: %s" % perimeter)

    elif isinstance(perimeter, list):
        geometries = []
        for path in perimeter:
            if path.endswith(".shp") or path.endswith(".gpkg"):
                gdf = gpd.read_file(path, engine = "pyogrio", encoding = "latin1").to_crs("epsg:2056")
                geometries.append(gdf.geometry.apply(make_valid).buffer(0).unary_union)
            else:
                raise ValueError("Unsupported file format: %s" % path)
        target_region = unary_union(geometries)

    else:
        raise ValueError(
            "swiss_cantons must be a file path or a list of file paths. Got: %s" % type(perimeter)
        )

    df = df[df.within(target_region)]

    df = df.to_crs("EPSG:2154")
    df.loc[:, "x"] = df.geometry.x
    df.loc[:, "y"] = df.geometry.y

    return df[["person_id", "x", "y", "geometry"]]


def validate(context):
    data_path  = context.config("data_path")
    are_synpop = context.config("are_synpop")
    path = "{}/{}".format(data_path, are_synpop)

    if not os.path.exists(path):
        raise RuntimeError("ARE SynPop data is not available: %s" % path)

    return os.path.getsize(path)
