import os
import geopandas as gpd

"""
Swiss lakes (swisstopo generalized boundaries, g1s18), including the French
parts of Lake Geneva, used as barriers in the secondary location assignment
(synthesis.population.spatial.secondary.locations).
"""

def configure(context):
    context.config("data_path")
    context.config("swiss_lakes_path", "other_locations/CH_lakes/g1s18.shp")

def get_path(context):
    return "%s/%s" % (context.config("data_path"), context.config("swiss_lakes_path"))

def execute(context):
    df = gpd.read_file(get_path(context), engine = "pyogrio", encoding = "utf8")

    # The file's CRS is LV95 + LN02 height, keep the horizontal part
    df = df.set_crs("epsg:2056", allow_override = True).to_crs("epsg:2154")

    df = df.rename(columns = { "SEENAME": "lake_name" })
    return df[["lake_name", "geometry"]]

def validate(context):
    path = get_path(context)

    if not os.path.exists(path):
        raise RuntimeError("Swiss lakes not available: %s" % path)

    return os.path.getsize(path)
