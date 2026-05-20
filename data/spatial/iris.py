import pandas as pd
import geopandas as gpd
import os
import py7zr
import glob
from shapely.ops import unary_union

"""
Loads the IRIS zoning system.
"""

def configure(context):
    context.config("data_path")
    context.config("iris_path", "iris_2024")
    context.stage("data.spatial.codes")

    context.config("hts", default = "emp")
    context.config("edgt74_version", default = "adisp")


def execute(context):
    df_codes = context.stage("data.spatial.codes").copy()
    df_codes = df_codes[df_codes["iris_id"] != "CH"]

    source_path = find_iris("{}/{}".format(context.config("data_path"), context.config("iris_path")))

    with py7zr.SevenZipFile(source_path) as archive:
        contour_paths = [
            path for path in archive.getnames()
            if "LAMB93" in path
        ]

        archive.extract(context.path(), contour_paths)
    
    gpkg_path = [path for path in contour_paths if path.endswith(".gpkg")]

    if len(gpkg_path) != 1:
        raise RuntimeError("Cannot find IRIS shapes inside the archive, please report this as an error!")

    df_iris = gpd.read_file("{}/{}".format(context.path(), gpkg_path[0]),dtype={"code_iris":str,"code_insee":str})[[
        "code_insee", "code_iris", "geometry"
    ]].rename(columns = {
        "code_iris": "iris_id",
        "code_insee": "commune_id"
    })

    assert df_iris.crs == "EPSG:2154"

    # if hts == edgt_74, assign survey_coverage to iris to detect who lives in Annemasse vs Annecy area
    hts            = context.config("hts")
    edgt74_version = context.config("edgt74_version")
    if hts == "edgt_74" and edgt74_version == "adisp":
        envelope_annemasse = unary_union(gpd.read_file("{}/{}".format(context.config("data_path"), "edgt_2017_annemasse/annemasse_extent.gpkg"), crs = "EPSG:2154").geometry.buffer(1000))
        envelope_annecy    = unary_union(gpd.read_file("{}/{}".format(context.config("data_path"), "edgt_2017_annecy/annecy_extent.gpkg"), crs = "EPSG:2154").geometry.buffer(1000))

        df_iris["edgt_area"] = None
        df_iris.loc[df_iris.geometry.within(envelope_annemasse), "edgt_area"] = "annemasse"
        df_iris.loc[df_iris.geometry.within(envelope_annecy),    "edgt_area"] = "annecy"

        df_iris.loc[df_iris["iris_id"] == "742120000", "edgt_area"] = "annemasse"

        #df_iris = df_iris[df_iris["edgt_area"].notna()]


    df_iris["iris_id"]    = df_iris["iris_id"].astype("category")
    df_iris["commune_id"] = df_iris["commune_id"].astype("category")

    # Merge with requested codes and verify integrity
    df_iris = pd.merge(df_iris, df_codes, on = ["iris_id", "commune_id"])    

    #requested_iris = set(df_codes["iris_id"].unique())
    #merged_iris    = set(df_iris["iris_id"].unique())

    #if requested_iris != merged_iris:
    #    raise RuntimeError("Some IRIS are missing: %s" % (requested_iris - merged_iris,))

    return df_iris


def find_iris(path):
    candidates = sorted(list(glob.glob("{}/*.7z".format(path))))

    if len(candidates) == 0:
        raise RuntimeError("IRIS data is not available in {}".format(path))
    
    if len(candidates) > 1:
        raise RuntimeError("Multiple candidates for IRIS are available in {}".format(path))
    
    return candidates[0]


def validate(context):
    path = find_iris("{}/{}".format(context.config("data_path"), context.config("iris_path")))
    return os.path.getsize(path)
