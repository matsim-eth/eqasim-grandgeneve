import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.validation import make_valid



def configure(context):
    context.config("data_path")
    context.config("statent", "other_locations/CH/statent/250221_STATENT_2022_LOC_17042025.csv")

    context.config("swiss_cantons", None)
    context.config("cantons", None)

    context.stage("data.spatial.ch.municipalities")


def to_gpd(context, df, x="x", y="y", crs="epsg:2056", coord_type="", chunk_size=10000):
    
    result = []
    chunk_count = max(1, int(len(df) / chunk_size))
    for chunk in context.progress(np.array_split(df, chunk_count), 
                              total=chunk_count,
                              label="Converting %s coordinates" % coord_type):
        result.append(
            gpd.GeoDataFrame(
                chunk,
                geometry=gpd.points_from_xy(chunk[x], chunk[y], crs=crs)
                )
            )
        
    df = gpd.GeoDataFrame(
        pd.concat(result).reset_index(),
        crs=result[0].crs
        )
    del result
    
    if not crs == "epsg:2056":
        df = df.to_crs("epsg:2056")
        df.crs = "epsg:2056"

    return df


def execute(context):
    data_path = context.config("data_path")
    statent   = context.config("statent")

    df = pd.DataFrame(pd.read_csv(f"{data_path}/{statent}", encoding = "latin1", sep = ","))

    df         = pd.DataFrame(df[["METER_X", "METER_Y", "NOGA08_CD", "EMPTOT"]])
    df.columns = ["x", "y", "noga", "number_employees"]
    df         = df.astype({"noga": str})
    df.loc[:, "enterprise_id"] = np.arange(len(df))

    df.loc[df["noga"].str.startswith("851"), "education_type"] = "kindergarten"
    df.loc[df["noga"].str.startswith("852"), "education_type"] = "primary"
    df.loc[df["noga"].str.startswith("853"), "education_type"] = "secondary"
    df.loc[df["noga"].str.startswith("854"), "education_type"] = "tertiary"
    df["education_type"] = df["education_type"].astype("category")

    df_spatial = pd.DataFrame(df[["enterprise_id", "x", "y", "number_employees", "noga"]])
    df_spatial = to_gpd(context, df_spatial)
    df         = df_spatial.copy()

    # Select only locations in perimeter
    perimeter = context.config("swiss_cantons")
    cantons   = context.config("cantons")

    if isinstance(perimeter, str):
        if perimeter.endswith(".shp") or perimeter.endswith(".gpkg"):
            target_region = gpd.read_file(perimeter, engine="pyogrio", encoding="utf8").to_crs("epsg:2056")
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
                gdf = gpd.read_file(path, engine="pyogrio", encoding="latin1").to_crs("epsg:2056")
                geometries.append(gdf.geometry.apply(make_valid).buffer(0).unary_union)
            else:
                raise ValueError("Unsupported file format: %s" % path)
        target_region = unary_union(geometries)

    else:
        raise ValueError(
            "outbound_flows_perimeter must be a number, a file path, or a list of file paths. Got: %s" % type(perimeter)
        )
    
    municipalities = context.stage("data.spatial.ch.municipalities")[["municipality_id", "geometry"]].copy()
    municipalities["geometry"] = municipalities.simplify(tolerance=100)

    locations      = gpd.GeoDataFrame(df[["enterprise_id", "x", "y"]], geometry=gpd.points_from_xy(df["x"], df["y"]),crs=municipalities.crs)
    loc_mun        = gpd.sjoin(locations, municipalities, how="left", predicate="intersects")
    loc_mun        = loc_mun[~loc_mun.index.duplicated(keep="first")]

    df["municipality_id"] = loc_mun["municipality_id"]

    df = df[df["municipality_id"].notna()]
    df["municipality_id"] = "CH" + df["municipality_id"].astype(int).astype(str).str.zfill(5)

    df = df[df.within(target_region)]
    df = df.to_crs("EPSG:2154")
    df.loc[:, "x"] =  df.x
    df.loc[:, "y"] =  df.y

    return df
