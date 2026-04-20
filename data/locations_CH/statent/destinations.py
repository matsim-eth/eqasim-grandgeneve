import pandas as pd
import numpy as np
import geopandas as gpd

def configure(context):
    context.stage("data.locations_CH.statent.statent")


def to_gpd(context, df, x="x", y="y", crs="epsg:2154", coord_type="", chunk_size=10000):
    
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
    
    if not crs == "epsg:2154":
        df = df.to_crs("epsg:2154")
        df.crs = "epsg:2154"

    return df


def execute(context):
    df = pd.DataFrame(context.stage("data.locations_CH.statent.statent")[["enterprise_id", "x", "y", "noga", "geometry", "number_employees", "municipality_id"]],
                                    copy=True)
    df.columns = ["destination_id", "destination_x", "destination_y", "noga", "geometry", "number_employees", "municipality_id"]

    df.loc[:, "offers_work"]  = True
    df.loc[:, "offers_other"] = True

    # 85 = education
    df.loc[:, "offers_education"] = df["noga"].str.startswith("85")

    # 90 = arts, entertainment, leisure; 56 = gastronomy
    df.loc[:, "offers_leisure"] = df["noga"].str.startswith("90") | df["noga"].str.startswith("56")

    # 47 = retail
    df.loc[:, "offers_shop"] = df["noga"].str.startswith("47")

    #del df["noga"]

    return df[["destination_id", "destination_x", "destination_y", "municipality_id",
               "offers_work", "offers_education", "offers_leisure", "offers_shop", "offers_other",
               "geometry",  "number_employees", "noga"]]
