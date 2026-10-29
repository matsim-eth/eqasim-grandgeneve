import os
import pandas as pd

"""
Loads the canton of each French commune (commune_id -> canton_id). The codes
are those of the census CANTVILLE (canton-ou-ville of residence), which is the
only location known for census records living in communes without IRIS. Used
to restrict the communes these records can be placed in
(synthesis.population.spatial.home.zones, home_zone_use_cantons).

The default file is the data.gouv.fr "communes de France" compilation (2023),
whose canton_code matches CANTVILLE for all records of the 2022 and 2023
censuses in departements 01 and 74.
"""

def configure(context):
    context.config("data_path")
    context.config("canton_codes_path", "communes_2023/communes-france-2023.csv")

    context.stage("data.spatial.codes")


def execute(context):
    df = pd.read_csv("{}/{}".format(context.config("data_path"), context.config("canton_codes_path")),
        usecols = ["code_insee", "canton_code"], dtype = str).rename(columns = {
        "code_insee": "commune_id", "canton_code": "canton_id" })

    df_codes = context.stage("data.spatial.codes")
    requested_communes = set(df_codes.loc[df_codes["iris_id"] != "CH", "commune_id"].astype(str))
    df = df[df["commune_id"].isin(requested_communes)].drop_duplicates("commune_id")

    missing = requested_communes - set(df["commune_id"])
    print("Canton codes: %d French communes, %d without canton" % (len(df), len(missing)))

    return df[["commune_id", "canton_id"]]


def validate(context):
    path = "{}/{}".format(context.config("data_path"), context.config("canton_codes_path"))
    if not os.path.exists(path):
        raise RuntimeError("Canton codes are not available at %s" % path)

    return os.path.getsize(path)
