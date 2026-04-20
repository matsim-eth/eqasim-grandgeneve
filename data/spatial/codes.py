import os
import pandas as pd
import zipfile

"""
This stages loads a file containing all spatial codes in France and how
they can be translated into each other. These are mainly IRIS, commune,
departement and région.
"""

def configure(context):
    context.config("data_path")

    context.config("regions", [11])
    context.config("departments", [])
    context.config("codes_path", "codes_2024/reference_IRIS_geo2024.zip")
    context.config("codes_xlsx", "reference_IRIS_geo2024.xlsx")

    context.config("generate_outbound_flows", "False")

    if context.config("generate_outbound_flows"):
        context.stage("data.spatial.ch.spatial")

def execute(context):
    # Load IRIS registry
    with zipfile.ZipFile(
        "{}/{}".format(context.config("data_path"), context.config("codes_path"))) as archive:
        with archive.open(context.config("codes_xlsx")) as f:
            df_codes = pd.read_excel(f,
                skiprows = 5, sheet_name = "Emboitements_IRIS",dtype={"CODE_IRIS":str,"DEPCOM":str}
            )[["CODE_IRIS", "DEPCOM", "DEP", "REG"]].rename(columns = {
                "CODE_IRIS": "iris_id",
                "DEPCOM": "commune_id",
                "DEP": "departement_id",
                "REG": "region_id"
            }).fillna('0')

    df_codes["iris_id"]        = df_codes["iris_id"].astype("category")
    df_codes["commune_id"]     = df_codes["commune_id"].astype("category")
    df_codes["departement_id"] = df_codes["departement_id"].astype("category")
    df_codes["region_id"]      = df_codes["region_id"].astype(int)

    # Filter zones
    requested_regions     = list(map(int, context.config("regions")))
    requested_departments = list(map(str, context.config("departments")))

    if len(requested_regions) > 0:
        df_codes = df_codes[df_codes["region_id"].isin(requested_regions)]

    if len(requested_departments) > 0:
        df_codes = df_codes[df_codes["departement_id"].isin(requested_departments)]    

    if context.config("generate_outbound_flows"):
        _, municipalities = context.stage("data.spatial.ch.spatial")

        municipalities.loc[:, "commune_id"] = "CH" + municipalities["municipality_id"].astype(str).str.zfill(5)
        municipalities["iris_id"]           = "CH"
        municipalities["departement_id"]    = "CH"
        municipalities["region_id"]         = "CH"

        df_codes = pd.concat([df_codes, municipalities[["iris_id", "commune_id", "departement_id", "region_id"]]])
        df_codes["iris_id"]        = df_codes["iris_id"].astype("category")
        df_codes["commune_id"]     = df_codes["commune_id"].astype("category")
        df_codes["departement_id"] = df_codes["departement_id"].astype("category")
        df_codes["region_id"]      = df_codes["region_id"].astype("category")

    df_codes["iris_id"]        = df_codes["iris_id"].cat.remove_unused_categories()
    df_codes["commune_id"]     = df_codes["commune_id"].cat.remove_unused_categories()
    df_codes["departement_id"] = df_codes["departement_id"].cat.remove_unused_categories()

    print(df_codes.head())
    print(df_codes.tail())

    return df_codes

def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("codes_path"))):
        raise RuntimeError("Spatial reference codes are not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), context.config("codes_path")))
