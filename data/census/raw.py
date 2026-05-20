import os
import polars as pl

"""
This stage loads the raw data from the French population census.
"""

def configure(context):
    context.stage("data.spatial.codes")

    context.config("data_path")
    context.config("census_path", "rp_2022/RP2022_indcvi.parquet")


COLUMNS_DTYPES = {
    "CANTVILLE": "str", 
    "NUMMI":     "str", 
    "AGED":      "str",
    "COUPLE":    "str", 
    "GS":        "str",
    "STAT_GSEC": "str",
    "DEPT":      "str", 
    "ETUD":      "str",
    "IPONDI":    "str", 
    "IRIS":      "str",
    "REGION":    "str", 
    "SEXE":      "str",
    "TACT":      "str", 
    "TRANS":     "str",
    "VOIT":      "str", 
    "DEROU":     "str"
}


def execute(context):
    df_codes = context.stage("data.spatial.codes").copy()
    df_codes = df_codes[df_codes["iris_id"] != "CH"]

    requested_departements = df_codes["departement_id"].unique()

    with context.progress(label = "Reading census ...") as progress:
        parquet = pl.read_parquet( "{}/{}".format(context.config("data_path"), context.config("census_path")),
                        columns=  COLUMNS_DTYPES.keys())
        
        parquet  = parquet.cast(pl.String)
        dpts_str = [str(dep) for dep in requested_departements]
        parquet  = parquet.filter(pl.col("DEPT").is_in(dpts_str))

        progress.update(len(parquet))                    

    return parquet.to_pandas()


def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("census_path"))):
        raise RuntimeError("RP 2022 data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("census_path")))
