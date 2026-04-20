import pandas as pd
import os
import zipfile
import polars as pl
"""
This stage loads the raw data from the French service registry.
"""

def configure(context):
    context.config("data_path")
    context.config("bpe_path", "bpe_2024/BPE24.parquet")
    context.stage("data.spatial.codes")

def execute(context):
    df_codes = context.stage("data.spatial.codes").copy()
    df_codes = df_codes[df_codes["iris_id"]!="CH"]

    requested_departements = df_codes["departement_id"].unique()

    with context.progress(label = "Reading BPE ...") as progress:
        parquet = pl.read_parquet("{}/{}".format(context.config("data_path"), context.config("bpe_path")), columns = [ "CAPACITE",
                        "DCIRIS", "LAMBERT_X", "LAMBERT_Y",
                        "TYPEQU", "DEPCOM", "DEP"
                    ],
                )

        parquet  = parquet.cast( dict(DEPCOM = str, DEP = str, DCIRIS = str))
        dpts_str = [str(dep) for dep in requested_departements]
        parquet  = parquet.filter(pl.col("DEP").cast(pl.Utf8).is_in(dpts_str))

        progress.update(len(parquet))

    return parquet.to_pandas()

def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("bpe_path"))):
        raise RuntimeError("BPE data is not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), context.config("bpe_path")))
