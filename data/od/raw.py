import os
import polars as pl
"""
Loads raw OD data from French census data.
"""

def configure(context):
    context.config("generate_outbound_flows", "False")
    context.config("data_path")
    context.config("od_pro_path", "rp_2022/RP2022_mobpro.parquet")
    context.config("od_sco_path", "rp_2022/RP2022_mobsco.parquet")

    context.stage("data.spatial.codes")
    if context.config("generate_outbound_flows"):
        context.stage("data.od.read_CH_mun_insee_names")



def execute(context):
    df_codes           = context.stage("data.spatial.codes")
    requested_communes = list(set(df_codes["commune_id"].astype(str).values.tolist()))
    insee_to_ch_id     = context.stage("data.od.read_CH_mun_insee_names")

    # First, load work
    with context.progress(label = "Reading work flows ...") as progress:
        COLUMNS_DTYPES = {
            "COMMUNE": pl.String, 
            "ARM"    : pl.String, 
            "TRANS"  : pl.Int32,
            "IPONDI" : pl.Float32, 
            "DCLT"   : pl.String,
            "DCFLT"  : pl.String
        }

        parquet = pl.read_parquet("{}/{}".format(context.config("data_path"), context.config("od_pro_path")),columns=  list(COLUMNS_DTYPES.keys()))
        
        parquet = parquet.cast(COLUMNS_DTYPES)

        if context.config("generate_outbound_flows"):
            parquet = parquet.with_columns(pl.when(pl.col("DCLT") == "99999").then(pl.col("DCFLT")).otherwise(pl.col("DCLT")).alias("DCLT"))
            commutes_to_CH_filter = parquet["DCLT"].str.starts_with("SU")
            d = insee_to_ch_id[["insee_code", "municipality_id"]].set_index("insee_code")["municipality_id"].to_dict()
            parquet = parquet.with_columns(pl.when(commutes_to_CH_filter).then(pl.col("DCLT").replace(d)).otherwise(pl.col("DCLT")).alias("DCLT"))
        
        parquet = parquet.drop("DCFLT")

        rq_com  = [str(commune) for commune in requested_communes]
        parquet = parquet.filter(((pl.col("COMMUNE").is_in(rq_com)) | (pl.col("ARM").is_in(rq_com))) & (pl.col("DCLT").is_in(rq_com)))
        
        progress.update(len(parquet))

    work = parquet.to_pandas()

    # Second, load education
    with context.progress(label = "Reading education flows ...") as progress:
        COLUMNS_DTYPES = {
            "COMMUNE" : pl.String, 
            "ARM"     : pl.String, 
            "IPONDI"  : pl.Float32,
            "DCETUF"  : pl.String,
            "AGEREV10": pl.String,
            "DCETUE"  : pl.String,
        }

        parquet = pl.read_parquet("{}/{}".format(context.config("data_path"), context.config("od_sco_path")), columns=  COLUMNS_DTYPES.keys())

        parquet = parquet.cast(COLUMNS_DTYPES)

        if context.config("generate_outbound_flows"):
            parquet = parquet.with_columns(pl.when(pl.col("DCETUF") == "99999").then(pl.col("DCETUE")).otherwise(pl.col("DCETUF")).alias("DCETUF"))
            commutes_to_CH_filter = parquet["DCETUF"].str.starts_with("SU")

            d = insee_to_ch_id[["insee_code", "municipality_id"]].set_index("insee_code")["municipality_id"].to_dict()
            parquet = parquet.with_columns(pl.when(commutes_to_CH_filter).then(pl.col("DCETUF").replace(d)).otherwise(pl.col("DCETUF")).alias("DCETUF"))
        
        parquet = parquet.drop("DCETUE")

        parquet = parquet.filter(((pl.col("COMMUNE").is_in(rq_com)) | (pl.col("ARM").is_in(rq_com))) & (pl.col("DCETUF").is_in(rq_com)))

        progress.update(len(parquet))

    education = parquet.to_pandas()

    print(work[work["DCLT"].str.startswith("CH")].groupby("DCLT", as_index = False)["IPONDI"].sum())
    print(education[education["DCETUF"].str.startswith("CH")].groupby("DCETUF", as_index = False)["IPONDI"].sum())

    return work, education


def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("od_pro_path"))):
        raise RuntimeError("RP MOBPRO data is not available")

    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("od_sco_path"))):
        raise RuntimeError("RP MOBSCO data is not available")

    return [
        os.path.getsize("%s/%s" % (context.config("data_path"), context.config("od_pro_path"))),
        os.path.getsize("%s/%s" % (context.config("data_path"), context.config("od_sco_path")))
    ]
