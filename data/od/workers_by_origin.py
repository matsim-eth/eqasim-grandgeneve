import polars as pl

"""
Counts MOBPRO workers per residence commune, place of work (as in the census
ILT: home commune, elsewhere in France, abroad) and commute mode, over all
destinations. Unlike data.od.raw, destinations outside the requested area are
kept, so that these counts describe every worker living in a commune. Used to
place census households whose commune is unknown
(synthesis.population.spatial.home.zones, home_zone_method = "ilt").
"""

def configure(context):
    context.config("data_path")
    context.config("od_pro_path", "rp_2022/RP2023_mobpro.parquet")

    context.stage("data.spatial.codes")


# MOBPRO TRANS -> commute_mode, as in data.od.cleaned
COMMUTE_MODES = { 1: "no transport", 2: "walk", 3: "bike", 4: "car", 5: "car", 6: "pt" }


def execute(context):
    df_codes = context.stage("data.spatial.codes")
    requested_communes = [str(commune_id) for commune_id in df_codes["commune_id"].unique()]

    df = pl.read_parquet("{}/{}".format(context.config("data_path"), context.config("od_pro_path")),
        columns = ["COMMUNE", "ARM", "DCLT", "TRANS", "IPONDI"])

    df = df.with_columns(
        pl.col(["COMMUNE", "ARM", "DCLT"]).cast(pl.String),
        pl.col("TRANS").cast(pl.Int32), pl.col("IPONDI").cast(pl.Float64))

    # Arrondissements, as in data.od.cleaned
    df = df.with_columns(pl.when(pl.col("ARM").str.contains("Z")).then(pl.col("COMMUNE")).otherwise(pl.col("ARM")).alias("origin_id"))
    df = df.filter(pl.col("origin_id").is_in(requested_communes))

    df = df.with_columns(
        pl.when(pl.col("DCLT") == pl.col("origin_id")).then(pl.lit("home_commune"))
            .when(pl.col("DCLT") == "99999").then(pl.lit("abroad"))
            .otherwise(pl.lit("france")).alias("place_of_work"),
        pl.col("TRANS").replace_strict(COMMUTE_MODES, default = None).alias("commute_mode"))

    df = df.group_by(["origin_id", "place_of_work", "commute_mode"]).agg(pl.col("IPONDI").sum().alias("weight"))
    return df.to_pandas()
