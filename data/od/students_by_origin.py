import polars as pl

"""
Counts MOBSCO students per residence commune, place of study (as in the census
ILETUD: home commune, elsewhere in France, abroad) and age range, over all
destinations. Unlike data.od.raw, destinations outside the requested area are
kept, so that these counts describe every student living in a commune. Used
to draw places of study outside the modelled area
(synthesis.population.spatial.primary.education_first).
"""

def configure(context):
    context.config("data_path")
    context.config("od_sco_path", "rp_2022/RP2022_mobsco.parquet")

    context.stage("data.spatial.codes")


def age_range(age):
    """MOBSCO AGEREV10 -> age_range, as in data.od.cleaned."""
    return (pl.when(age <= 6).then(pl.lit("primary_school"))
        .when(age == 11).then(pl.lit("middle_school"))
        .when(age == 15).then(pl.lit("high_school"))
        .otherwise(pl.lit("higher_education")))


def execute(context):
    df_codes = context.stage("data.spatial.codes")
    requested_communes = [str(commune_id) for commune_id in df_codes["commune_id"].unique()]

    df = pl.read_parquet("{}/{}".format(context.config("data_path"), context.config("od_sco_path")),
        columns = ["COMMUNE", "ARM", "DCETUF", "AGEREV10", "IPONDI"])

    df = df.with_columns(
        pl.col(["COMMUNE", "ARM", "DCETUF"]).cast(pl.String),
        pl.col("AGEREV10").cast(pl.Int32), pl.col("IPONDI").cast(pl.Float64))

    # Arrondissements, as in data.od.cleaned
    df = df.with_columns(pl.when(pl.col("ARM").str.contains("Z")).then(pl.col("COMMUNE")).otherwise(pl.col("ARM")).alias("origin_id"))
    df = df.filter(pl.col("origin_id").is_in(requested_communes))

    df = df.with_columns(
        pl.when(pl.col("DCETUF") == pl.col("origin_id")).then(pl.lit("home_commune"))
            .when(pl.col("DCETUF") == "99999").then(pl.lit("abroad"))
            .otherwise(pl.lit("france")).alias("place_of_study"),
        age_range(pl.col("AGEREV10")).alias("age_range"))

    df = df.group_by(["origin_id", "place_of_study", "age_range"]).agg(pl.col("IPONDI").sum().alias("weight"))
    return df.to_pandas()
