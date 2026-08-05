import geopandas as gpd


def configure(context):
    context.config("swiss_cantons")
    context.config("cantons")


def execute(context):
    # Load data
    data_path = context.config("swiss_cantons")
    cantons   = context.config("cantons")

    df = gpd.read_file(data_path, engine="pyogrio", encoding="utf8").to_crs("epsg:2056")
    df.crs = "epsg:2056"

    df = df.rename({"KANTONSNUM": "canton_id", "NAME": "canton_name"}, axis=1)
    df = df[["canton_id", "canton_name", "geometry"]]

    if cantons: 
        return df[df["canton_name"].isin(cantons)]
    
    return df



