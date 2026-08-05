import pandas as pd
import geopandas as gpd


def configure(context):
    context.stage("synthesis.locations.secondary_FR")
    context.config("generate_outbound_flows")

    if context.config("generate_outbound_flows"):
        context.stage("data.locations_CH.secondary")


def execute(context):
    destinations = context.stage("synthesis.locations.secondary_FR")

    if context.config("generate_outbound_flows"):
        destinations_CH = context.stage("data.locations_CH.secondary")
        destinations    = pd.concat([destinations, destinations_CH])

    # Several raw BPE records can collapse onto the same location_id (e.g.
    # one SIRET, several sites); merge them since MATSim needs unique ids.
    offers_columns = [c for c in destinations.columns if c.startswith("offers_")]
    other_columns  = [c for c in destinations.columns if c not in offers_columns and c != "location_id"]

    aggregation = {c: "any" for c in offers_columns}
    aggregation.update({c: "first" for c in other_columns})

    crs = destinations.crs
    destinations = destinations.groupby("location_id", as_index = False).agg(aggregation)
    destinations = gpd.GeoDataFrame(destinations, geometry = "geometry", crs = crs)

    print(destinations.head())
    print(destinations.tail())

    return destinations