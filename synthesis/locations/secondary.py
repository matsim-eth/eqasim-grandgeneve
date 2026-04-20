import pandas as pd

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

    print(destinations.head())
    print(destinations.tail())

    return destinations