import pandas as pd

def configure(context):
    context.stage("synthesis.locations.education_FR")
    context.config("generate_outbound_flows")

    if context.config("generate_outbound_flows"):
        context.stage("data.locations_CH.education")


def execute(context):
    educplaces = context.stage("synthesis.locations.education_FR")

    if context.config("generate_outbound_flows"):
        educplaces_CH = context.stage("data.locations_CH.education")
        educplaces    = pd.concat([educplaces, educplaces_CH])

    print(educplaces.head())
    print(educplaces.tail())

    return educplaces