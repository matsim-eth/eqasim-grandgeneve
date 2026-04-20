import pandas as pd

"""
This stage provides a list of work places that serve as potential locations for
work activities. It is derived from the SIRENE enterprise database.

Municipalities which do not have any registered enterprise receive a fake work
place at their centroid to be in line with INSEE OD data.
"""

def configure(context):
    context.stage("synthesis.locations.work_FR")
    context.config("generate_outbound_flows")

    if context.config("generate_outbound_flows"):
        context.stage("data.locations_CH.work")


def execute(context):
    workplaces = context.stage("synthesis.locations.work_FR")

    if context.config("generate_outbound_flows"):
        workplaces_CH = context.stage("data.locations_CH.work")
        workplaces    = pd.concat([workplaces, workplaces_CH])

    print(workplaces.head())
    print(workplaces.tail())

    return workplaces