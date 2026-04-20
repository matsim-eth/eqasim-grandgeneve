def configure(context):
    context.config("generate_outbound_flows", "False")

    if context.config("generate_outbound_flows"):
        context.stage("data.spatial.ch.cantons")
        context.stage("data.spatial.ch.municipalities")


def execute(context):
    if context.config("generate_outbound_flows"):
        cantons        = context.stage("data.spatial.ch.cantons")
        municipalities = context.stage("data.spatial.ch.municipalities")

        canton_ids     = cantons["canton_id"].values.tolist()
        municipalities = municipalities[municipalities["canton_id"].isin(canton_ids)].copy()

        return cantons, municipalities
    
    return None