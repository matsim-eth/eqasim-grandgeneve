import analysis.debug.commute_flows as commute_flows

"""
Shows the commuting flows of every employed census agent, with the work
commune attached before statistical matching by
synthesis.population.spatial.primary.work_first (MOBPRO destinations by home
commune, census commute mode and census place of work), in the same format as
analysis.debug.commute_flows (heatmap, interactive map, CSV).

Unlike the pipeline output (analysis.debug.commute_flows), this counts every
employed agent, not only those whose matched HTS chain contains a work trip,
so its total is close to the MOBPRO total (workers drawn outside the
modelled area are left out, as in the MOBPRO reference). Homes are those of
the pipeline, so this also shows the effect of home_zone_method /
home_zone_use_cantons.
"""

def configure(context):
    commute_flows.configure(context)
    context.stage("synthesis.population.spatial.primary.work_first")

    context.config("home_zone_method", "population")
    context.config("home_zone_use_cantons", False)


def execute(context):
    home_zone_method = context.config("home_zone_method")
    use_cantons = context.config("home_zone_use_cantons")

    source = {
        "name": "Census + workplace",
        "short": "census",
        "description": "census-employed agents with the work commune of synthesis.population.spatial.primary.work_first (home_zone_method: %s, home_zone_use_cantons: %s)" % (
            home_zone_method, str(use_cantons).lower()),
        "unit": "workers",
        "column": "census",
    }

    df_home = commute_flows.home_communes(context)
    # Workers outside the modelled area are not in the MOBPRO reference either
    df_work = context.stage("synthesis.population.spatial.primary.work_first")
    df_work = df_work[~df_work["work_outside"]][["person_id", "work_commune_id"]].rename(
        columns = { "work_commune_id": "destination_id" })

    file_stem = "debug_commute_flows_census"
    if home_zone_method != "population":
        file_stem += "_home-%s" % home_zone_method
    if use_cantons:
        file_stem += "_cantons"

    df, scale = commute_flows.build_flows(context, df_home.merge(df_work, on = "person_id"), set(df_home["origin_id"]), source)
    commute_flows.write_outputs(context, df, scale, source, file_stem)
    return df
