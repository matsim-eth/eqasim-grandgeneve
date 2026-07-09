def configure(context):
    day = context.config("specific_day_scenario", "workday")

    if day in ["weekend", "saturday", "sunday"] or "saturday" in day or "sunday" in day:
        raise RuntimeError(f"Impossible day for the EDGT 74 survey: {day}. The Annemasse EDGT survey doesn't cover weekends. Please select ENTD as HTS or choose another day.")

    edgt74_version = context.config("edgt74_version", default = "adisp")

    if edgt74_version == "tpg":
        context.stage("data.hts.edgt_74.tpg.reweighted")
    elif edgt74_version == "adisp":
        context.stage("data.hts.edgt_74.adisp_merge.merge")
    else:
        raise RuntimeError("Unknown EDGT version: %s" % edgt74_version)

def execute(context):
    edgt74_version = context.config("edgt74_version")

    if edgt74_version == "tpg":
        return context.stage("data.hts.edgt_74.tpg.reweighted")
    elif edgt74_version == "adisp":
        return context.stage("data.hts.edgt_74.adisp_merge.merge")
    else:
        raise RuntimeError("Unknown EDGT version: %s" % edgt74_version)