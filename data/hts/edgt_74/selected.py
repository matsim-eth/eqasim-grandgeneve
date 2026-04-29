def configure(context):
    hts = context.config("hts")

    day = context.config("specific_day_scenario", "workday")

    if day in ["weekend", "saturday", "sunday"] or "saturday" in day or "sunday" in day:
        raise RuntimeError(f"Impossible HTS and day combination: {day} and {hts}. The Annemasse EDGT survey doesn't cover weekends. Please select ENTD as HTS or choose another day.")

    if hts == "edgt_74":
        edgt74_version = context.config("edgt74_version", default = "adisp")
        if edgt74_version == "tpg":
            context.stage("data.hts.edgt_74.tpg.reweighted", alias = "edgt74")
        elif edgt74_version == "adisp":
            context.stage("data.hts.edgt_74.adisp.reweighted", alias = "edgt74")
        else:
            raise RuntimeError("Unknown EDGT version: %s" % edgt74_version)

def execute(context):
    return context.stage("edgt74")