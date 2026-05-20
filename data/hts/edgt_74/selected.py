import pandas as pd

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
            context.stage("data.hts.edgt_74.adisp_annemasse.reweighted")
            context.stage("data.hts.edgt_74.adisp_annecy.reweighted")
        else:
            raise RuntimeError("Unknown EDGT version: %s" % edgt74_version)

def execute(context):
    hts = context.config("hts")
    if hts == "edgt_74":
        edgt74_version = context.config("edgt74_version")
        if edgt74_version == "adisp":
            annemasse_hhl, annemasse_persons, annemasse_trips = context.stage("data.hts.edgt_74.adisp_annemasse.reweighted")
            annecy_hhl, annecy_persons, annecy_trips          = context.stage("data.hts.edgt_74.adisp_annecy.reweighted")

            annemasse_hhl["edgt_area"]     = "annemasse"
            annemasse_persons["edgt_area"] = "annemasse"
            annemasse_trips["edgt_area"]   = "annemasse"
            annecy_hhl["edgt_area"]        = "annecy"
            annecy_persons["edgt_area"]    = "annecy"
            annecy_trips["edgt_area"]      = "annecy"

            households = pd.concat([annemasse_hhl, annecy_hhl])
            persons    = pd.concat([annemasse_persons, annecy_persons])
            trips      = pd.concat([annemasse_trips, annecy_trips])

            return households, persons, trips

    return context.stage("edgt74")