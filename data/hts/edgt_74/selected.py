import pandas as pd
import numpy as np

def configure(context):
    hts = context.config("hts")

    if hts == "edgt_74":
        edgt74_version = context.config("edgt74_version", default = "cerema")
        if edgt74_version == "tpg":
            context.stage("data.hts.edgt_74.tpg.reweighted", alias = "edgt74")
        elif edgt74_version == "cerema":
            context.stage("data.hts.edgt_74.cerema.reweighted", alias = "edgt74")
        else:
            raise RuntimeError("Unknown EDGT version: %s" % edgt74_version)

def execute(context):
    return context.stage("edgt74")