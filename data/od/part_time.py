import os
import pandas as pd

"""
Distribution of the employment time of part-time workers, from the French
labour force survey (Enquete Emploi en Continu, EEC). TXTPPRED is only
reported for part-time workers (TPPRED = 2) and gives a coarse class, to
which an employment time is assigned (assumptions).
"""

# TXTPPRED class -> assumed employment time (share of a full-time job)
#   1: less than half-time (<50%)   2: half-time (50%)   3: between 50 and 80%
#   4: 80%                          5: more than 80%
EMPLOYMENT_TIME = { 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.8, 5: 0.9 }

PART_TIME = 2

def configure(context):
    context.config("data_path")
    context.config("eec_path", "rp_2022/FD_csv_EEC25.csv")

def execute(context):
    path = "{}/{}".format(context.config("data_path"), context.config("eec_path"))
    df = pd.read_csv(path, sep = ";", usecols = ["TPPRED", "TXTPPRED", "EXTRIAN"], low_memory = False)

    df = df[(df["TPPRED"] == PART_TIME) & df["TXTPPRED"].notna()]
    assert set(df["TXTPPRED"].astype(int).unique()) <= set(EMPLOYMENT_TIME.keys())

    # Weighted share of each class; returned as a Series: employment time -> probability
    shares = df.groupby(df["TXTPPRED"].astype(int))["EXTRIAN"].sum()
    shares = shares / shares.sum()
    shares.index = shares.index.map(EMPLOYMENT_TIME)
    shares.index.name = "employment_time"

    print("Part-time employment time distribution (EEC, %d part-time records):" % len(df))
    print(shares.round(3).to_string())

    return shares

def validate(context):
    path = "{}/{}".format(context.config("data_path"), context.config("eec_path"))

    if not os.path.exists(path):
        raise RuntimeError("EEC data is not available: %s" % path)

    return os.path.getsize(path)
