import pandas as pd
import geopandas as gpd
import os

"""
This stage loads the raw data of the specified HTS (EDGT Lyon).

Adapted from the first implementation by Valentin Le Besond (IFSTTAR Nantes)
"""

def configure(context):
    context.config("data_path")
    context.config("output_path")


HOUSEHOLD_COLUMNS = {
    "ECH": str, "ZFM": str, # id
    "M6": int, "M21": int, "M14": int,  # number_of_cars, number_of_bikes, number_of_motorbikes
    "COE0": float # weights
}

PERSON_COLUMNS = {
    "ECH": str, "PER": int, "ZFP": str, # id
    "PENQ": str, # respondents of travel questionary section
    "P2": int, "P4": int, # sex, age
    "P3": str, # position in household (reference person, spouse, child, ...)
    "P8": str, # highest education level attained
    "P9": str, # employed, studies
    "P7": str, "P12": str, # has_license, has_pt_subscription
    "PCSC": str, # socioprofessional_class
    "P14": str, # teleworks / studies from home
    "P16": str, # car availability for the commute
    "P19": str, "P20": str, "P21": str, "P22": str, "P23": str, "P24": str, # frequency of use: walk, bike (driver), motorized 2-wheeler (driver), car (driver), car (passenger), urban PT
    "P25": str, # made a trip "yesterday" within the survey area / absent / etc.
    "P26": str, # work situation "yesterday" (worked out, teleworked, on leave, ...)
    "DP15": str, "GP5": str, # distance to and commune of the usual place of work / study (GP5 99xxx: abroad)
    "COEP": float, "COE1": float, # weights,
    "JOUR": str # day of the week, 1 = monday to 5 = friday
}

TRIP_COLUMNS = {
    "ECH": str, "PER": int, "NDEP": int, "ZFD": str, # id
    "D2A": int, "D5A": int, # preceding_purpose, following_purpose
    "D3": str, "D7": str, # origin_zone, destination_zone
    "D4": int, "D8": int, # time_departure, time_arrival
    "MODP": int, "D11": int, "D12": int # mode, euclidean_distance, routed_distance
}


# Local questions only found in the original (non-standardized) person files:
# P12A = network of the PT subscription held yesterday
ORIGINAL_PERSON_COLUMNS = ["ZFP", "ECH", "PER", "P12A"]


def attach_original_person_columns(df_persons, paths):
    """Adds the ORIGINAL_PERSON_COLUMNS of the original person files to the
    standard persons. The two share ZFP/ECH/PER, but the original files
    don't zero-pad ZFP, so the keys are compared as integers."""
    keys = ["ZFP", "ECH", "PER"]

    df_original = pd.concat([
        pd.read_csv(path, sep = ";", usecols = ORIGINAL_PERSON_COLUMNS, dtype = str)
        for path in paths
    ])
    df_original = df_original.astype({ key: int for key in keys })
    assert not df_original.duplicated(keys).any()

    df_keys = df_persons[keys].astype(int)
    df_original = pd.merge(df_keys, df_original, on = keys, how = "left", indicator = True)
    assert (df_original["_merge"] == "both").all(), "Persons missing from the original EDGT files"

    df_persons = df_persons.copy()
    for column in ORIGINAL_PERSON_COLUMNS:
        if not column in keys:
            df_persons[column] = df_original[column].values

    return df_persons


def execute(context):
    data_path   = context.config("data_path")
    edgt_path   = f"{data_path}/edgt_2017/edgt_2017_annemasse/adisp"

    df_households1 = pd.read_csv(f"{edgt_path}/lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Face_a_face/annemasse_2016_std_faf_men.csv", sep = ";", usecols = list(HOUSEHOLD_COLUMNS.keys()), dtype = HOUSEHOLD_COLUMNS)
    df_households2 = pd.read_csv(f"{edgt_path}/lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Telephonique/annemasse_2016_std_tel_men.csv", sep = ";", usecols = list(HOUSEHOLD_COLUMNS.keys()), dtype = HOUSEHOLD_COLUMNS)

    df_persons1 = pd.read_csv(f"{edgt_path}/lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Face_a_face/annemasse_2016_std_faf_pers.csv", sep = ";", usecols = list(PERSON_COLUMNS.keys()), dtype = PERSON_COLUMNS)
    df_persons2 = pd.read_csv(f"{edgt_path}/lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Telephonique/annemasse_2016_std_tel_pers.csv", sep = ";", usecols = list(PERSON_COLUMNS.keys()), dtype = PERSON_COLUMNS)

    df_trips1 = pd.read_csv(f"{edgt_path}/lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Face_a_face/annemasse_2016_std_faf_depl.csv", sep = ";", usecols = list(TRIP_COLUMNS.keys()), dtype = TRIP_COLUMNS)
    df_trips2 = pd.read_csv(f"{edgt_path}/lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Telephonique/annemasse_2016_std_tel_depl.csv", sep = ";", usecols = list(TRIP_COLUMNS.keys()), dtype = TRIP_COLUMNS)

    df_households = pd.concat([df_households1, df_households2])
    df_persons    = pd.concat([df_persons1, df_persons2])
    df_persons    = attach_original_person_columns(df_persons, [
        f"{edgt_path}/{ORIGINAL_PERSON_FILES[0]}", f"{edgt_path}/{ORIGINAL_PERSON_FILES[1]}"
    ])
    df_trips      = pd.concat([df_trips1, df_trips2])

    print("----------- HOUSEHOLDS -------------")
    print(df_households.head())
    print(df_households.columns)
    print("\n")

    print("----------- PERSONS -------------")
    print(df_persons.head())
    print(df_persons.columns)
    print("\n")

    print("----------- TRIPS -------------")
    print(df_trips.head())
    print(df_trips.columns)
    print("\n")

    spatial1 = gpd.read_file(f"{edgt_path}/lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZF.TAB")
    spatial1 = spatial1.set_crs(epsg = 2154, allow_override=True)
    spatial1.columns = spatial1.columns.str.lower()
    spatial1 = spatial1[["geometry", "zonefine", "depcom"]]
    spatial1.columns = ["geometry", "zone_id", "departement"]
    spatial1["departement_id"] = spatial1["departement"].astype(str).str[:2]
    spatial1["label"] = "intern"
    spatial1 = spatial1[["geometry", "zone_id", "departement_id", "label"]]

    spatial2 = gpd.read_file(f"{edgt_path}/lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZonesExternes.TAB")
    spatial2 = spatial2.set_crs(epsg = 2154, allow_override=True)
    spatial2.columns = spatial2.columns.str.lower()
    spatial2 = spatial2[["geometry", "num_zf", "nom_d10"]]
    spatial2["label"] = "extern"
    spatial2.columns = ["geometry", "zone_id", "departement_id", "label"]

    df_spatial = gpd.GeoDataFrame(pd.concat([spatial1, spatial2]), crs=spatial1.crs)

    return df_households, df_persons, df_trips, df_spatial


ORIGINAL_PERSON_FILES = [
    "lil-1212/lil-1212.csv/Csv/Fichiers_Original_Face_a_face/annemasse_2016_ori_faf_pers.csv",
    "lil-1212/lil-1212.csv/Csv/Fichiers_Original_Telephonique/annemasse_2016_ori_tel_pers.csv",
]

FILES = ORIGINAL_PERSON_FILES + [
    "lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Face_a_face/annemasse_2016_std_faf_men.csv",
    "lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Telephonique/annemasse_2016_std_tel_men.csv",
    "lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Face_a_face/annemasse_2016_std_faf_pers.csv",
    "lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Telephonique/annemasse_2016_std_tel_pers.csv",
    "lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Face_a_face/annemasse_2016_std_faf_depl.csv",
    "lil-1212/lil-1212.csv/Csv/Fichiers_Standard_Telephonique/annemasse_2016_std_tel_depl.csv",
    "lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZF.TAB",
    "lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZF.ID",
    "lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZF.IND",
    "lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZF.MAP",
    "lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZonesExternes.TAB",
    "lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZonesExternes.ID",
    "lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZonesExternes.IND",
    "lil-1212/lil-1212.csv/Doc/SIG/EDGTFVG2016_ZonesExternes.MAP",
]


def validate(context):
    data_path   = context.config("data_path")
    edgt_path   = f"{data_path}/edgt_2017/edgt_2017_annemasse/adisp"
    
    for name in FILES:
        current_path = f"{edgt_path}/{name}"
        if not os.path.exists(current_path):
            raise RuntimeError("File missing from EDGT: %s" % current_path)

    return [
        os.path.getsize(f"{edgt_path}/{name}")
        for name in FILES
    ]
