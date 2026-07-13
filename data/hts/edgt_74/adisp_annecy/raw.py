from pandas.core.frame import DataFrame
import numpy as np
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
    "P9": str, # employed, studies
    "P7": str, "P12": str, # has_license, has_pt_subscription
    "PCSC": str, # socioprofessional_class
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


def execute(context):
    data_path   = context.config("data_path")
    edgt_path   = f"{data_path}/edgt_2017/edgt_2017_annecy/adisp"

    df_households1 = pd.read_csv(f"{edgt_path}/lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_FaF/annecy_2017_std_faf_men.csv", sep = ";", usecols = list(HOUSEHOLD_COLUMNS.keys()), dtype = HOUSEHOLD_COLUMNS)
    df_households2 = pd.read_csv(f"{edgt_path}/lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_Tel/annecy_2017_std_tel_men.csv", sep = ";", usecols = list(HOUSEHOLD_COLUMNS.keys()), dtype = HOUSEHOLD_COLUMNS)

    df_persons1 = pd.read_csv(f"{edgt_path}/lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_FaF/annecy_2017_std_faf_pers.csv", sep = ";", usecols = list(PERSON_COLUMNS.keys()), dtype = PERSON_COLUMNS)
    df_persons2 = pd.read_csv(f"{edgt_path}/lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_Tel/annecy_2017_std_tel_pers.csv", sep = ";", usecols = list(PERSON_COLUMNS.keys()), dtype = PERSON_COLUMNS)

    df_trips1 = pd.read_csv(f"{edgt_path}/lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_FaF/annecy_2017_std_faf_depl.csv", sep = ";", usecols = list(TRIP_COLUMNS.keys()), dtype = TRIP_COLUMNS)
    df_trips2 = pd.read_csv(f"{edgt_path}/lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_Tel/annecy_2017_std_tel_depl.csv", sep = ";", usecols = list(TRIP_COLUMNS.keys()), dtype = TRIP_COLUMNS)

    df_households = pd.concat([df_households1, df_households2])
    df_persons    = pd.concat([df_persons1, df_persons2])
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

    spatial1 = gpd.read_file(f"{edgt_path}/lil-1315/lil-1315.csv/Doc/SIG/EDGT_Reste74_2017_ZF_404009.TAB")
    spatial1 = spatial1.set_crs(epsg = 2154, allow_override = True)
    spatial1.columns = spatial1.columns.str.lower()
    spatial1 = spatial1[["geometry", "zf"]]
    spatial1["departement"] = 74
    spatial1.columns = ["geometry", "zone_id", "departement"]
    spatial1["departement_id"] = spatial1["departement"].astype(str).str[:2]
    spatial1 = spatial1[["geometry", "zone_id", "departement_id"]]

    df_spatial = gpd.GeoDataFrame(spatial1, crs=spatial1.crs)

    return df_households, df_persons, df_trips, df_spatial


FILES = [
    "lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_FaF/annecy_2017_std_faf_men.csv",
    "lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_Tel/annecy_2017_std_tel_men.csv",
    "lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_FaF/annecy_2017_std_faf_pers.csv",
    "lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_Tel/annecy_2017_std_tel_pers.csv",
    "lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_FaF/annecy_2017_std_faf_depl.csv",
    "lil-1315/lil-1315.csv/Csv/Annecy_2017_Standard_Tel/annecy_2017_std_tel_depl.csv",
    "lil-1315/lil-1315.csv/Doc/SIG/EDGT_Reste74_2017_ZF_404009.TAB",
    "lil-1315/lil-1315.csv/Doc/SIG/EDGT_Reste74_2017_ZF_404009.ID",
    "lil-1315/lil-1315.csv/Doc/SIG/EDGT_Reste74_2017_ZF_404009.IND",
    "lil-1315/lil-1315.csv/Doc/SIG/EDGT_Reste74_2017_ZF_404009.MAP",
]


def validate(context):
    data_path   = context.config("data_path")
    edgt_path   = f"{data_path}/edgt_2017/edgt_2017_annecy/adisp"
    
    for name in FILES:
        current_path = f"{edgt_path}/{name}"
        if not os.path.exists(current_path):
            raise RuntimeError("File missing from EDGT: %s" % current_path)

    return [
        os.path.getsize(f"{edgt_path}/{name}")
        for name in FILES
    ]
