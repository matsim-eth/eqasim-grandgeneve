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
    "NUM_MEN": str, # id
    "NB_AUTO": str, "NB_VELO": str, "NB_2RM": str,  # number_of_cars, number_of_bikes, number_of_motorbikes
    "POND_MEN": str # weights
}

PERSON_COLUMNS = {
    "NUM_MEN": str, "NUM_PERS": str, # id
    "DEPL_OUI_NON": str, # respondents of travel questionary section
    "SEXE": int, "AGE": int, # sex, age
    "STATUT_TRAVAIL": str, # employed, studies
    "PERMIS_CONDUIRE": str, "ABO_TC": str, # has_license, has_pt_subscription
    "PROFESSION": str, # socioprofessional_class
    "POND_PERS": str, # weights
    "JOUR_SEM_DEPL": int # day of the week, 1 = monday to 5 = friday
}

TRIP_COLUMNS = {
    "NUM_MEN_FR": str, "NUM_PERS_FR": str, "NUM_DEPL_FR_PROV": str, # id
    "D2A": str, "D5A": str, # preceding_purpose, following_purpose
    "LIEU_DEPART_FR": str, "LIEU_ARRIVEE_FR": str, # origin_zone, destination_zone
    "D4": int, "D8": int, # time_departure, time_arrival
    "MODP": int, "DOIB": int, "DIST": int # mode, euclidean_distance, routed_distance
}

PURPOSE_TO_KEY = {"DOMICILE": 1,
                  "RÉSIDENCE SECONDAIRE, AUTRE DOMICILE": 2,
                  " TRAVAIL SUR LE LIEU D\x92EMPLOI DÉCLARÉ": 11,
                  " TRAVAIL SUR UN AUTRE LIEU - TÉLÉTRAVAIL": 12,
                  " TRAVAIL SUR UN AUTRE LIEU - HORS TÉLÉTRAVAIL": 13,
                  " ETRE GARDÉ (NOURRICE, CRÈCHE,,,)": 21,
                  " ETUDIER À L'ÉCOLE MATERNELLE ET PRIMAIRE (SUR LE LIEU DÉCLARÉ)": 22,
                  " ETUDIER AU COLLÈGE (SUR LE LIEU DÉCLARÉ)": 23,
                  " ETUDIER AU LYCÉE (SUR LE LIEU DÉCLARÉ) ": 24,
                  " ETUDIER À L'UNIVERSITÉ ET GRANDES ÉCOLES (SUR LE LIEU DÉCLARÉ) ": 25,
                  " ETUDIER SUR UN AUTRE LIEU DÉCLARÉ (ECOLE MATERNELLE ET PRIMAIRE)": 26,
                  " ETUDIER SUR UN AUTRE LIEU DÉCLARÉ (COLLÈGE) ": 27,
                  " ETUDIER SUR UN AUTRE LIEU DÉCLARÉ (LYCÉE)": 28,
                  " ETUDIER SUR UN AUTRE LIEU DÉCLARÉ (UNIVERSITÉ ET GRANDES ÉCOLES) ": 29,
                  " VISITE D\x92UN MAGASIN, D\x92UN CENTRE COMMERCIAL OU D\x92UN MARCHÉ DE PLEIN VENT SANS EFFECTUER D\x92ACHAT": 30,
                  " RÉALISER PLUSIEURS MOTIFS EN CENTRE COMMERCIAL": 31,
                  " FAIRE DES ACHATS EN GRAND MAGASIN, SUPERMARCHÉ, HYPERMARCHÉ ET LEURS GALERIES MARCHANDES": 32,
                  " FAIRE DES ACHATS EN PETIT ET MOYEN COMMERCE ET ": 33,
                  "34": 34,
                  "35": 35,
                  "41": 41,
                  "42": 42,
                  "43": 43,
                  "50": 51, 
                  "51": 51,
                  "52": 52,
                  "53": 53,
                  "54": 54,
                  "61": 61,
                  "62": 62,
                  "63": 63,
                  "64": 64,
                  "65": 63,
                  "66": 64, 
                  "67": 63, 
                  "68": 64, 
                  "72": 72,
                  "73": 73,
                  "76": 72, 
                  "77": 73, 
                  "81": 81,
                  "82": 82,
                  "91": 91}


def execute(context):
    data_path   = context.config("data_path")
    edgt_path   = f"{data_path}/edgt_2017_annemasse/TPG"

    df_households = pd.read_csv(f"{edgt_path}/Menages_EMD.csv", sep = ";", usecols = list(HOUSEHOLD_COLUMNS.keys()), dtype = HOUSEHOLD_COLUMNS)
    df_persons    = pd.read_csv(f"{edgt_path}/Personnes_EMD.csv", sep = ";", usecols = list(PERSON_COLUMNS.keys()), dtype = PERSON_COLUMNS)
    df_trips      = pd.read_csv(f"{edgt_path}/deplacements_emd_full_OCT.csv", sep = ";", usecols = list(TRIP_COLUMNS.keys()), dtype = TRIP_COLUMNS, encoding = "latin1")

    df_trips["D2A"] = df_trips["D2A"].map(PURPOSE_TO_KEY).fillna(91).astype(int)
    df_trips["D5A"] = df_trips["D5A"].map(PURPOSE_TO_KEY).fillna(91).astype(int)

    df_households["POND_MEN"] = df_households["POND_MEN"].str.replace(",", ".").astype(float)
    for veh_col in ["NB_AUTO", "NB_VELO", "NB_2RM"]:
        df_households[veh_col] = df_households[veh_col].replace(" ", "0").astype(int)

    df_persons["POND_PERS"] = df_persons["POND_PERS"].str.replace(",", ".").astype(float)

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

    spatial1 = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGTFVG2016_ZF.TAB")
    spatial1 = spatial1.set_crs(epsg = 2154, allow_override=True)
    spatial1.columns = spatial1.columns.str.lower()
    spatial1 = spatial1[["geometry", "zonefine", "depcom"]]
    spatial1.columns = ["geometry", "zone_id", "departement"]
    spatial1["departement_id"] = spatial1["departement"].astype(str).str[:2]
    spatial1 = spatial1[["geometry", "zone_id", "departement_id"]]

    spatial2 = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGTFVG2016_ZonesExternes.TAB")
    spatial2 = spatial2.set_crs(epsg = 2154, allow_override=True)
    spatial2.columns = spatial2.columns.str.lower()
    spatial2 = spatial2[["geometry", "num_zf", "nom_d10"]]
    spatial2.columns = ["geometry", "zone_id", "departement_id"]

    df_spatial = gpd.GeoDataFrame(pd.concat([spatial1, spatial2]), crs=spatial1.crs)

    return df_households, df_persons, df_trips, df_spatial


FILES = [
    "Menages_EMD.csv",
    "Personnes_EMD.csv",
    "deplacements_emd_full_OCT.csv",
    "Doc/SIG/EDGTFVG2016_ZF.TAB",
    "Doc/SIG/EDGTFVG2016_ZF.ID",
    "Doc/SIG/EDGTFVG2016_ZF.IND",
    "Doc/SIG/EDGTFVG2016_ZF.MAP",
    "Doc/SIG/EDGTFVG2016_ZonesExternes.TAB",
    "Doc/SIG/EDGTFVG2016_ZonesExternes.ID",
    "Doc/SIG/EDGTFVG2016_ZonesExternes.IND",
    "Doc/SIG/EDGTFVG2016_ZonesExternes.MAP",
]


def validate(context):
    data_path   = context.config("data_path")
    edgt_path   = f"{data_path}/edgt_2017_annemasse/TPG"
    
    for name in FILES:
        current_path = f"{edgt_path}/{name}"
        if not os.path.exists(current_path):
            raise RuntimeError("File missing from EDGT: %s" % name)

    return [
        os.path.getsize(f"{edgt_path}/{name}")
        for name in FILES
    ]
