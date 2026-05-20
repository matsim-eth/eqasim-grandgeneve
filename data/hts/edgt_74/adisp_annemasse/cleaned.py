import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the Annemasse EDGT (Adisp version).
"""

def configure(context):
    context.stage("data.hts.edgt_74.adisp_annemasse.raw")

    context.config("specific_day_scenario", "workday")


PURPOSE_MAP = {
    "home": [1, 2],
    "work": [11, 12, 13, 14, 81],
    "education": [21, 22, 23, 24, 25, 26, 27, 28, 29, 96, 97],
    "shop": [30, 31, 32, 33, 34, 35, 82, 98],
    "leisure": [51, 52, 53, 54],
    "other": [41, 42, 43, 61, 62, 63, 64, 71, 72, 73, 74, 91]
}

MODES_MAP = {
    "car": [10, 13, 15, 21, 81], # 10 is (driving) an ambulance
    "car_passenger": [14, 16, 22, 82],
    "pt": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 51, 52, 53, 61, 71, 91, 92, 94, 95],
    "bike": [11, 17, 12, 18, 93],
    "walk": [1, 2] # Actually, 2 is not really explained, but we assume it is walk
}

WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday"}
WEEKEND  = {"saturday", "sunday"}
ALL_DAYS = WEEKDAYS | WEEKEND

DAY_TO_INT = {
    "monday": "1", "tuesday": "2", "wednesday": "3",
    "thursday": "4", "friday": "5"
}


def resolve_days(day):
    if isinstance(day, str):
        if day == "workday":
            return sorted(WEEKDAYS)
        elif day == "weekend":
            return sorted(WEEKEND)
        elif day in ALL_DAYS:
            return [day]
        else:
            raise RuntimeError(f"Invalid day value: {day}. Expected a day name, workday, or weekend.")
        
    elif isinstance(day, (list, tuple)):
        day     = list(day)
        invalid = set(day) - ALL_DAYS
        if invalid:
            raise RuntimeError(f"Invalid day value(s): {invalid}. Expected day names from {ALL_DAYS}.")
        return day
    
    else:
        raise RuntimeError(f"Invalid type for day parameter: {type(day)}. Expected str or list.")
    

def filter_by_days(df, days):
    day_ints = [DAY_TO_INT[d] for d in days]
    return df[df["JOUR"].isin(day_ints)]


def execute(context):
    df_households, df_persons, df_trips, df_spatial = context.stage("data.hts.edgt_74.adisp_annemasse.raw")

    # Merge departement into households
    df_spatial        = df_spatial[["zone_id", "departement_id"]].copy()
    df_spatial["ZFM"] = df_spatial["zone_id"].astype(str).str.pad(width = 8, side = "left", fillchar = "0")
    df_spatial        = df_spatial[["ZFM", "departement_id"]]

    # Attention, some households get lost here!
    df_households                   = pd.merge(df_households, df_spatial, on = "ZFM", how = "left")
    df_households["departement_id"] = df_households["departement_id"].fillna("unknown")

    # Transform original IDs to integer (they are hierarchichal)
    df_households["edgt_household_id"] = (df_households["ZFM"] + df_households["ECH"].str.pad(width = 3, side = "left", fillchar = "0")).astype(int)
    df_persons["edgt_person_id"]       = df_persons["PER"].astype(int)
    df_persons["edgt_household_id"]    = (df_persons["ZFP"] + df_persons["ECH"].str.pad(width = 3, side = "left", fillchar = "0")).astype(int)
    df_trips["edgt_person_id"]         = df_trips["PER"].astype(int)
    df_trips["edgt_household_id"]      = (df_trips["ZFD"] + df_trips["ECH"].str.pad(width = 3, side = "left", fillchar = "0")).astype(int)
    df_trips["edgt_trip_id"]           = df_trips["NDEP"].astype(int)

    # Construct new IDs for households, persons and trips (which are unique globally)
    #df_households["household_id"] = np.arange(len(df_households))
    df_households["household_id"] = df_households["edgt_household_id"]

    # Select only records from the selected scenario day(s)
    days = resolve_days(context.config("specific_day_scenario"))
    if any(d in WEEKEND for d in days):
        raise RuntimeError(
            f"Invalid day for EDGT Annemasse: {days}. "
            f"The Annemasse EDGT survey does not cover weekends. "
            f"Please select ENTD as HTS or choose another day."
        )    
    
    initial_length = len(df_persons)
    df_persons     = filter_by_days(df_persons, days)
    final_length   = len(df_persons)
    persdiff       = initial_length - final_length
    persreldiff    = persdiff / initial_length * 100

    hhl_length1   = len(df_households)
    df_households = df_households[df_households["edgt_household_id"].isin(df_persons["edgt_household_id"].values.tolist())]
    hhl_length2   = len(df_households)
    hhldiff       = hhl_length1 - hhl_length2
    hhlreldiff    = hhldiff / hhl_length1 * 100

    trips_length1 = len(df_trips)
    df_trips      = df_trips[df_trips["edgt_person_id"].isin(df_trips["edgt_person_id"].values.tolist())]
    trips_length2 = len(df_trips)
    tripsdiff     = trips_length1 - trips_length2
    tripsreldiff  = tripsdiff / trips_length1 * 100

    print(f"Filtering data based on selected days. Removed {hhldiff} ({round(hhlreldiff, 2)}%) households, {persdiff} ({round(persreldiff, 2)}%) persons, and {tripsdiff} ({round(tripsreldiff, 2)}%) trips.")

    df_persons = pd.merge(
        df_persons, df_households[["edgt_household_id", "household_id", "departement_id"]],
        on = ["edgt_household_id"]
    ).sort_values(by = ["household_id", "edgt_person_id"])
    df_persons["person_id"] = np.arange(len(df_persons))

    df_trips = pd.merge(
        df_trips, df_persons[["edgt_person_id", "edgt_household_id", "person_id", "household_id"]],
        on = ["edgt_person_id", "edgt_household_id"]
    ).sort_values(by = ["household_id", "person_id", "edgt_trip_id"])
    df_trips["trip_id"] = np.arange(len(df_trips))

    # Trip flags
    df_trips = hts.compute_first_last(df_trips)

    # Weight
    df_persons["person_weight"]       = df_persons["COEP"].astype(float)
    df_households["household_weight"] = df_households["COE0"].astype(float)

    # Clean age
    df_persons["age"] = df_persons["P4"].astype(int)

    # Clean sex
    df_persons.loc[df_persons["P2"] == 1, "sex"] = "male"
    df_persons.loc[df_persons["P2"] == 2, "sex"] = "female"
    df_persons["sex"] = df_persons["sex"].astype("category")

    # Household size
    df_size       = df_persons.groupby("household_id").size().reset_index(name = "household_size")
    df_households = pd.merge(df_households, df_size, on = "household_id")

    # Clean departement
    df_trips = pd.merge(df_trips, df_spatial.rename(columns = {
        "ZFM": "D3", "departement_id": "origin_departement_id"
    }), on = "D3", how = "left")

    df_trips = pd.merge(df_trips, df_spatial.rename(columns = {
        "ZFM": "D7", "departement_id": "destination_departement_id"
    }), on = "D7", how = "left")

    df_trips["origin_departement_id"]      = df_trips["origin_departement_id"].fillna("unknown")
    df_trips["destination_departement_id"] = df_trips["destination_departement_id"].fillna("unknown")

    df_households["departement_id"]        = df_households["departement_id"].astype("category")
    df_persons["departement_id"]           = df_persons["departement_id"].astype("category")
    df_trips["origin_departement_id"]      = df_trips["origin_departement_id"].astype("category")
    df_trips["destination_departement_id"] = df_trips["destination_departement_id"].astype("category")

    # Clean employment
    df_persons["employed"] = df_persons["P9"].isin(["1", "2"])

    # Studies
    df_persons["studies"] = df_persons["P9"].isin(["3", "4", "5"])

    # Number of vehicles
    df_households["number_of_vehicles"] = df_households["M6"] + df_households["M14"]
    df_households["number_of_vehicles"] = df_households["number_of_vehicles"].astype(int)
    df_households["number_of_bikes"]    = df_households["M21"].astype(int)

    # License
    df_persons["has_license"] = df_persons["P7"] == "1"

    # Has subscription
    df_persons["has_pt_subscription"] = df_persons["P12"].isin(["1", "2", "3", "5", "6"])

    # Survey respondents 
    # PENQ 1 : fully awnsered the travel questionary section, having a chain or non-movers
    # PENQ 2 : nonrespondent of travel questionary section
    df_persons["PENQ"] = df_persons["PENQ"].fillna("2").astype("int")
    df_persons.loc[df_persons["PENQ"] == 1, "travel_respondent"] = True
    df_persons.loc[df_persons["PENQ"] == 2, "travel_respondent"] = False

    assert np.count_nonzero(df_persons["travel_respondent"].isna()) == 0

    # Trip purpose
    df_trips["following_purpose"] = "invalid"
    df_trips["preceding_purpose"] = "invalid"

    for purpose, values in PURPOSE_MAP.items():
        df_trips.loc[df_trips["D5A"].isin(values), "following_purpose"] = purpose
        df_trips.loc[df_trips["D2A"].isin(values), "preceding_purpose"] = purpose

    assert np.count_nonzero(df_trips["following_purpose"] == "invalid") == 0
    assert np.count_nonzero(df_trips["preceding_purpose"] == "invalid") == 0

    df_trips["following_purpose"] = df_trips["following_purpose"].astype("category")
    df_trips["preceding_purpose"] = df_trips["preceding_purpose"].astype("category")

    # Trip mode
    for mode, values in MODES_MAP.items():
        df_trips.loc[df_trips["MODP"].isin(values), "mode"] = mode

    assert np.count_nonzero(df_trips["following_purpose"] == "invalid") == 0
    df_trips["mode"] = df_trips["mode"].astype("category")

    # Further trip attributes
    df_trips["euclidean_distance"] = df_trips["D11"]
    df_trips["routed_distance"]    = df_trips["D12"]

    # Trip times
    df_trips["departure_time"] = 3600.0 * (df_trips["D4"] // 100) # hour
    df_trips["departure_time"] += 60.0  * (df_trips["D4"] % 100) # minute

    df_trips["arrival_time"] = 3600.0 * (df_trips["D8"] // 100) # hour
    df_trips["arrival_time"] += 60.0  * (df_trips["D8"] % 100) # minute

    df_trips = df_trips.sort_values(by = ["household_id", "person_id", "trip_id"])
    df_trips = hts.fix_trip_times(df_trips)

    # Durations
    df_trips["trip_duration"] = df_trips["arrival_time"] - df_trips["departure_time"]
    hts.compute_activity_duration(df_trips)

    # Add weight to trips
    df_trips = pd.merge(
        df_trips, df_persons[["person_id", "COE1"]], on = "person_id", how = "left"
    ).rename(columns = { "COE1": "trip_weight" })
    df_persons["trip_weight"] = df_persons["COE1"]

    # Chain length
    df_count = df_trips[["person_id"]].groupby("person_id").size().reset_index(name = "number_of_trips")

    # People with at least one trip (number_of_trips > 0)
    df_persons = pd.merge(df_persons, df_count, on = "person_id", how = "left")

    # People that answered the travel questionary section but stayed at home (number_of_trips = 0)
    df_persons.loc[(df_persons["travel_respondent"] == True) & (df_persons["number_of_trips"].isna()), "number_of_trips"] = 0
    
    # Nonrespondent of travel questionary section (number_of_trips = -1)
    df_persons["number_of_trips"] = df_persons["number_of_trips"].fillna(-1).astype(int)

    # Passenger attribute
    df_persons["is_passenger"] = df_persons["person_id"].isin(
        df_trips[df_trips["mode"] == "car_passenger"]["person_id"].unique()
    )

    # Calculate consumption units
    hts.check_household_size(df_households, df_persons)
    df_households = pd.merge(df_households, hts.calculate_consumption_units(df_persons), on = "household_id")

    # Socioprofessional class
    df_persons["socioprofessional_class"] = df_persons["PCSC"].fillna(8).astype(int)

    # Check departure and arrival times
    assert np.count_nonzero(df_trips["departure_time"].isna()) == 0
    assert np.count_nonzero(df_trips["arrival_time"].isna()) == 0

    # Fix activity types (because of inconsistent EGT data and removing in the timing fixing step)
    hts.fix_activity_types(df_trips)

    return df_households, df_persons, df_trips