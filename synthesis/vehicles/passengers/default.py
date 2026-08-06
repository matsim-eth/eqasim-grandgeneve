import pandas as pd

"""
Creates a vehicle fleet based on a default vehicle type for the dummy passenger mode
"""

def configure(context):
    context.stage("synthesis.population.enriched")

def execute(context):
    df_persons = context.stage("synthesis.population.enriched")

    df_vehicle_types = pd.DataFrame.from_records([{
        "type_id": "default_car_passenger", "nb_seats": 4, "length": 5.0, "width": 1.0, "pce": 1.0, "mode": "car_passenger",
        "hbefa_cat": "pass. car", "hbefa_tech": "average", "hbefa_size": "average", "hbefa_emission": "average",
        'cnossos_cat': "1"
    }])

    df_vehicles = df_persons[["person_id"]].copy()
    df_vehicles = df_vehicles.rename(columns = { "person_id": "owner_id" })

    df_vehicles["vehicle_id"] = df_vehicles["owner_id"].astype(str) + ":car_passenger"
    df_vehicles["type_id"] = "default_car_passenger"
    df_vehicles["critair"] = "Crit'air 1"
    df_vehicles["technology"] = "Gazole"
    df_vehicles["age"] = 0
    df_vehicles["euro"] = 6

    # "car_passenger_loop" legs (very short "went around the block" trips,
    # see data.hts.edgt_74.adisp_merge.merge.tag_short_trip_loop_mode) need
    # their own PersonVehicles entry (MATSim keys the map by mode), of the
    # same default vehicle type as regular "car_passenger" legs.
    df_vehicles_loop = df_vehicles.copy()
    df_vehicles_loop["mode"] = "car_passenger_loop"
    df_vehicles_loop["vehicle_id"] = df_vehicles_loop["owner_id"].astype(str) + ":car_passenger_loop"

    df_vehicles["mode"] = "car_passenger"
    df_vehicles = pd.concat([df_vehicles, df_vehicles_loop])

    return df_vehicle_types, df_vehicles