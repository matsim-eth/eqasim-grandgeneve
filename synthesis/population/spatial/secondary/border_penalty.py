import geopandas as gpd

"""
Calibration of the border penalty of the secondary location assignment
(see BorderPenaltyAssignmentSolver in synthesis.population.spatial.secondary.locations):
a chain whose fixed activities are in France may place a secondary activity in
Switzerland with probability HOME_ACCEPTANCE (EDGT D30 sector of the home)
times DESTINATION_WEIGHTS (EDGT DTIR of the Swiss location), capped at 1.

The values were calibrated against the EDGT74 with
analysis.cross_border_commuters.secondary_activities, on a 10% sample with
the force model chain solver: share of persons with a secondary but no
mandatory activity in Switzerland by home D30 sector, and distribution of
their Swiss activities by DTIR. Editing them changes this stage's code,
which makes synpp rerun it and the secondary locations.
"""

# Share of chains allowed to cross the border, by EDGT D30 sector of the
# home. Sectors not listed (far from the border, too few cross-border
# respondents for a meaningful target) get 1.0.
HOME_ACCEPTANCE = {
    "Annemasse": 0.13,
    "Zone urbaine Annemasse": 0.09,
    "Thonon": 0.10,
    "Pays d'Evian": 0.05,
    "CC du Genevois": 0.20,
    "Vallée Verte Quatre Rivières": 0.30,
    "Pays Bellegardien": 0.48,
    "Arve et Salève Pays Rochois": 0.26,
    "Bas Chablais Collines du Léman": 0.35,
    "Zone périurbaine Pays de Gex": 0.37,
    "Zone périurbaine Annemasse": 0.41,
    "Zone urbaine Saint-Genis": 0.52,
    "Zone urbaine Gex Ferney": 0.69,
    "Semine Usses Cruseilles Fillières": 0.78,
}

# Factor on that share by EDGT DTIR of the Swiss destination, relative to
# the north of the canton of Geneva: EDGT over synthetic share of the
# destinations without weights. Valais and Reste Suisse have no secondary
# locations (only the cantons of Geneva and Vaud are loaded).
DESTINATION_WEIGHTS = {
    "Genève": 2.81,
    "Nord canton de Genève (+ Terres Saintes VD)": 1.0,
    "Ouest canton de Genève": 0.65,
    "Est canton de Genève": 0.66,
    "District de Nyon (hors Terres Saintes VD)": 0.83,
    # 1.72 when calibrated, but driven by the cross-lake flows from the
    # Chablais, now blocked by secondary_activities.lake_barrier
    "Agglomeration de Lausanne-Morges": 1.0,
    "Reste canton de Vaud": 0.06,
    "Canton du Valais": 1.0,
    "Reste Suisse": 1.0,
}

def configure(context):
    context.stage("data.hts.edgt_74.adisp_merge.zones")
    context.stage("data.locations_CH.secondary")

def check_names(names, known, label):
    unknown = set(names) - set(known)
    if len(unknown) > 0:
        raise RuntimeError("Unknown %s in synthesis.population.spatial.secondary.border_penalty: %s" % (label, sorted(unknown)))

def execute(context):
    df_zones, _, _ = context.stage("data.hts.edgt_74.adisp_merge.zones")
    df_zones = df_zones.copy()
    df_zones["geometry"] = df_zones.geometry.make_valid()

    # Home sectors: internal zones of both surveys, by D30
    df_sectors = df_zones[df_zones["source"].isin(["annemasse_zf", "annecy_zf"])]
    check_names(HOME_ACCEPTANCE, df_sectors["d30_name"], "D30 sectors")

    df_sectors = df_sectors.dissolve(by = "d30_name", as_index = False)[["d30_name", "geometry"]]
    df_sectors["border_acceptance"] = df_sectors["d30_name"].map(HOME_ACCEPTANCE).fillna(1.0)

    # Swiss destinations: nearest Swiss DTIR, as in
    # analysis.cross_border_commuters.secondary_activities
    df_swiss_zones = df_zones[df_zones["d10_name"] == "Suisse"][["dtir_name", "geometry"]]
    check_names(DESTINATION_WEIGHTS, df_swiss_zones["dtir_name"], "Swiss DTIR")

    # data.locations_CH.secondary is a plain DataFrame in Lambert-93, the
    # CRS of the zones
    df_swiss = context.stage("data.locations_CH.secondary")
    df_swiss = gpd.GeoDataFrame(df_swiss[["location_id", "geometry"]], crs = df_zones.crs)
    df_swiss = gpd.sjoin_nearest(df_swiss, df_swiss_zones, how = "left").drop_duplicates("location_id")
    df_swiss["weight"] = df_swiss["dtir_name"].map(DESTINATION_WEIGHTS).fillna(1.0)

    print("Border penalty: Swiss secondary locations by DTIR and weight")
    print(df_swiss.groupby(["dtir_name", "weight"]).size())

    return df_sectors, df_swiss[["location_id", "dtir_name", "weight"]]
