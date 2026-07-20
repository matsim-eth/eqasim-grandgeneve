import os

import numpy as np
import pandas as pd
import geopandas as gpd

"""
This stage builds a unified zoning system for the Annemasse and Annecy EDGT
surveys (edgt_74), and matches each survey's generator points (specific
venues some trips reference instead of a full zone: stations, schools,
parkings, ...) to it.
"""


# In EDGTFVG2016_ZonesExternes.TAB, these NOM_DTIR labels cover the "rest of
# Haute-Savoie" at one zone per commune. The Annecy survey zones the same area
# at a finer resolution (up to 23 zones per commune), so they are dropped here
# in favour of read_annecy_zones().

HAUTE_SAVOIE_EXTERNAL_LABELS = [
    "Reste périmètre EDGT 74",
    "Chablais Giffre",
    "Mont-Blanc",
    "Agglomération d'ANNECY",
]


def configure(context):
    context.config("data_path")


def get_annemasse_path(context):
    return "{}/edgt_2017/edgt_2017_annemasse/adisp/lil-1212/lil-1212.csv".format(context.config("data_path"))


def get_annecy_path(context):
    return "{}/edgt_2017/edgt_2017_annecy/adisp/lil-1315/lil-1315.csv".format(context.config("data_path"))


def read_annemasse_zones(context):
    edgt_path = get_annemasse_path(context)

    zones = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGTFVG2016_ZF.TAB")
    zones = zones.set_crs(epsg = 2154, allow_override = True)
    zones["source"] = "annemasse_zf"

    return zones


def read_annemasse_external_zones(context):
    """Annemasse survey's external zones, excluding the Haute-Savoie zones replaced by the Annecy survey."""
    edgt_path = get_annemasse_path(context)

    zones = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGTFVG2016_ZonesExternes.TAB")
    zones = zones.set_crs(epsg = 2154, allow_override = True)
    zones["geometry"] = zones.geometry.make_valid()

    # A few communes (e.g. Greifensee, Brugg) are split across two rows with
    # identical attributes; merge these multi-part geometries back into one
    # row per commune so zone ids stay unique. Grouping on NOM_DTIR too keeps
    # the handful of no-commune, region-level catch-all rows separate.
    zones = zones.dissolve(by = ["Id_com", "NOM_DTIR"], as_index = False, aggfunc = "first")

    zones = zones[~zones["NOM_DTIR"].isin(HAUTE_SAVOIE_EXTERNAL_LABELS)].copy()
    zones["source"] = "annemasse_external"

    return zones


def read_annemasse_haute_savoie_replaced_ids(context):
    """Id_com -> Num_ZF for the Haute-Savoie external zones dropped from
    read_annemasse_external_zones(), i.e. the id each of those communes would
    have had in the Annemasse survey before being replaced by the Annecy
    survey's finer zoning."""
    edgt_path = get_annemasse_path(context)

    zones = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGTFVG2016_ZonesExternes.TAB")
    zones["geometry"] = zones.geometry.make_valid()
    zones = zones.dissolve(by = ["Id_com", "NOM_DTIR"], as_index = False, aggfunc = "first")

    zones = zones[zones["NOM_DTIR"].isin(HAUTE_SAVOIE_EXTERNAL_LABELS)]

    return zones.set_index("Id_com")["Num_ZF"]


def read_annecy_zones(context):
    """Annecy survey's own zoning, also covering the 'rest of Haute-Savoie' for the Annemasse survey."""
    edgt_path = get_annecy_path(context)

    zones = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGT_Reste74_2017_ZF_404009.TAB")
    zones = zones.set_crs(epsg = 2154, allow_override = True)
    zones["source"] = "annecy_zf"

    return zones


def _make_zone_id(gdf, prefix, id_column, fallback_columns = ()):
    def make_id(row):
        raw_id = str(row[id_column]).strip()

        for fallback_column in fallback_columns:
            if raw_id:
                break
            raw_id = str(row[fallback_column]).strip()

        return f"{prefix}_{raw_id}"

    gdf = gdf.copy()
    gdf["zone_id"] = gdf.apply(make_id, axis = 1)

    return gdf


# One shared column name per concept, mapped from each source's own naming.
# Sources that have no equivalent for a concept (e.g. no iris or zone-level
# code for the commune-level external zones) simply end up with NaN there
# after concatenation. Everything not listed is either mostly empty, a
# duplicate of another column here (verified 1:1 against the data), or
# unrelated to zone identity (e.g. face-to-face/telephone collection flags).
RENAME_COLUMNS = {
    "annemasse_zf": {
        "DepCom": "commune_id", "Nom_Com": "commune_name",
        "DComIris": "iris_id", "Nom_Iris": "iris_name",
        "DTIR": "dtir_id", "Nom_DTIR": "dtir_name",
        "ZoneFine": "zone_code", "Nom_ZoneFine": "zone_name",
        "NUM_D30": "d30_id", "NOM_D30": "d30_name",
        "NUM_D10": "d10_id", "NOM_D10": "d10_name",
    },
    "annemasse_external": {
        "Id_com": "commune_id", "Nom_com": "commune_name",
        "NUM_DTIR": "dtir_id", "NOM_DTIR": "dtir_name",
        "NUM_D30": "d30_id", "NOM_D30": "d30_name",
        "NUM_D10": "d10_id", "NOM_D10": "d10_name",
        "Num_ZF": "zone_code",
    },
    "annecy_zf": {
        "Insee": "commune_id", "Commune": "commune_name",
        "Dcomiris": "iris_id", "Nom_iris": "iris_name",
        "Dtir": "dtir_id", "Nom_dtir_2": "dtir_name",
        "Zf": "zone_code", "Nom_zf": "zone_name",
        "D30": "d30_id", "D30_nom": "d30_name",
        "D10": "d10_id", "D10_nom": "d10_name",
    },
}


def _keep_and_rename_columns(gdf, source):
    mapping = RENAME_COLUMNS[source]
    return gdf[list(mapping.keys()) + ["geometry", "source", "zone_id"]].rename(columns = mapping)


# The shared regional "Code Externe" scheme's foreign-country codes (see
# ZonageExterne / Zonage_externe in the survey documentation) — no polygon
# exists for these anywhere in the source data, so they get a virtual,
# geometry-less zone purely so trips to/from abroad show a readable name
# instead of an unmatched NaN. 999000 and 999100 both mean "Reste du monde".
FOREIGN_COUNTRY_CODES = {
    "999010": "ALLEMAGNE",
    "999020": "ANDORRE",
    "999030": "BELGIQUE",
    "999040": "ESPAGNE",
    "999050": "ITALIE",
    "999060": "LUXEMBOURG",
    "999070": "MONACO",
    "999080": "ROYAUME-UNI",
    "999100": "RESTE DU MONDE",
    "999000": "RESTE DU MONDE",
}


def build_foreign_zones(crs):
    """Virtual, geometry-less zones for the foreign-country codes."""
    rows = [
        {
            "zone_id": f"FOREIGN_{code}",
            "zone_code": code,
            "zone_name": name,
            "commune_name": name,
            "source": "foreign",
            "geometry": None,
        }
        for code, name in FOREIGN_COUNTRY_CODES.items()
    ]

    return gpd.GeoDataFrame(rows, crs = crs)


def build_unified_zoning(context):
    """Combine both surveys' zones into a single zoning system.

    Every source is mapped onto the same set of attribute columns (commune_id,
    commune_name, iris_id, iris_name, dtir_id, dtir_name, zone_code, zone_name,
    d30_id, d30_name, d10_id, d10_name); a source has NaN for a column when it
    has no equivalent concept. A new "zone_id" column is added that is unique
    across the combined system. annemasse_ext_id/annecy_ext_id cross-reference
    each survey's zones to the other survey's shared-convention commune code.
    """
    annemasse_zf       = read_annemasse_zones(context)
    annemasse_external = read_annemasse_external_zones(context)
    annecy_zf          = read_annecy_zones(context)

    annemasse_zf       = _make_zone_id(annemasse_zf, "AM", "ZoneFine")
    annemasse_external = _make_zone_id(annemasse_external, "AME", "Num_ZF", fallback_columns = ("Id_com", "NOM_DTIR"))
    annecy_zf          = _make_zone_id(annecy_zf, "AC", "Zf")

    annemasse_zf       = _keep_and_rename_columns(annemasse_zf, "annemasse_zf")
    annemasse_external = _keep_and_rename_columns(annemasse_external, "annemasse_external")
    annecy_zf          = _keep_and_rename_columns(annecy_zf, "annecy_zf")

    # Traceability for the zones that replaced Annemasse's coarser "rest of
    # Haute-Savoie" external zones: the id that commune would have had in the
    # Annemasse survey. NaN for every other zone.
    replaced_ids = read_annemasse_haute_savoie_replaced_ids(context)
    annecy_zf["annemasse_ext_id"] = annecy_zf["commune_id"].map(replaced_ids)

    # Mirror in the other direction: the shared regional "Code Externe"
    # convention ("8" + commune INSEE code) that the Annecy survey uses to
    # refer to a commune inside Annemasse's own perimeter — confirmed to
    # match annemasse_ext_id's own values exactly (0 exceptions checked
    # against real data), so it's derived directly rather than needing a
    # separate lookup file. NaN for every other zone.
    annemasse_zf["annecy_ext_id"] = "8" + annemasse_zf["commune_id"].astype(str)

    foreign_zones = build_foreign_zones(annemasse_zf.crs)

    zones = gpd.GeoDataFrame(
        pd.concat([annemasse_zf, annemasse_external, annecy_zf, foreign_zones], ignore_index = True),
        crs = annemasse_zf.crs
    )

    assert zones["zone_id"].is_unique

    return zones


def read_annemasse_points(context):
    """Annemasse survey's generator points (specific venues some trips
    reference instead of a full zone: stations, schools, parkings, ...)."""
    edgt_path = get_annemasse_path(context)

    points = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGTFVG2016_GT.TAB")
    points = points.set_crs(epsg = 2154, allow_override = True)
    points["source"] = "annemasse_gt"

    return points


def read_annecy_points(context):
    """Annecy survey's generator points."""
    edgt_path = get_annecy_path(context)

    points = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGT_Reste74_2017_GT.TAB")
    points = points.set_crs(epsg = 2154, allow_override = True)
    points["source"] = "annecy_gt"

    return points


# Both surveys' generator points use their own zone-code scheme (point_code,
# e.g. Annemasse's "107051") and their own field names for the same concepts.
POINT_RENAME_COLUMNS = {
    "annemasse_gt": {
        "NUM_ZF": "point_code", "NOM": "point_name", "CATEGORIE": "category",
        "INSEE": "commune_id", "COMMUNE": "commune_name",
    },
    "annecy_gt": {
        "Zf": "point_code", "Nom": "point_name", "Typegd": "category",
        "Insee": "commune_id", "Commune": "commune_name",
    },
}


def build_generator_points(context, zones):
    """Both surveys' generator points, matched to the zone polygon they
    physically fall in via a spatial join against the unified zoning (rather
    than their own point_code, which isn't part of any zone's zone_code)."""
    annemasse_points = read_annemasse_points(context)
    annecy_points     = read_annecy_points(context)

    annemasse_points = annemasse_points[list(POINT_RENAME_COLUMNS["annemasse_gt"].keys()) + ["geometry", "source"]]
    annemasse_points = annemasse_points.rename(columns = POINT_RENAME_COLUMNS["annemasse_gt"])

    annecy_points = annecy_points[list(POINT_RENAME_COLUMNS["annecy_gt"].keys()) + ["geometry", "source"]]
    annecy_points = annecy_points.rename(columns = POINT_RENAME_COLUMNS["annecy_gt"])

    points = gpd.GeoDataFrame(pd.concat([annemasse_points, annecy_points], ignore_index = True), crs = annemasse_points.crs)

    matched = gpd.sjoin(points, zones[["zone_id", "geometry"]], how = "left", predicate = "intersects")
    matched = matched.drop(columns = "index_right").drop_duplicates(subset = ["source", "point_code"])

    # A handful of points can fall just outside every polygon (edge-snapping
    # precision issues); fall back to the nearest zone for those.
    still_missing = matched["zone_id"].isna()
    if still_missing.any():
        nearest = gpd.sjoin_nearest(points[still_missing.values], zones[["zone_id", "geometry"]], how = "left")
        nearest = nearest[~nearest.index.duplicated()]
        matched.loc[still_missing, "zone_id"] = nearest["zone_id"].reindex(matched.index[still_missing]).values

    return matched


def build_point_to_zone_id(points):
    """point_code -> zone_id, from the generator points spatially matched to
    the unified zoning in build_generator_points()."""
    points = points.dropna(subset = ["zone_id"]).drop_duplicates("point_code")
    return points.set_index("point_code")["zone_id"]


def read_external_code_to_dtir(context):
    """Code Externe -> DTIR, from the region's shared "hors périmètre" zone
    numbering convention (documented separately by each survey, but the codes
    and DTIR groupings overlap)."""
    annemasse_lookup = pd.read_excel(
        f"{get_annemasse_path(context)}/Doc/SIG/RegroupementdesZonesExploitationStandard.xlsx",
        sheet_name = "ZonageExterne"
    )[["Code Externe", "DTIR"]]

    annecy_lookup = pd.read_excel(
        f"{get_annecy_path(context)}/Doc/SIG/EDGT_Reste74_2017_DTIR_D30_D10_D2.xls",
        sheet_name = "Zonage_externe"
    )[["Code Externe", "DTIR"]]

    # This sheet encodes DTIR as a descriptive string for every row, e.g.
    # '801001 = "Ext-DTIR-06-Communes des EPCI AU Lyon..."'; extract just the
    # code and normalize it to the "Ext_DTIR-XX" form used as dtir_id.
    annecy_lookup["DTIR"] = (
        annecy_lookup["DTIR"].astype(str)
        .str.extract(r"(Ext-DTIR-\d+)")[0]
        .str.replace("-", "_", n = 1)
    )

    lookup = pd.concat([annemasse_lookup, annecy_lookup]).drop_duplicates("Code Externe")
    lookup["Code Externe"] = lookup["Code Externe"].astype(str).str.strip()

    return lookup.set_index("Code Externe")["DTIR"].astype(str)


# Pre-merger commune codes still referenced by old trip records, mapped to
# the surviving commune's code. Metz-Tessy (74181) merged into Épagny to
# form "Épagny Metz-Tessy" (74112) between the two survey vintages, so only
# 74112 exists in the Annecy zoning.
COMMUNE_CODE_CORRECTIONS = {
    "874181": "874112",
}

# Codes with no zone anywhere in the system for their DTIR ("Bas Chablais
# Thonon Evian" has no polygon of its own), mapped to the commune_id(s) whose
# zones are a reasonable stand-in.
MANUAL_CODE_COMMUNES = {
    "874106": ["74281", "74119"],  # Thonon-les-Bains, Évian-les-Bains
}


def build_manual_code_lookup(zones):
    lookups = []

    for code, commune_ids in MANUAL_CODE_COMMUNES.items():
        zone_ids = zones.loc[zones["commune_id"].astype(str).isin(commune_ids), "zone_id"].tolist()
        lookups.append(pd.Series([zone_ids], index = [code]))

    return pd.concat(lookups)


def build_zone_code_lookups(zones):
    """Three lookups used to resolve a raw trip zone code (D3/D7) to our
    unified zone_id, from the least to the most approximate:

    1. code_to_zone_id: exact match on zone_code (own ZoneFine/Zf zones, and
       Num_ZF for external communes that have their own polygon).
    2. dtir_to_zone_id: DTIR-level fallback, only for DTIRs that collapse to
       a single zone (far-away catch-all regions like "Reste France") — kept
       NaN for DTIRs resolved at commune level, since those are already
       covered by (1) and picking one of many zones there would be wrong.
    3. legacy_to_zone_ids: cross-survey commune codes, in both directions —
       legacy Annemasse-survey codes now covered by the finer Annecy zoning
       (annemasse_ext_id, on annecy_zf zones) and the Annecy survey's own
       "Code Externe" references into Annemasse's own perimeter
       (annecy_ext_id, on annemasse_zf zones). Every zone_id sharing that
       code, since the commune-level code alone can't tell us which
       sub-zone; the two directions can't collide since they key off
       disjoint sets of communes.

    A 4th tier, point_to_zone_id (generator points spatially matched to a
    zone), is built separately by build_point_to_zone_id().
    """
    zone_code = zones["zone_code"].astype(str).str.strip()
    has_code = zone_code.notna() & (zone_code != "") & (zone_code != "nan")
    code_to_zone_id = pd.Series(zones.loc[has_code, "zone_id"].values, index = zone_code[has_code].values)
    code_to_zone_id = code_to_zone_id[~code_to_zone_id.index.duplicated()]

    dtir_zone_counts = zones.groupby("dtir_id")["zone_id"].nunique()
    single_zone_dtirs = dtir_zone_counts[dtir_zone_counts == 1].index
    dtir_to_zone_id = zones[zones["dtir_id"].isin(single_zone_dtirs)].groupby("dtir_id")["zone_id"].first()
    dtir_to_zone_ids = zones.groupby("dtir_id")["zone_id"].apply(list)

    annecy_zf = zones[zones["source"] == "annecy_zf"]
    annemasse_zf = zones[zones["source"] == "annemasse_zf"]
    legacy_to_zone_ids = pd.concat([
        annecy_zf.groupby("annemasse_ext_id")["zone_id"].apply(list),
        annemasse_zf.groupby("annecy_ext_id")["zone_id"].apply(list),
        build_manual_code_lookup(zones),
    ])

    return code_to_zone_id, dtir_to_zone_id, dtir_to_zone_ids, legacy_to_zone_ids


def resolve_zone_id(codes, code_to_zone_id, code_to_dtir, dtir_to_zone_id, dtir_to_zone_ids, legacy_to_zone_ids, point_to_zone_id, rng):
    """Resolve a Series of raw trip zone codes to unified zone_id, tier by
    tier (see build_zone_code_lookups). Unresolved codes are left as NaN."""
    codes = codes.astype(str).str.strip()

    resolved = codes.map(code_to_zone_id)

    still_missing = resolved.isna()
    resolved[still_missing] = codes[still_missing].map(code_to_dtir).map(dtir_to_zone_id)

    still_missing = resolved.isna()
    corrected_codes = codes[still_missing].replace(COMMUNE_CODE_CORRECTIONS)
    candidates = corrected_codes.map(legacy_to_zone_ids)
    resolved[still_missing] = candidates.map(lambda c: rng.choice(c) if isinstance(c, list) else np.nan)

    still_missing = resolved.isna()
    resolved[still_missing] = codes[still_missing].map(point_to_zone_id)

    # Last resort: codes whose DTIR genuinely spans several zones (e.g. a
    # point like Geneva airport with no zone of its own, or a legacy commune
    # code for a since-merged commune) — pick randomly among every zone
    # sharing that DTIR rather than leaving them unmatched.
    still_missing = resolved.isna()
    candidates = codes[still_missing].map(code_to_dtir).map(dtir_to_zone_ids)
    resolved[still_missing] = candidates.map(lambda c: rng.choice(c) if isinstance(c, list) else np.nan)

    return resolved


def execute(context):
    zones = build_unified_zoning(context)
    points = build_generator_points(context, zones)
    code_to_dtir = read_external_code_to_dtir(context)

    zones.loc[zones["dtir_name"] == "Canton du Valais", "d30_name"] = "Canton du Valais"

    print(f"Built {len(zones)} unified zones")
    print(f"Matched {points['zone_id'].notna().sum()} / {len(points)} generator points to a zone")

    context_path = context.path()
    zones.to_file(f"{context_path}/zones_unified.gpkg")

    return zones, points, code_to_dtir


# Sidecar files (.DAT/.ID/.IND/.MAP) accompanying each .TAB file are not
# tracked individually here, since they are only ever read/written together
# with it by the survey provider's tooling.
ANNEMASSE_SIG_FILES = [
    "EDGTFVG2016_ZF.TAB",
    "EDGTFVG2016_ZonesExternes.TAB",
    "EDGTFVG2016_GT.TAB",
    "RegroupementdesZonesExploitationStandard.xlsx",
]

ANNECY_SIG_FILES = [
    "EDGT_Reste74_2017_ZF_404009.TAB",
    "EDGT_Reste74_2017_GT.TAB",
    "EDGT_Reste74_2017_DTIR_D30_D10_D2.xls",
]


def validate(context):
    annemasse_path = f"{get_annemasse_path(context)}/Doc/SIG"
    annecy_path    = f"{get_annecy_path(context)}/Doc/SIG"

    paths = (
        [f"{annemasse_path}/{name}" for name in ANNEMASSE_SIG_FILES] +
        [f"{annecy_path}/{name}" for name in ANNECY_SIG_FILES]
    )

    for path in paths:
        if not os.path.exists(path):
            raise RuntimeError("File missing from EDGT: %s" % path)

    return [os.path.getsize(path) for path in paths]
