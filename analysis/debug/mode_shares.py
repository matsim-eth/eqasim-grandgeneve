import os
import zipfile
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import json
import html
import pydeck as pdk
from pydeck.data_utils.viewport_helpers import compute_view
import branca
from shapely import contains_xy

import matplotlib.pyplot as plt

from data.od.cleaned import PART_TIME_CODES
from analysis.cross_border_commuters.mode_shares import (
    MODE_ORDER,
    PT_SHARE_COLORMAP, opacity_scale,
)

DIFF_COLORMAP_COLORS = ["#b2182b", "#f4a582", "#f7f7f7", "#92c5de", "#2166ac"]
DIFF_COLORMAP = branca.colormap.LinearColormap(colors = DIFF_COLORMAP_COLORS, vmin = -50, vmax = 50)
# relative difference in the number of commuters, clipped at -100 % / +100 %
# Car share: gray (none) to orange (all car), to tell it apart from the blue PT share
CAR_SHARE_COLORMAP = branca.colormap.LinearColormap(
    colors = ["#9e9e9e", "#b5a08a", "#cb8a52", "#d9702f", "#b8490d"], vmin = 0, vmax = 100
)
REL_DIFF_COLORMAP = branca.colormap.LinearColormap(colors = DIFF_COLORMAP_COLORS, vmin = -100, vmax = 100)

ANALYSIS_FOLDER = "analysis_debug"

BASEMAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


NO_TRANSPORT = "no transport"

MATSIM_TRIPS_PATH = "/home/asallard/Documents/Results/20261018_ch_5pct_60it/matsim.simulation.run__f196524380fb58b98b95774e38776793.cache/simulation_output/output_trips.csv.gz"
PERIMETER_PATH = "/home/asallard/Euler_data/ch_data/pipeline/spatial/MMT/MMT_extent_cleaned.gpkg"
MATSIM_CRS = "EPSG:2056"

# MOBPRO (INSEE TRANS): 1 no transport (work from home), 2 walk, 3 bike,
# 4 motorised two-wheeler, 5 car, 6 pt (4 and 5 folded into car, as in data.od.cleaned)
MOBPRO_MODES = { 1: NO_TRANSPORT, 2: "walk", 3: "bike", 4: "car", 5: "car", 6: "pt" }

CH_PREFIX = "CH"

# (map column prefix, label, config key holding the path relative to data_path),
# in chronological order.
MOBPRO_SOURCES = [
    ("mobpro2017", "MOBPRO 2017 (census)", "od_pro_2017_path"),
    ("mobpro", "MOBPRO 2022 (census)", "od_pro_2022_path"),
    ("mobpro2023", "MOBPRO 2023 (census)", "od_pro_2023_path"),
]

# Release the difference views are computed against
REFERENCE_PREFIX = "mobpro2023"
REFERENCE_LABEL = "MOBPRO 2023"

SYNTHETIC_LABEL = "Synthetic population"
MATSIM_LABEL = "MATSim simulation"

# Left to right in the plot: MOBPRO releases, synthetic population (before
# MATSim), MATSim (after)
SOURCE_ORDER = [label for _, label, _ in MOBPRO_SOURCES] + [SYNTHETIC_LABEL, MATSIM_LABEL]
SOURCE_COLORS = {
    "MOBPRO 2017 (census)": "#d4d4d4",
    "MOBPRO 2022 (census)": "#a0a0a0",
    "MOBPRO 2023 (census)": "#6b6b6b",
    SYNTHETIC_LABEL: "#8ecae6",
    MATSIM_LABEL: "#1d4e89",
}


def configure(context):
    context.config("data_path")
    context.config("od_pro_2022_path", "rp_2022/RP2022_mobpro.parquet")
    context.config("od_pro_2017_path", "rp_2022/RP2017_MOBPRO_csv.zip")
    context.config("od_pro_2023_path", "rp_2022/RP2023_mobpro.parquet")
    context.config("output_path")
    context.config("output_prefix", "")
    context.config("random_seed")

    context.config("debug_matsim_trips_path", MATSIM_TRIPS_PATH)
    context.config("debug_matsim_sampling_rate", 0.1)
    context.config("debug_perimeter_path", PERIMETER_PATH)

    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.ch.municipalities")
    context.stage("data.od.read_CH_mun_insee_names")
    context.stage("data.od.part_time")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.spatial.primary.candidates")
    context.stage("synthesis.population.spatial.primary.locations")


def canonical_mode(mode_series):
    modes = mode_series.astype(str).str.replace("_loop", "", regex = False)
    return modes.replace({"car_passenger": "car", "motorcycle": "car"})


def mode_shares(modes, weights):
    df = pd.DataFrame({"mode": modes.values, "weight": weights.values})
    df = df[df["mode"] != NO_TRANSPORT]

    totals = df.groupby("mode")["weight"].sum()
    n = totals.sum()

    return (totals / n if n > 0 else totals), n


def find_perimeter_communes(df_municipalities, perimeter_path):
    """Communes whose representative point lies inside the perimeter."""
    perimeter = gpd.read_file(perimeter_path).to_crs(df_municipalities.crs).union_all()
    points = df_municipalities.geometry.representative_point()
    return set(df_municipalities.loc[points.within(perimeter), "commune_id"].astype(str))


def find_swiss_perimeter_ids(df_ch_municipalities, insee_to_ch_id, perimeter_path):
    """MOBPRO destination codes ("SU...") of the Swiss municipalities whose
    representative point lies inside the perimeter, as the "CH#####" ids."""
    perimeter = gpd.read_file(perimeter_path).to_crs(df_ch_municipalities.crs).union_all()
    points = df_ch_municipalities.geometry.representative_point()
    bfs_ids = df_ch_municipalities.loc[points.within(perimeter), "municipality_id"].astype(int)
    ch_ids = set("CH" + bfs_ids.astype(str).str.zfill(5))
    return set(insee_to_ch_id.loc[insee_to_ch_id["municipality_id"].isin(ch_ids), "insee_code"])


def read_synthetic(df_trips, home_commune, work_commune, perimeter_communes):
    """Synthetic population (synthesis.population.trips), restricted to
    residents of perimeter communes, in the same (general, work) shape as
    read_matsim. Cross-border = assigned workplace commune is a "CH..." one
    (only Swiss communes of the configured cantons exist there). Every
    home -> work and work -> home trip is used, each weighted 1 / (the
    person's number of such trips) so that a person counts once, like in
    MOBPRO. The population is
    already at full scale."""
    df = df_trips[df_trips["person_id"].map(home_commune).astype(str).isin(perimeter_communes)].copy()
    df["mode"] = canonical_mode(df["mode"])

    general = pd.DataFrame({ "mode": df["mode"].values, "weight": 1.0 })

    is_work_leg = (
        ((df["preceding_purpose"] == "home") & (df["following_purpose"] == "work")) |
        ((df["preceding_purpose"] == "work") & (df["following_purpose"] == "home"))
    )
    df_work = df[is_work_leg]
    trip_weight = 1.0 / df_work.groupby("person_id")["person_id"].transform("size")

    work = pd.DataFrame({
        "commune_id": df_work["person_id"].map(home_commune).astype(str).values,
        "mode": df_work["mode"].values,
        "weight": trip_weight.values,
        "is_cross_border": df_work["person_id"].map(work_commune).astype(str).str.startswith(CH_PREFIX).values,
    })

    return general, work


def read_matsim(path, sampling_rate, df_municipalities, df_ch_municipalities, perimeter_communes):
    """MATSim output_trips.csv.gz, restricted to French residents living in a
    perimeter commune (freight/lcv agents and Swiss residents have no French
    home commune, hence drop out; MOBPRO only covers French residents).
    Returns (general, work): every trip's mode, and every home -> work and
    work -> home trip with the person's residence commune and is_cross_border
    (that trip's workplace is inside Switzerland). General trips weigh
    1 / sampling_rate; work trips weigh 1 / sampling_rate / (the person's
    number of such trips), so that a person counts once, like in MOBPRO."""
    columns = ["person", "main_mode", "start_activity_type", "end_activity_type", "start_x", "start_y", "end_x", "end_y"]
    df = pd.read_csv(path, sep = ";", usecols = columns, dtype = { "person": str }, low_memory = False)
    df = df[df["person"].str.fullmatch(r"\d+")].copy()

    # Home commune: first trip leaving home
    home = df[df["start_activity_type"] == "home"].drop_duplicates("person")
    points = gpd.GeoDataFrame(home[["person"]], geometry = gpd.points_from_xy(home["start_x"], home["start_y"]), crs = MATSIM_CRS)
    points = gpd.sjoin(points.to_crs(df_municipalities.crs), df_municipalities[["commune_id", "geometry"]], predicate = "within")
    home_commune = points.drop_duplicates("person").set_index("person")["commune_id"].astype(str)
    home_commune = home_commune[home_commune.isin(perimeter_communes)]

    df = df[df["person"].isin(home_commune.index)].copy()
    df["mode"] = canonical_mode(df["main_mode"])

    # "outside"/"truck" are not resident travel modes
    unsupported = ~df["mode"].isin(MODE_ORDER)
    print("MATSim: dropping %d / %d trips with a mode outside %s" % (unsupported.sum(), len(df), MODE_ORDER))
    df = df[~unsupported]

    weight = 1.0 / sampling_rate
    general = pd.DataFrame({ "mode": df["mode"].values, "weight": weight })

    is_work_leg = (
        ((df["start_activity_type"] == "home") & (df["end_activity_type"] == "work")) |
        ((df["start_activity_type"] == "work") & (df["end_activity_type"] == "home"))
    )
    df_work = df[is_work_leg]
    trip_weight = weight / df_work.groupby("person")["person"].transform("size")
    work_is_end = (df_work["end_activity_type"] == "work").values
    work_x = np.where(work_is_end, df_work["end_x"], df_work["start_x"])
    work_y = np.where(work_is_end, df_work["end_y"], df_work["start_y"])

    switzerland = df_ch_municipalities.to_crs(MATSIM_CRS).union_all()

    work = pd.DataFrame({
        "commune_id": home_commune.reindex(df_work["person"]).values,
        "mode": df_work["mode"].values,
        "weight": trip_weight.values,
        "is_cross_border": contains_xy(switzerland, work_x, work_y),
    })

    return general, work


def read_mobpro(path, employment_time_distribution, random, swiss_perimeter_codes):
    """Reads a MOBPRO parquet directly (whole file, no commune filtering):
    origin_id (residence commune, arrondissement where applicable),
    commute_mode, weight, and is_cross_border (works in Switzerland: DCLT is
    "99999", i.e. abroad, and DCFLT, the foreign destination, is the "SU" code
    of a Swiss municipality inside the perimeter, swiss_perimeter_codes).
    Domestic = any destination inside France; commuters to other foreign
    countries are in neither group.

    "weight" is in full-time equivalents: IPONDI times an employment time,
    which is 1 for full-time workers and, for each part-time record (TP), one
    draw from employment_time_distribution (same approach as
    data.od.cleaned), because part-time workers commute on fewer days and
    MATSim only represents one day. "person_weight" is the plain IPONDI."""
    columns = ["COMMUNE", "ARM", "TRANS", "IPONDI", "DCLT", "DCFLT", "TP"]

    if path.endswith(".zip"):
        # INSEE detail file (e.g. RP2017_MOBPRO_csv.zip): ";"-separated CSV
        # with the same columns and codes as the parquet releases
        with zipfile.ZipFile(path) as archive:
            name = [n for n in archive.namelist() if n.startswith("FD_") and n.endswith(".csv")][0]
            with archive.open(name) as f:
                df = pd.read_csv(f, sep = ";", usecols = columns, dtype = str)
    else:
        # some releases store the code columns as categoricals
        df = pl.read_parquet(path, columns = columns).with_columns(
            pl.col(["COMMUNE", "ARM", "DCLT", "DCFLT"]).cast(pl.String)
        ).to_pandas()

    origin_id = df["COMMUNE"].where(df["ARM"].str.contains("Z"), df["ARM"])
    abroad = df["DCLT"] == "99999"

    person_weight = pd.to_numeric(df["IPONDI"]).astype(float).values
    is_part_time = pd.to_numeric(df["TP"]).isin(PART_TIME_CODES).values
    employment_time = np.ones(len(df))
    employment_time[is_part_time] = random.choice(
        employment_time_distribution.index.values, size = np.count_nonzero(is_part_time), p = employment_time_distribution.values)

    return pd.DataFrame({
        "origin_id": origin_id.values,
        "commute_mode": pd.to_numeric(df["TRANS"]).map(MOBPRO_MODES).values,
        "weight": person_weight * employment_time,
        "person_weight": person_weight,
        "is_cross_border": (abroad & df["DCFLT"].isin(swiss_perimeter_codes)).values,
        "is_domestic": (~abroad).values,
    })


def mobpro_commuting(df_mobpro):

    results = {}
    for label, column in [("domestic", "is_domestic"), ("cross_border", "is_cross_border")]:
        subset = df_mobpro[df_mobpro[column]]
        results[label] = mode_shares(subset["commute_mode"], subset["weight"])

    return results


def general_mode_shares(df_general):

    return mode_shares(df_general["mode"], df_general["weight"])


def work_mode_shares(df_work):

    results = {}
    for cross_border, label in [(False, "domestic"), (True, "cross_border")]:
        subset = df_work[df_work["is_cross_border"] == cross_border]
        results[label] = mode_shares(subset["mode"], subset["weight"])

    return results


def commune_mode_shares(df, commune_column, mode_column, weight_column):

    totals = df.groupby([commune_column, mode_column])[weight_column].sum().unstack(fill_value = 0.0)
    nb_commuters = totals.sum(axis = 1)

    result = { "commune_id": totals.index.astype(str), "nb_commuters": nb_commuters.round().astype(int).values }
    for mode, _, _ in MAP_MODES:
        nb_mode = totals[mode] if mode in totals.columns else pd.Series(0.0, index = totals.index)
        result[f"{mode}_share"] = (nb_mode / nb_commuters.where(nb_commuters > 0) * 100).round(1).values

    return pd.DataFrame(result)


def mobpro_commune_mode_shares(df_mobpro):

    df = df_mobpro[df_mobpro["is_cross_border"] & (df_mobpro["commute_mode"] != NO_TRANSPORT)]
    return commune_mode_shares(df, "origin_id", "commute_mode", "weight")


def cross_border_commune_mode_shares(df_work):

    return commune_mode_shares(df_work[df_work["is_cross_border"]], "commune_id", "mode", "weight")


def build_pt_share_communes(context, pt_shares):
    """pt_shares: prefix -> commune_mode_shares table, for every source shown on
    the map (must include REFERENCE_PREFIX, "synthetic" and "matsim", the
    latter two being compared against the former)."""

    df_fr = context.stage("data.spatial.municipalities")[["commune_id", "geometry"]].copy()
    df_fr["commune_id"] = df_fr["commune_id"].astype(str)

    # Only communes where at least one source has cross-border commuters
    df_fr = df_fr[df_fr["commune_id"].isin(set().union(*[set(df["commune_id"]) for df in pt_shares.values()]))].copy()

    for prefix, df in pt_shares.items():
        df_fr = df_fr.merge(df.add_prefix(prefix + "_").rename(columns = { prefix + "_commune_id": "commune_id" }), on = "commune_id", how = "left")
        df_fr[f"{prefix}_nb_commuters"] = df_fr[f"{prefix}_nb_commuters"].fillna(0.0).round().astype(int)

    for prefix in ("synthetic", "matsim"):
        both_present = (df_fr[f"{prefix}_nb_commuters"] > 0) & (df_fr[f"{REFERENCE_PREFIX}_nb_commuters"] > 0)
        for mode, _, _ in MAP_MODES:
            df_fr[f"diff_{prefix}_{mode}_share"] = np.nan
            df_fr.loc[both_present, f"diff_{prefix}_{mode}_share"] = (
                df_fr.loc[both_present, f"{prefix}_{mode}_share"] - df_fr.loc[both_present, f"{REFERENCE_PREFIX}_{mode}_share"]
            ).round(1)
        df_fr[f"diff_{prefix}_nb_commuters"] = df_fr[[f"{prefix}_nb_commuters", f"{REFERENCE_PREFIX}_nb_commuters"]].min(axis = 1)
        nb_reference = df_fr[f"{REFERENCE_PREFIX}_nb_commuters"]
        df_fr[f"diffrel_{prefix}"] = ((df_fr[f"{prefix}_nb_commuters"] - nb_reference) / nb_reference.where(nb_reference > 0) * 100).round(1)

    return gpd.GeoDataFrame(df_fr, geometry = "geometry", crs = df_fr.crs)


NO_COMMUTER_FILL = [244, 244, 244, 150]

# (mode as named after canonical_mode/MOBPRO_MODES, label, share colormap).
# Car = car driver + car passenger (+ motorised two-wheelers, as in data.od.cleaned).
MAP_MODES = [("pt", "PT", PT_SHARE_COLORMAP), ("car", "Car", CAR_SHARE_COLORMAP)]


def share_key(mode, prefix):
    """View key of a source's share of `mode` ("pt" keys are just the source prefix)."""
    return prefix if mode == "pt" else f"{mode}_{prefix}"


def diff_key(mode, prefix):
    return f"diff_{prefix}" if mode == "pt" else f"diff_{mode}_{prefix}"

# Only one PolygonLayer is drawn: it carries one "<key>_color" column per
# selectable view and the dropdown switches which one is used (the geometry
# is embedded once, not once per view)
DIFF_SOURCES = [("synthetic", "Synthetic", SYNTHETIC_LABEL), ("matsim", "MATSim", MATSIM_LABEL)]
# Sources with a relative commuter-count difference view
COUNT_DIFF_SOURCES = [("matsim", "MATSim", MATSIM_LABEL)]


def colormap_rgb(colormap, value):
    color = colormap(value)
    return [int(color[i:i + 2], 16) for i in (1, 3, 5)]


def colormap_css_stops(colormap, low, high, steps = 6):
    return ", ".join(colormap(low + (high - low) * i / (steps - 1))[:7] for i in range(steps))


def polygon_records(df_map, share_sources):
    """One record per polygon part (PolygonLayer draws polygons, not
    multipolygons), carrying every view's fill colour ("<key>_color", see
    share_key / diff_key / "diffrel_<prefix>") and one tooltip listing all
    sources. share_sources: [(prefix, label)]."""
    max_share = { prefix: df_map[f"{prefix}_nb_commuters"].max() for prefix, _ in share_sources }
    max_diff = { prefix: df_map[f"diff_{prefix}_nb_commuters"].max() for prefix, _, _ in DIFF_SOURCES }

    records = []
    for _, row in df_map.iterrows():
        record = {}
        tooltip = ["<b>Municipality %s</b>" % row["commune_id"], "Cross-border commuters (residents), " + " | ".join("%s share" % label for _, label, _ in MAP_MODES) + ":"]

        for prefix, label in share_sources:
            nb_commuters = row[f"{prefix}_nb_commuters"]
            if not nb_commuters:
                for mode, _, _ in MAP_MODES:
                    record[f"{share_key(mode, prefix)}_color"] = NO_COMMUTER_FILL
                tooltip.append("%s: none" % label)
            else:
                alpha = int(255 * opacity_scale(nb_commuters, max_share[prefix]))
                for mode, _, colormap in MAP_MODES:
                    record[f"{share_key(mode, prefix)}_color"] = colormap_rgb(colormap, row[f"{prefix}_{mode}_share"]) + [alpha]
                tooltip.append("%s: %s" % (label, " | ".join("%s%%" % row[f"{prefix}_{mode}_share"] for mode, _, _ in MAP_MODES)))

        for prefix, short_label, _ in DIFF_SOURCES:
            nb_commuters = row[f"diff_{prefix}_nb_commuters"]
            parts = []
            for mode, _, _ in MAP_MODES:
                diff = row[f"diff_{prefix}_{mode}_share"]
                if not nb_commuters or pd.isna(diff):
                    record[f"{diff_key(mode, prefix)}_color"] = NO_COMMUTER_FILL
                else:
                    alpha = int(255 * opacity_scale(nb_commuters, max_diff[prefix]))
                    record[f"{diff_key(mode, prefix)}_color"] = colormap_rgb(DIFF_COLORMAP, diff) + [alpha]
                    parts.append("%+.1f pp" % diff)
            if parts:
                tooltip.append("%s - %s: %s" % (short_label, REFERENCE_LABEL, " | ".join(parts)))

        for prefix, short_label, _ in COUNT_DIFF_SOURCES:
            nb_reference = row[f"{REFERENCE_PREFIX}_nb_commuters"]
            nb_source = row[f"{prefix}_nb_commuters"]
            relative = row[f"diffrel_{prefix}"]
            if not nb_reference and not nb_source:
                record[f"diffrel_{prefix}_color"] = NO_COMMUTER_FILL
                continue
            if pd.isna(relative):
                # commuters in the source, none in the reference: no relative difference
                record[f"diffrel_{prefix}_color"] = NO_COMMUTER_FILL
                tooltip.append("Commuters, %s - %s: n/a (%d vs %d FTE)" % (short_label, REFERENCE_LABEL, nb_source, nb_reference))
                continue
            record[f"diffrel_{prefix}_color"] = colormap_rgb(REL_DIFF_COLORMAP, float(np.clip(relative, -100, 100))) + [220]
            tooltip.append("Cross border commuters: %s - %s: %+.0f%% (%d vs %d FTE)" % (short_label, REFERENCE_LABEL, relative, nb_source, nb_reference))

        record["_tooltip_html"] = "<br/>".join(tooltip)

        geometry = row["geometry"]
        parts = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in parts:
            rings = [part.exterior.coords] + [ring.coords for ring in part.interiors]
            records.append({
                "polygon": [[[round(x, 5), round(y, 5)] for x, y, *_ in ring] for ring in rings],
                **record,
            })

    return pd.DataFrame(records)


def inject_layer_selector(path, options, default_key):
    """Adds a "Show" dropdown + colour legend to the saved pydeck HTML and a
    script switching the fill colour column of the polygon layer, like
    eqasim-switzerland's analysis/counts/flow_metrics.py (accessor mode).
    options: [{key, label, stops, low, high}]."""
    options_html = "\n".join(
        '<option value="%s"%s>%s</option>' % (o["key"], " selected" if o["key"] == default_key else "", html.escape(o["label"]))
        for o in options
    )
    info = { o["key"]: { "label": o["label"], "stops": o["stops"], "low": o["low"], "high": o["high"] } for o in options }

    legend = """
<div style="position:absolute;z-index:20;bottom:12px;left:12px;padding:10px 12px;border-radius:7px;
            background:rgba(255,255,255,0.95);box-shadow:0 2px 10px rgba(0,0,0,0.28);color:#222;font:12px/1.3 Arial, sans-serif;">
  <label style="display:block;margin-bottom:8px;"><div style="margin-bottom:3px;">Show</div>
    <select id="dbg-select" style="width:300px;">%s</select></label>
  <div id="dbg-title" style="font-weight:700;margin-bottom:6px;max-width:300px;"></div>
  <div id="dbg-bar" style="width:220px;height:12px;border:1px solid #777;"></div>
  <div style="display:flex;justify-content:space-between;width:220px;margin-top:3px;"><span id="dbg-low"></span><span id="dbg-high"></span></div>
  <div style="margin-top:4px;color:#777;">Light gray = no commuters (or nothing to compare)</div>
</div>
""" % options_html

    # updateTriggers is required: without it deck.gl can keep the previous
    # colours when only the accessor function changes (see flow_metrics.py)
    script = """
<script>
(function () {
  const info = %s;
  const select = document.getElementById("dbg-select");
  function apply(key) {
    const o = info[key];
    document.getElementById("dbg-title").textContent = o.label;
    document.getElementById("dbg-bar").style.background = "linear-gradient(to right, " + o.stops + ")";
    document.getElementById("dbg-low").textContent = o.low;
    document.getElementById("dbg-high").textContent = o.high;
    if (typeof deckInstance === "undefined" || !deckInstance.props || !deckInstance.props.layers) { return; }
    deckInstance.setProps({ layers: deckInstance.props.layers.map(function (layer) {
      return layer.id.startsWith("pt-share::") ? layer.clone({
        getFillColor: function (d) { return d[key + "_color"]; },
        updateTriggers: { getFillColor: key },
      }) : layer;
    }) });
  }
  select.addEventListener("change", function () { apply(select.value); });
  apply(select.value);
})();
</script>
""" % json.dumps(info)

    with open(path, encoding = "utf-8") as f:
        page = f.read()
    page = page.replace("</body>", legend + "\n</body>", 1).replace("</html>", script + "\n</html>", 1)
    with open(path, "w", encoding = "utf-8") as f:
        f.write(page)


def plot_pt_share_map(df_communes, path):
    """Interactive pydeck map (same CARTO positron basemap as the
    eqasim-switzerland cross-border maps) of the PT and car shares among
    cross-border (-> Switzerland) work commuters, by residence commune: one
    view per source, plus synthetic / MATSim minus MOBPRO 2023, and the
    relative difference in commuter numbers, switched with a dropdown."""
    df_map = df_communes.to_crs("EPSG:4326").copy()

    share_sources = [(prefix, label) for prefix, label, _ in MOBPRO_SOURCES] + [("synthetic", SYNTHETIC_LABEL), ("matsim", MATSIM_LABEL)]
    diff_stops = colormap_css_stops(DIFF_COLORMAP, -50, 50)

    relative_stops = colormap_css_stops(REL_DIFF_COLORMAP, -100, 100)

    default_key = REFERENCE_PREFIX
    options = []
    for mode, mode_label, colormap in MAP_MODES:
        mode_stops = colormap_css_stops(colormap, 0, 100)
        options += [
            { "key": share_key(mode, prefix), "label": "%s share of cross-border commuters (%%): %s" % (mode_label, label), "stops": mode_stops, "low": "0", "high": "100" }
            for prefix, label in share_sources
        ]
        options += [
            { "key": diff_key(mode, prefix), "label": "Difference in %s share (%s - %s, pp)" % (mode_label, short_label, REFERENCE_LABEL), "stops": diff_stops, "low": "-50", "high": "+50" }
            for prefix, short_label, _ in DIFF_SOURCES
        ]
    options += [
        { "key": f"diffrel_{prefix}", "label": "Relative difference in number of cross-border commuters (%s vs %s in full-time equivalents, %%)" % (short_label, REFERENCE_LABEL), "stops": relative_stops, "low": "-100%", "high": "+100%" }
        for prefix, short_label, _ in COUNT_DIFF_SOURCES
    ]

    layer = pdk.Layer(
        "PolygonLayer", polygon_records(df_map, share_sources),
        id = "pt-share::all::0",
        pickable = True, auto_highlight = True,
        get_polygon = "polygon", get_fill_color = f"{default_key}_color",
        get_line_color = [85, 85, 85], get_line_width = 1,
        line_width_min_pixels = 1, stroked = True, filled = True,
        highlight_color = [0, 200, 255],
    )

    minx, miny, maxx, maxy = df_map.total_bounds
    margin_x, margin_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    view_state = compute_view([[minx - margin_x, miny - margin_y], [maxx + margin_x, maxy + margin_y]])
    view_state.pitch = 0

    deck = pdk.Deck(
        layers = [layer],
        initial_view_state = view_state,
        tooltip = { "html": "{_tooltip_html}", "style": { "color": "white", "background-color": "black", "padding": "10px" } },
        map_style = BASEMAP_STYLE,
    )
    deck.to_html(path, notebook_display = False)
    inject_layer_selector(path, options, default_key)


def plot_mode_comparison(df_comparison, path):
    """Grouped bars, one panel per scope, one bar per source in SOURCE_ORDER
    (same slot and colour in every panel: gray gradient for the MOBPRO
    releases, light blue synthetic population, dark blue MATSim)."""
    share_columns = [f"{mode}_share" for mode in MODE_ORDER]
    scopes = list(df_comparison["scope"].unique())
    sources = [s for s in SOURCE_ORDER if s in set(df_comparison["source"])]

    fig, axes = plt.subplots(1, len(scopes), figsize = (4.2 * len(scopes), 4.5), sharey = True, squeeze = False)
    axes = axes[0]

    width = 0.8 / len(sources)
    for ax, scope in zip(axes, scopes):
        df_scope = df_comparison[df_comparison["scope"] == scope].set_index("source")

        for index, source in enumerate(sources):
            if source not in df_scope.index:
                continue
            values = df_scope.loc[source, share_columns].astype(float).fillna(0.0).values
            ax.bar(np.arange(len(MODE_ORDER)) + index * width, values, width = width,
                   color = SOURCE_COLORS[source], label = source)

        ax.set_xticks(np.arange(len(MODE_ORDER)) + width * (len(sources) - 1) / 2)
        ax.set_xticklabels(MODE_ORDER)
        ax.set_title(scope, fontsize = 9)
        ax.grid(axis = "y", alpha = 0.3)

    axes[0].set_ylabel("Share (%)")
    axes[-1].legend(loc = "upper right", fontsize = 7)

    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)


def build_comparison(mobpro_results, synthetic_general_result, synthetic_commuting_result, matsim_general_result, matsim_commuting_result):
    rows = []

    def add_row(scope, source, result):
        share, n = result
        row = dict(scope = scope, source = source, n = round(n, 1))
        for mode in MODE_ORDER:
            if mode in share.index:
                row[f"{mode}_share"] = round(100 * share[mode], 2)
        rows.append(row)

    add_row("General (all trips)", SYNTHETIC_LABEL, synthetic_general_result)
    add_row("General (all trips)", MATSIM_LABEL, matsim_general_result)

    labels = { "domestic": "Commuting (domestic)", "cross_border": "Commuting (cross-border, -> Switzerland)" }
    for key, scope in labels.items():
        for _, label, _ in MOBPRO_SOURCES:
            add_row(scope, label, mobpro_results[label][key])
        add_row(scope, SYNTHETIC_LABEL, synthetic_commuting_result[key])
        add_row(scope, MATSIM_LABEL, matsim_commuting_result[key])

    columns = ["scope", "source", "n"] + [f"{mode}_share" for mode in MODE_ORDER]
    return pd.DataFrame(rows).reindex(columns = columns)


def execute(context):
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")

    analysis_output_path = os.path.join(output_path, ANALYSIS_FOLDER)
    if not os.path.exists(analysis_output_path):
        os.mkdir(analysis_output_path)

    df_municipalities = context.stage("data.spatial.municipalities")
    perimeter_communes = find_perimeter_communes(df_municipalities, context.config("debug_perimeter_path"))
    print("Perimeter: %d communes" % len(perimeter_communes))

    employment_time_distribution = context.stage("data.od.part_time")
    random = np.random.default_rng(context.config("random_seed"))

    swiss_perimeter_codes = find_swiss_perimeter_ids(
        context.stage("data.spatial.ch.municipalities"), context.stage("data.od.read_CH_mun_insee_names"),
        context.config("debug_perimeter_path"))
    print("Perimeter: %d Swiss destination codes" % len(swiss_perimeter_codes))

    # (os.path.join so an absolute path in the config overrides data_path)
    df_mobpros = {}
    for prefix, _, config_key in MOBPRO_SOURCES:
        df = read_mobpro(os.path.join(context.config("data_path"), context.config(config_key)), employment_time_distribution, random, swiss_perimeter_codes)
        df_mobpros[prefix] = df[df["origin_id"].astype(str).isin(perimeter_communes)]

    df_matsim_general, df_matsim_work = read_matsim(
        context.config("debug_matsim_trips_path"), context.config("debug_matsim_sampling_rate"),
        df_municipalities, context.stage("data.spatial.ch.municipalities"), perimeter_communes
    )

    df_synthetic_trips = context.stage("synthesis.population.trips")
    df_synthetic_work, _, _ = context.stage("synthesis.population.spatial.primary.locations")
    work_commune = df_synthetic_work.drop_duplicates("person_id").set_index("person_id")["commune_id"]
    home_commune = context.stage("synthesis.population.spatial.primary.candidates")["persons"].set_index("person_id")["commune_id"]

    df_synthetic_general, df_synthetic_work = read_synthetic(df_synthetic_trips, home_commune, work_commune, perimeter_communes)

    mobpro_results = { label: mobpro_commuting(df_mobpros[prefix]) for prefix, label, _ in MOBPRO_SOURCES }

    df_comparison = build_comparison(
        mobpro_results,
        general_mode_shares(df_synthetic_general), work_mode_shares(df_synthetic_work),
        general_mode_shares(df_matsim_general), work_mode_shares(df_matsim_work)
    )

    csv_path = f"{analysis_output_path}/{output_prefix}debug_mode_shares.csv"
    df_comparison.to_csv(csv_path, sep = ";", index = None)

    plot_path = f"{analysis_output_path}/{output_prefix}debug_mode_shares.png"
    plot_mode_comparison(df_comparison, plot_path)

    # Interactive map: pt share among cross-border work commuters, by
    # residence commune, MOBPRO releases vs synthetic population vs MATSim
    # (toggleable layers), plus synthetic / MATSim minus the reference release
    pt_shares = { prefix: mobpro_commune_mode_shares(df_mobpros[prefix]) for prefix, _, _ in MOBPRO_SOURCES }
    pt_shares["synthetic"] = cross_border_commune_mode_shares(df_synthetic_work)
    pt_shares["matsim"] = cross_border_commune_mode_shares(df_matsim_work)

    df_pt_share_communes = build_pt_share_communes(context, pt_shares)
    map_path = f"{analysis_output_path}/{output_prefix}debug_pt_share_map.html"
    plot_pt_share_map(df_pt_share_communes, map_path)

    print("Mode share comparison written to %s" % analysis_output_path)

    return df_comparison
