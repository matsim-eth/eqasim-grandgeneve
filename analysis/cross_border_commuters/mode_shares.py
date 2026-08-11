import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import branca

import data.od.thonon_boat_zones as thonon_boat_zones

"""
Compares commuting mode shares for the Thonon <-> Lausanne/Nyon corridor
(Arrondissement de Thonon, FR; Lausanne area / district de Nyon, CH) between
MOBPRO (census) and the HTS-matched synthetic population, per commune, plus
a CSV, a plot and an interactive bivariate map (color = pt share, opacity =
volume) built the same way as /home/asallard/Documents/Tests/Cross border
commuters/commuters_boat.py, from which this stage borrows its design.

Everything here reflects the population as matched against the HTS --
strictly *before* synthesis.population.spatial.primary.locations.
select_boat_users identifies and re-routes any boat commuters. This is the
evidence behind that stage's census-based recalibration, not a summary of
its output.
"""

ANALYSIS_FOLDER = "analysis_cross_border_commuters"

# "no transport" (MOBPRO TRANS == 1, work-from-home) is excluded before
# shares are computed -- those people do not commute, so they should not
# dilute the mode split of those who do.
MODE_ORDER = ["car", "pt", "bike", "walk"]

# Gray (all car) to blue (all public transport), as in commuters_boat.py.
PT_SHARE_COLORMAP = branca.colormap.LinearColormap(
    colors = ["#9e9e9e", "#7f93b0", "#4a7ebb", "#1f5fa9", "#08519c"],
    vmin = 0, vmax = 100
)
NO_COMMUTER_COLOR = "#f4f4f4"
DESTINATION_COLORS = { thonon_boat_zones.LAUSANNE_GROUP: "#6b4c7a", thonon_boat_zones.NYON_GROUP: "#a05a2c" }
OPACITY_RANGE = (0.15, 0.90)


def configure(context):
    context.config("output_path")
    context.config("output_prefix", "ile_de_france_")

    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.ch.municipalities")
    context.stage("data.od.thonon_boat_zones")
    context.stage("data.od.cleaned")
    context.stage("data.hts.edgt_74.adisp_merge.zones")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.spatial.primary.candidates")
    context.stage("synthesis.population.spatial.primary.locations")


def commune_names():
    """FR commune_id -> name, inverted from thonon_boat_zones' own mapping (it
    is keyed the other way around, name -> INSEE code)."""
    return { v: k for k, v in thonon_boat_zones.ARRONDISSEMENT_DE_THONON.items() }


def mobpro_long(df_od_work):
    """(commune_id, group, mode, weight) rows: MOBPRO car/pt work commutes
    from the Arrondissement de Thonon to the Lausanne area / district de
    Nyon, one row per (origin, destination, mode) cell."""
    thonon_codes = set(thonon_boat_zones.ARRONDISSEMENT_DE_THONON.values())
    destination_to_group = {
        ch_id: group
        for group, ch_ids in thonon_boat_zones.DESTINATION_GROUPS.items()
        for ch_id in ch_ids
    }

    df = df_od_work.copy()
    df["commune_id"] = df["origin_id"].astype(str)
    df["group"] = df["destination_id"].astype(str).map(destination_to_group)
    df["mode"] = df["commute_mode"].astype(str)

    df = df[df["commune_id"].isin(thonon_codes) & df["group"].notna() & df["mode"].isin(["car", "pt"])]
    return df[["commune_id", "group", "mode", "weight"]]


def synthetic_long(df_trips, home_commune, work_commune):
    """Same (commune_id, group, mode, weight) shape as mobpro_long, but from
    the HTS-matched synthetic population -- car/pt work-commute legs only,
    exactly as select_boat_users treats them (no car_passenger folding)."""
    thonon_codes = set(thonon_boat_zones.ARRONDISSEMENT_DE_THONON.values())
    destination_to_group = {
        ch_id: group
        for group, ch_ids in thonon_boat_zones.DESTINATION_GROUPS.items()
        for ch_id in ch_ids
    }

    group = work_commune.astype(str).map(destination_to_group)
    in_scope = home_commune.reindex(work_commune.index).astype(str).isin(thonon_codes) & group.notna()
    scoped_persons = in_scope[in_scope].index

    is_work_leg = (df_trips["following_purpose"] == "work") | (df_trips["preceding_purpose"] == "work")
    mode_by_person = df_trips[is_work_leg & df_trips["person_id"].isin(scoped_persons)] \
        .drop_duplicates("person_id").set_index("person_id")["mode"].astype(str)
    mode_by_person = mode_by_person[mode_by_person.isin(["car", "pt"])]

    return pd.DataFrame({
        "commune_id": home_commune.reindex(mode_by_person.index).astype(str).values,
        "group": group.reindex(mode_by_person.index).values,
        "mode": mode_by_person.values,
        "weight": 1.0,
    })


def commune_mode_counts(df_long, prefix):
    """Aggregates a (commune_id, group, mode, weight) long table to one row
    per commune of the Arrondissement de Thonon (communes with no commuter at
    all get a row of zeros, not a missing row), with nb_car/nb_pt per
    destination group and in total, and pt_share (0 = all car, 100 = all
    public transport). Columns are prefixed (e.g. "mobpro_" / "synthetic_")
    so the two sources can sit side by side in the same table."""
    thonon_codes = sorted(thonon_boat_zones.ARRONDISSEMENT_DE_THONON.values())
    result = pd.DataFrame({"commune_id": thonon_codes})

    for group in thonon_boat_zones.DESTINATION_GROUPS:
        in_group = df_long[df_long["group"] == group]
        for mode in ("car", "pt"):
            counts = in_group[in_group["mode"] == mode].groupby("commune_id")["weight"].sum()
            result[f"nb_{mode}_to_{group}"] = result["commune_id"].map(counts).fillna(0.0).round().astype(int)
        result[f"nb_to_{group}"] = result[f"nb_car_to_{group}"] + result[f"nb_pt_to_{group}"]

    result["nb_car"] = sum(result[f"nb_car_to_{group}"] for group in thonon_boat_zones.DESTINATION_GROUPS)
    result["nb_pt"] = sum(result[f"nb_pt_to_{group}"] for group in thonon_boat_zones.DESTINATION_GROUPS)
    result["nb_commuters"] = result["nb_car"] + result["nb_pt"]
    result["pt_share"] = (result["nb_pt"] / result["nb_commuters"].where(result["nb_commuters"] > 0) * 100).round(1)

    result.columns = ["commune_id"] + [f"{prefix}_{c}" for c in result.columns if c != "commune_id"]
    return result


def build_communes(context, mobpro_counts, synthetic_counts):
    """One row per commune of interest, FR (Arrondissement de Thonon, with
    per-commune MOBPRO and HTS-matched-synthetic car/pt counts and the
    boat-time advantage) and CH (Lausanne area / district de Nyon, group
    only)."""
    thonon_codes = set(thonon_boat_zones.ARRONDISSEMENT_DE_THONON.values())
    group_by_ch_id = {
        ch_id: group
        for group, ch_ids in thonon_boat_zones.DESTINATION_GROUPS.items()
        for ch_id in ch_ids
    }

    # French side: Arrondissement de Thonon
    df_fr = context.stage("data.spatial.municipalities")[["commune_id", "geometry"]].copy()
    df_fr["commune_id"] = df_fr["commune_id"].astype(str)
    df_fr = df_fr[df_fr["commune_id"].isin(thonon_codes)].copy()
    df_fr["commune_name"] = df_fr["commune_id"].map(commune_names())
    df_fr["country"] = "FR"
    df_fr["group"] = "Arrondissement de Thonon"

    advantage = context.stage("data.od.thonon_boat_zones").rename(columns = { "code_insee": "commune_id" })
    df_fr = df_fr.merge(advantage, on = "commune_id", how = "left")
    df_fr = df_fr.merge(mobpro_counts, on = "commune_id", how = "left")
    df_fr = df_fr.merge(synthetic_counts, on = "commune_id", how = "left")

    # Swiss side: Lausanne area / district de Nyon
    df_ch = context.stage("data.spatial.ch.municipalities")[["municipality_id", "municipality_name", "geometry"]].copy()
    df_ch = gpd.GeoDataFrame(df_ch, geometry = "geometry", crs = "EPSG:2056").to_crs("EPSG:2154")
    df_ch["commune_id"] = "CH" + df_ch["municipality_id"].astype(str).str.zfill(5)
    df_ch = df_ch[df_ch["commune_id"].isin(group_by_ch_id.keys())].copy()
    df_ch = df_ch.rename(columns = { "municipality_name": "commune_name" })
    df_ch["country"] = "CH"
    df_ch["group"] = df_ch["commune_id"].map(group_by_ch_id)

    shared_columns = ["commune_id", "commune_name", "country", "group", "geometry"]
    fr_extra_columns = [c for c in df_fr.columns if c not in shared_columns]

    df_communes = pd.concat([
        df_fr[shared_columns + fr_extra_columns],
        df_ch[shared_columns],
    ], ignore_index = True)

    return gpd.GeoDataFrame(df_communes, geometry = "geometry", crs = "EPSG:2154")


def edgt_covered_communes(df_zones):
    """Communes actually covered by EDGT74's own survey zoning (Annemasse
    "annemasse_zf" / Annecy "annecy_zf" sources). The scenario's
    "departments" config (["74", "01"]) takes whole administrative
    departments, but EDGT74 only really interviewed households in a small
    part of departement 01 (Pays de Gex, ~45 communes); the rest falls
    under a coarse "annemasse_external" catch-all zone used only to resolve
    trip destinations, not as a real home sample -- so it should not be
    counted as "surveyed" for a MOBPRO/HTS comparison."""
    real = df_zones[df_zones["source"].isin(["annemasse_zf", "annecy_zf"])]
    return set(real["commune_id"].dropna().astype(str))


def mobpro_mode_share(df_od_work, origin_ids = None, destination_ids = None):
    """Mode shares among MOBPRO commuters, excluding "no transport" (i.e.
    people who work from home and thus do not commute at all) from the
    denominator. origin_ids/destination_ids, when given, restrict the flows
    further, e.g. to the communes actually covered by EDGT74's own survey
    zoning (see edgt_covered_communes) rather than the full "departments"
    config scope."""
    df = df_od_work
    if origin_ids is not None:
        df = df[df["origin_id"].astype(str).isin(origin_ids)]
    if destination_ids is not None:
        df = df[df["destination_id"].astype(str).isin(destination_ids)]

    df = df[df["commute_mode"] != "no transport"]
    totals = df.groupby("commute_mode", observed = True)["weight"].sum()
    return totals / totals.sum()


def hts_mode_share_all(df_trips, home_commune, allowed_communes):
    """Whole-perimeter work-commute mode share (all modes), aligned to
    MOBPRO's categories (car_passenger/motorcycle -> car, "_loop" suffix
    stripped) and restricted to allowed_communes (see edgt_covered_communes)
    so this stays a like-for-like comparison with mobpro_mode_share's own
    restriction -- a general sanity check, not specific to the Thonon
    corridor or to the car/pt-only eligibility filter used there. There is
    no "no transport"/work-from-home equivalent here: persons without a
    work trip simply have no work leg and are already excluded by the
    is_work_leg filter below."""
    is_work_leg = (df_trips["following_purpose"] == "work") | (df_trips["preceding_purpose"] == "work")
    mode_by_person = df_trips[is_work_leg].drop_duplicates("person_id").set_index("person_id")["mode"].astype(str)

    in_scope = home_commune.reindex(mode_by_person.index).astype(str).isin(allowed_communes)
    mode_by_person = mode_by_person[in_scope]

    mode_by_person = mode_by_person.str.replace("_loop", "", regex = False).replace({
        "car_passenger": "car", "motorcycle": "car"
    })
    counts = mode_by_person.value_counts()
    return counts / counts.sum()


def build_mode_comparison(df_od_work, df_trips, home_commune, edgt_communes, mobpro_counts, synthetic_counts):
    """Wide-format table (one row per scope/source, one column per mode)
    comparing MOBPRO (census) and HTS-matched synthetic work-commute mode
    shares: the whole EDGT74-covered perimeter (all modes, as a general
    sanity check -- both sources restricted to edgt_communes so this is a
    fair like-for-like comparison, not the full "departments" admin scope)
    and the two destination zones (car/pt only, matching exactly what
    select_boat_users treats as eligible)."""
    rows = []

    def add_row(scope, source, share):
        row = dict(scope = scope, source = source)
        for mode in MODE_ORDER:
            if mode in share.index:
                row[f"{mode}_share"] = round(100 * share[mode], 2)
        rows.append(row)

    add_row("Entire perimeter (all modes)", "MOBPRO (census)", mobpro_mode_share(df_od_work, origin_ids = edgt_communes))
    add_row("Entire perimeter (all modes)", "HTS-matched synthetic", hts_mode_share_all(df_trips, home_commune, edgt_communes))

    def car_pt_share(df_counts, prefix, group_suffix):
        nb_car = df_counts[f"{prefix}_nb_car{group_suffix}"].sum()
        nb_pt = df_counts[f"{prefix}_nb_pt{group_suffix}"].sum()
        total = nb_car + nb_pt
        if total == 0:
            return pd.Series(dtype = float)
        return pd.Series({ "car": nb_car / total, "pt": nb_pt / total })

    zone_labels = { thonon_boat_zones.LAUSANNE_GROUP: "Thonon arrdt -> Lausanne area", thonon_boat_zones.NYON_GROUP: "Thonon arrdt -> District de Nyon" }
    for group, label in zone_labels.items():
        suffix = f"_to_{group}"
        add_row(label, "MOBPRO (census)", car_pt_share(mobpro_counts, "mobpro", suffix))
        add_row(label, "HTS-matched synthetic", car_pt_share(synthetic_counts, "synthetic", suffix))

    columns = ["scope", "source"] + [f"{mode}_share" for mode in MODE_ORDER]
    return pd.DataFrame(rows).reindex(columns = columns)


def plot_mode_comparison(df_comparison, path):
    share_columns = [f"{mode}_share" for mode in MODE_ORDER]
    df_long = df_comparison.melt(
        id_vars = ["scope", "source"], value_vars = share_columns,
        var_name = "mode", value_name = "share_pct"
    )
    df_long["mode"] = df_long["mode"].str.replace("_share", "", regex = False)
    df_long = df_long.dropna(subset = ["share_pct"])

    scopes = df_comparison["scope"].unique()
    sources = df_comparison["source"].unique()

    fig, axes = plt.subplots(1, len(scopes), figsize = (3.2 * len(scopes), 4), sharey = True)

    for ax, scope in zip(axes, scopes):
        df_scope = df_long[df_long["scope"] == scope]
        modes = [m for m in MODE_ORDER if m in df_scope["mode"].unique()]

        width = 0.8 / max(len(sources), 1)
        for index, source in enumerate(sources):
            df_source = df_scope[df_scope["source"] == source].set_index("mode").reindex(modes)
            positions = np.arange(len(modes)) + index * width
            ax.bar(positions, df_source["share_pct"].fillna(0.0), width = width, label = source)

        ax.set_xticks(np.arange(len(modes)) + width * (len(sources) - 1) / 2)
        ax.set_xticklabels(modes, rotation = 30, ha = "right")
        ax.set_title(scope, fontsize = 8)
        ax.grid(axis = "y", alpha = 0.3)

    axes[0].set_ylabel("Share (%)")
    axes[-1].legend(loc = "upper right", fontsize = 7)

    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)


def opacity_scale(nb_commuters, max_commuters, opacity_range = OPACITY_RANGE):
    """Square-root scale, as in commuters_boat.py: the arrondissement's flows
    are dominated by a couple of communes (Thonon-les-Bains, Evian-les-Bains),
    and a linear scale would leave every other commune indistinguishable."""
    lo, hi = opacity_range
    if max_commuters <= 0:
        return lo
    return lo + (hi - lo) * (nb_commuters / max_commuters) ** 0.5


def add_source_layer(m, df_fr, prefix, name, show):
    """One toggleable choropleth layer for a car/pt data source (MOBPRO or
    the HTS-matched synthetic population): color = pt_share (gray = all car,
    blue = all pt), opacity = commuter volume."""
    max_commuters = df_fr[f"{prefix}_nb_commuters"].max()

    def style_function(feature):
        props = feature["properties"]
        nb_commuters = props[f"{prefix}_nb_commuters"]
        pt_share = props[f"{prefix}_pt_share"]

        if not nb_commuters:
            return { "fillColor": NO_COMMUTER_COLOR, "color": "#9a9a9a", "weight": 0.6, "dashArray": "3, 3", "fillOpacity": 0.6 }

        return {
            "fillColor": PT_SHARE_COLORMAP(pt_share),
            "color": "#555555", "weight": 0.6,
            "fillOpacity": opacity_scale(nb_commuters, max_commuters),
        }

    fields = ["commune_name", f"{prefix}_nb_commuters", f"{prefix}_pt_share", f"{prefix}_nb_car", f"{prefix}_nb_pt"]
    aliases = ["Municipality:", "Commuters to the corridor:", "PT share (%):", "— car:", "— pt:"]

    folium.GeoJson(
        df_fr[["commune_name"] + [c for c in df_fr.columns if c.startswith(prefix + "_")] + ["geometry"]],
        name = name, show = show,
        style_function = style_function,
        highlight_function = lambda f: { "weight": 2.5, "color": "#222222" },
        tooltip = folium.GeoJsonTooltip(fields = fields, aliases = aliases, sticky = True),
    ).add_to(m)


def add_destination_overlays(m, df_communes):
    df_ch = df_communes[df_communes["country"] == "CH"]
    for group, color in DESTINATION_COLORS.items():
        geometries = df_ch[df_ch["group"] == group]
        folium.GeoJson(
            geometries[["commune_name", "geometry"]],
            name = f"Destination: {group}",
            style_function = lambda f, color = color: { "fillColor": color, "color": color, "weight": 2, "fillOpacity": 0.35 },
            tooltip = folium.GeoJsonTooltip(fields = ["commune_name"], aliases = [f"{group} —"]),
        ).add_to(m)


GPKG_LAYERS = {
    "Arrondissement de Thonon": "arrondissement_de_thonon",
    thonon_boat_zones.LAUSANNE_GROUP: "lausanne_area",
    thonon_boat_zones.NYON_GROUP: "district_de_nyon",
}


def write_communes_gpkg(df_communes, path):
    """One GPKG layer per group (Arrondissement de Thonon, Lausanne area,
    district de Nyon) rather than a single mixed layer, so each area can be
    loaded/styled independently in GIS software."""
    for group, layer in GPKG_LAYERS.items():
        df_communes[df_communes["group"] == group].to_file(path, layer = layer, driver = "GPKG")


def plot_communes_map(df_communes, path):
    df_map = df_communes.to_crs("EPSG:4326").copy()
    df_fr = df_map[df_map["country"] == "FR"]

    centroid = df_map.geometry.union_all().centroid
    m = folium.Map(location = [centroid.y, centroid.x], zoom_start = 10, tiles = "cartodbpositron", prefer_canvas = True)

    add_source_layer(m, df_fr, "mobpro", "MOBPRO (measured)", show = True)
    add_source_layer(m, df_fr, "synthetic", "HTS-matched synthetic (modelled)", show = False)
    add_destination_overlays(m, df_map)

    PT_SHARE_COLORMAP.caption = "Public-transport share of car+pt commuters (%)"
    PT_SHARE_COLORMAP.add_to(m)

    folium.LayerControl(collapsed = False).add_to(m)
    m.save(path)


def execute(context):
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")

    analysis_output_path = os.path.join(output_path, ANALYSIS_FOLDER)
    if not os.path.exists(analysis_output_path):
        os.mkdir(analysis_output_path)

    candidates = context.stage("synthesis.population.spatial.primary.candidates")
    home_commune = candidates["persons"].set_index("person_id")["commune_id"]

    df_work, _, _ = context.stage("synthesis.population.spatial.primary.locations")
    work_commune = df_work.drop_duplicates("person_id").set_index("person_id")["commune_id"]

    df_od_work, _ = context.stage("data.od.cleaned")
    df_trips = context.stage("synthesis.population.trips")
    df_zones, _, _ = context.stage("data.hts.edgt_74.adisp_merge.zones")
    edgt_communes = edgt_covered_communes(df_zones)

    mobpro_counts = commune_mode_counts(mobpro_long(df_od_work), "mobpro")
    synthetic_counts = commune_mode_counts(synthetic_long(df_trips, home_commune, work_commune), "synthetic")

    # Communes of interest (FR + CH), with per-commune car/pt counts (FR side)
    df_communes = build_communes(context, mobpro_counts, synthetic_counts)

    # GPKG: one layer per area (Arrondissement de Thonon, Lausanne area, district de Nyon)
    gpkg_path = f"{analysis_output_path}/{output_prefix}commuting_modes_communes.gpkg"
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    write_communes_gpkg(df_communes, gpkg_path)

    # CSV: MOBPRO vs HTS mode-share comparison
    df_comparison = build_mode_comparison(df_od_work, df_trips, home_commune, edgt_communes, mobpro_counts, synthetic_counts)
    df_comparison.to_csv(f"{analysis_output_path}/{output_prefix}commuting_modes.csv", sep = ";", index = None)

    # Plot: grouped bar chart of the comparison above
    plot_mode_comparison(df_comparison, f"{analysis_output_path}/{output_prefix}commuting_modes.png")

    # Interactive map: bivariate choropleth (pt share / volume), MOBPRO vs HTS-matched synthetic
    plot_communes_map(df_communes, f"{analysis_output_path}/{output_prefix}commuting_modes_map.html")

    print("Commuting-mode comparison written to %s" % analysis_output_path)

    return dict(communes = df_communes, mode_comparison = df_comparison)
