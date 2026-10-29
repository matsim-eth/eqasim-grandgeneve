import os
import json
import html
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import pydeck as pdk
from pydeck.data_utils.viewport_helpers import compute_view
import branca
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from analysis.debug.mode_shares import (
    ANALYSIS_FOLDER, BASEMAP_STYLE, DIFF_COLORMAP_COLORS, CH_PREFIX,
    colormap_rgb, colormap_css_stops,
)

"""
Compares commuting flows (home commune -> work commune) between MOBPRO and the
synthetic population, as a heatmap (main origins x main destinations) and an
interactive map (arcs between communes; as fill colour, intra-commune flows
or the relative difference of all workers living in the commune), each
showing MOBPRO, the synthetic population and their difference.

- MOBPRO: workers by usual workplace (IPONDI, all transport modes including
  work from home, i.e. exactly the matrix data.od.weighted samples from),
  restricted to the home communes present in the synthetic population.
- Synthetic population: agents with a work location
  (synthesis.population.spatial.primary.locations), i.e. agents whose matched
  HTS chain contains a work activity on the simulated day.

There are fewer synthetic commuters than MOBPRO workers (not everybody goes to
work on a given day), so the heatmap and the CSV also show the difference with
the synthetic flows rescaled to the MOBPRO total, which shows where the
spatial pattern differs independently of that volume gap. The map only shows
actual counts.

The comparison is generic in the second source (see SYNTHETIC_SOURCE and
write_outputs), which analysis.debug.commute_flows_census reuses.
"""

HEATMAP_TOP_ORIGINS = 20
HEATMAP_TOP_DESTINATIONS = 20

OTHER_ORIGINS = "Other home communes"
OTHER_FR = "Other France"
OTHER_CH = "Other Switzerland"

# How the source compared with MOBPRO is named in titles, legends, tooltips
# and the CSV columns
SYNTHETIC_SOURCE = {
    "name": "Synthetic population",
    "short": "synthetic",
    "description": "agents with a work activity on the simulated day",
    "unit": "commuters",
    "column": "synthetic",
}

FLOW_VIEWS = ["mobpro", "synthetic"]

FLOW_SOURCE_COLOR = [230, 120, 20]
FLOW_TARGET_COLOR = [30, 70, 160]
INTRA_COLORMAP = branca.colormap.LinearColormap(colors = ["#fff5eb", "#fdbe85", "#fd8d3c", "#d94701", "#8c2d04"], vmin = 0, vmax = 1)

MAX_ARC_WIDTH = 14
DEFAULT_THRESHOLD = 50

# Commune fill by all workers living in the commune (any work commune):
# relative difference to MOBPRO, full colour at +/- RESIDENT_LIMIT; grey for
# communes with fewer than RESIDENT_MINIMUM MOBPRO workers
RESIDENT_LIMIT = 0.3
RESIDENT_MINIMUM = 20


def configure(context):
    context.config("output_path")
    context.config("output_prefix", "")
    context.config("data_path")
    context.config("codes_path", "codes_2024/reference_IRIS_geo2024.zip")
    context.config("codes_xlsx", "reference_IRIS_geo2024.xlsx")

    context.stage("data.od.cleaned")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.ch.municipalities")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.home.zones")
    context.stage("synthesis.population.spatial.primary.locations")


def views(source):
    """Views of the interactive map: (key, label); the key is also the column."""
    return [
        ("mobpro", "MOBPRO: workers by usual workplace"),
        ("synthetic", "%s: %s" % (source["name"], source["description"])),
        ("diff", "Difference: %s - MOBPRO" % source["short"]),
    ]


def commune_names(context):
    """commune_id -> "Name (id)", French names from the INSEE IRIS reference
    (same file as data.spatial.codes), Swiss ones from swissBOUNDARIES."""
    with zipfile.ZipFile(os.path.join(context.config("data_path"), context.config("codes_path"))) as archive:
        with archive.open(context.config("codes_xlsx")) as f:
            df = pd.read_excel(f, skiprows = 5, sheet_name = "Emboitements_IRIS", usecols = ["DEPCOM", "LIBCOM"], dtype = str)
    names = df.drop_duplicates("DEPCOM").set_index("DEPCOM")["LIBCOM"]

    df_ch = context.stage("data.spatial.ch.municipalities")
    ch_ids = CH_PREFIX + df_ch["municipality_id"].astype(int).astype(str).str.zfill(5)
    names = pd.concat([names, pd.Series(df_ch["municipality_name"].values, index = ch_ids.values)])
    names = names[~names.index.duplicated()]

    return pd.Series(names.values + " (" + names.index + ")", index = names.index)


def home_communes(context):
    """person_id -> origin_id (home commune) of the synthetic population."""
    df_persons = context.stage("synthesis.population.sampled")[["person_id", "household_id"]]
    df_zones = context.stage("synthesis.population.spatial.home.zones")[["household_id", "commune_id"]]
    df_home = df_persons.merge(df_zones, on = "household_id").rename(columns = { "commune_id": "origin_id" })
    df_home["origin_id"] = df_home["origin_id"].astype(str)
    return df_home[["person_id", "origin_id"]]


def build_flows(context, df_commutes, origins, source):
    """df_commutes: one row per person with origin_id / destination_id.
    Returns one row per (origin_id, destination_id): mobpro, synthetic,
    synthetic_scaled, diff, diff_scaled, and the scale factor. MOBPRO is
    restricted to the given home communes."""
    df_synthetic = df_commutes.groupby(["origin_id", "destination_id"]).size().rename("synthetic").astype(float)

    df_mobpro, _ = context.stage("data.od.cleaned")
    df_mobpro = df_mobpro.astype({ "origin_id": str, "destination_id": str })
    df_mobpro = df_mobpro[df_mobpro["origin_id"].isin(origins)]
    df_mobpro = df_mobpro.groupby(["origin_id", "destination_id"])["weight"].sum().rename("mobpro")

    df = pd.concat([df_mobpro, df_synthetic], axis = 1).fillna(0.0)
    df = df[(df["mobpro"] > 0) | (df["synthetic"] > 0)].reset_index()

    scale = df["mobpro"].sum() / df["synthetic"].sum()
    df["synthetic_scaled"] = df["synthetic"] * scale
    df["diff"] = df["synthetic"] - df["mobpro"]
    df["diff_scaled"] = df["synthetic_scaled"] - df["mobpro"]

    print("Commute flows: %.0f MOBPRO workers, %d %s %s, scale factor %.3f" % (
        df["mobpro"].sum(), df["synthetic"].sum(), source["short"], source["unit"], scale))

    return df, scale


# ---------------------------------------------------------------- heatmap

def aggregate_for_heatmap(df, names):
    """Main origins (by MOBPRO residents commuting) x main destinations (by
    MOBPRO workers), every other commune folded into an "other" row/column."""
    top_origins = df.groupby("origin_id")["mobpro"].sum().nlargest(HEATMAP_TOP_ORIGINS).index
    top_destinations = df.groupby("destination_id")["mobpro"].sum().nlargest(HEATMAP_TOP_DESTINATIONS).index

    df = df.copy()
    df["row"] = df["origin_id"].map(names).fillna(df["origin_id"]).where(df["origin_id"].isin(top_origins), OTHER_ORIGINS)
    df["column"] = df["destination_id"].map(names).fillna(df["destination_id"])
    df.loc[~df["destination_id"].isin(top_destinations), "column"] = np.where(
        df.loc[~df["destination_id"].isin(top_destinations), "destination_id"].str.startswith(CH_PREFIX), OTHER_CH, OTHER_FR)

    rows = [names.get(i, i) for i in top_origins] + [OTHER_ORIGINS]
    columns = [names.get(i, i) for i in top_destinations] + [OTHER_FR, OTHER_CH]

    return {
        value: df.pivot_table(index = "row", columns = "column", values = value, aggfunc = "sum", fill_value = 0.0).reindex(index = rows, columns = columns, fill_value = 0.0)
        for value in ["mobpro", "synthetic", "diff", "diff_scaled"]
    }


def format_count(value, signed = False):
    sign = "+" if signed and value > 0 else ""
    if abs(value) >= 1e4:
        return "%s%.0fk" % (sign, value / 1e3)
    if abs(value) >= 1e3:
        return "%s%.1fk" % (sign, value / 1e3)
    return "%s%.0f" % (sign, value)


def draw_heatmap_panel(ax, matrix, title, norm, cmap, signed):
    image = ax.imshow(matrix.values, norm = norm, cmap = cmap, aspect = "auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation = 60, ha = "right", fontsize = 7)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize = 7)
    ax.set_xlabel("Work commune")
    ax.set_ylabel("Home commune")
    ax.set_title(title, fontsize = 10, fontweight = "bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            if value != 0:
                # white labels on the darkest cells of either colour scale
                intensity = abs(norm(value) - 0.5) * 2 if signed else norm(value)
                ax.text(j, i, format_count(value, signed), ha = "center", va = "center", fontsize = 5.5,
                    color = "white" if intensity > 0.7 else "#222")

    # separate the "other" row/columns from the named communes
    ax.axhline(matrix.shape[0] - 1.5, color = "#333", linewidth = 1)
    ax.axvline(matrix.shape[1] - 2.5, color = "#333", linewidth = 1)
    return image


def plot_heatmap(matrices, scale, source, path):
    fig, axes = plt.subplots(2, 2, figsize = (26, 22))

    flow_max = max(matrices["mobpro"].values.max(), matrices["synthetic"].values.max())
    flow_norm = LogNorm(vmin = 10, vmax = flow_max, clip = True)
    diff_max = max(np.abs(matrices["diff"].values).max(), np.abs(matrices["diff_scaled"].values).max())
    diff_norm = SymLogNorm(linthresh = 50, vmin = -diff_max, vmax = diff_max)

    panels = [
        ("mobpro", "1. MOBPRO: workers by usual workplace (all modes, incl. work from home)", flow_norm, "Blues", False),
        ("synthetic", "2. %s: %s" % (source["name"], source["description"]), flow_norm, "Blues", False),
        ("diff", "3. Difference: %s - MOBPRO (persons)" % source["short"], diff_norm, "RdBu", True),
        ("diff_scaled", "4. Difference after rescaling %s to the MOBPRO total (x%.3f) - MOBPRO" % (source["short"], scale), diff_norm, "RdBu", True),
    ]

    for ax, (key, title, norm, cmap, signed) in zip(axes.flatten(), panels):
        image = draw_heatmap_panel(ax, matrices[key], title, norm, cmap, signed)
        colorbar = fig.colorbar(image, ax = ax, shrink = 0.8)
        colorbar.set_label("%s too low  <-  persons  ->  %s too high" % (source["short"], source["short"]) if signed else "Persons (log scale)")

    fig.suptitle(
        "Commuting flows, %d main home communes x %d main work communes (ranked by MOBPRO)\n"
        "MOBPRO total %s workers, %s total %s %s, rescaling factor %.3f" % (
            HEATMAP_TOP_ORIGINS, HEATMAP_TOP_DESTINATIONS,
            format_count(matrices["mobpro"].values.sum()), source["short"],
            format_count(matrices["synthetic"].values.sum()), source["unit"], scale),
        fontsize = 14)
    fig.tight_layout(rect = (0, 0, 1, 0.96))
    fig.savefig(path, dpi = 120)
    plt.close(fig)


# ---------------------------------------------------------------- interactive map

def commune_geometries(context, commune_ids):
    df_fr = context.stage("data.spatial.municipalities")[["commune_id", "geometry"]].copy()
    df_fr["commune_id"] = df_fr["commune_id"].astype(str)

    df_ch = context.stage("data.spatial.ch.municipalities")
    df_ch = gpd.GeoDataFrame({
        "commune_id": CH_PREFIX + df_ch["municipality_id"].astype(int).astype(str).str.zfill(5),
        "geometry": df_ch.geometry.values }, crs = df_ch.crs).dissolve("commune_id").reset_index()

    df = pd.concat([
        df_fr.assign(geometry = df_fr.geometry.simplify(25)).to_crs("EPSG:4326"),
        df_ch.assign(geometry = df_ch.geometry.simplify(25)).to_crs("EPSG:4326"),
    ])
    df = gpd.GeoDataFrame(df[df["commune_id"].isin(commune_ids)], crs = "EPSG:4326")

    points = df.to_crs("EPSG:2056").geometry.representative_point().to_crs("EPSG:4326")
    df["x"], df["y"] = points.x.values, points.y.values
    return df


def diverging_limit(values):
    """Colour range for a difference view: 99th percentile of |difference|."""
    values = np.abs(np.asarray(values))
    values = values[values > 0]
    return max(10.0, float(np.quantile(values, 0.99))) if len(values) else 10.0


def arc_records(df, names, positions, scale, source):
    """One record per pair of distinct communes, with the value of every view
    (arc colours and widths are derived from them in the page, see
    inject_controls, to keep the file small)."""
    df = df[df["origin_id"] != df["destination_id"]].copy()
    df = df[df[["mobpro", "synthetic"]].max(axis = 1) >= 1]

    scales = {
        "flow_max": float(df[FLOW_VIEWS].values.max()),
        "diff_max": float(np.abs(df["diff"].values).max()),
        "diff_limit": diverging_limit(df["diff"].values),
    }

    records = []
    for row in df.itertuples(index = False):
        record = {
            "origin_id": row.origin_id, "destination_id": row.destination_id,
            "source": positions[row.origin_id], "target": positions[row.destination_id],
        }

        for key, _ in views(source):
            record[key] = round(float(getattr(row, key)), 1)

        record["_tooltip_html"] = "<br/>".join([
            "<b>%s &rarr; %s</b>" % (html.escape(names.get(row.origin_id, row.origin_id)), html.escape(names.get(row.destination_id, row.destination_id))),
            "MOBPRO workers: %.0f" % row.mobpro,
            "%s %s: %.0f" % (source["name"], source["unit"], row.synthetic),
            "Difference %s - MOBPRO: %+.0f" % (source["short"], row.diff),
        ])
        records.append(record)

    return pd.DataFrame(records), scales


def commune_records(df, df_geometry, names, source):
    """Polygons, filled with the intra-commune flow (home commune = work
    commune) of each view, or with the relative difference of all workers
    living in the commune; tooltip with intra, outgoing, incoming and
    resident totals."""
    is_intra = df["origin_id"] == df["destination_id"]
    columns = ["mobpro", "synthetic", "diff"]
    intra = df[is_intra].set_index("origin_id")[columns]
    outgoing = df[~is_intra].groupby("origin_id")[columns].sum()
    incoming = df[~is_intra].groupby("destination_id")[columns].sum()

    residents = df.groupby("origin_id")[columns].sum()
    residents["relative"] = residents["synthetic"] / residents["mobpro"] - 1.0
    resident_colormap = branca.colormap.LinearColormap(colors = DIFF_COLORMAP_COLORS, vmin = -RESIDENT_LIMIT, vmax = RESIDENT_LIMIT)

    intra_max = intra[FLOW_VIEWS].values.max()
    intra_limit = diverging_limit(intra["diff"].values)
    diff_colormap = branca.colormap.LinearColormap(colors = DIFF_COLORMAP_COLORS, vmin = -intra_limit, vmax = intra_limit)

    def line(label, table, commune_id):
        if commune_id not in table.index:
            return "%s: none" % label
        r = table.loc[commune_id]
        return "%s: MOBPRO %.0f | %s %.0f | diff %+.0f" % (
            label, r["mobpro"], source["short"], r["synthetic"], r["diff"])

    records = []
    for row in df_geometry.itertuples(index = False):
        commune_id = row.commune_id
        record = { "commune_id": commune_id }

        for key, _ in views(source):
            if commune_id not in intra.index:
                record[f"{key}_color"] = [244, 244, 244, 120]
            elif key in FLOW_VIEWS:
                value = intra.loc[commune_id, key]
                record[f"{key}_color"] = colormap_rgb(INTRA_COLORMAP, float(np.sqrt(value / intra_max))) + [190]
            else:
                value = intra.loc[commune_id, key]
                record[f"{key}_color"] = colormap_rgb(diff_colormap, float(np.clip(value, -intra_limit, intra_limit))) + [190]

        if commune_id not in residents.index or residents.loc[commune_id, "mobpro"] < RESIDENT_MINIMUM:
            record["resident_relative_color"] = [244, 244, 244, 120]
        else:
            value = float(np.clip(residents.loc[commune_id, "relative"], -RESIDENT_LIMIT, RESIDENT_LIMIT))
            record["resident_relative_color"] = colormap_rgb(resident_colormap, value) + [190]

        resident_line = "All workers living here: none"
        if commune_id in residents.index:
            r = residents.loc[commune_id]
            resident_line = "All workers living here: MOBPRO %.0f | %s %.0f | diff %+.0f (%+.1f%%)" % (
                r["mobpro"], source["short"], r["synthetic"], r["diff"], 100 * r["relative"])

        record["_tooltip_html"] = "<br/>".join([
            "<b>%s</b>" % html.escape(names.get(commune_id, commune_id)),
            resident_line,
            line("Living and working here", intra, commune_id),
            line("Living here, working elsewhere", outgoing, commune_id),
            line("Working here, living elsewhere", incoming, commune_id),
            "<i>Click to show only this commune's flows</i>",
        ])

        geometry = row.geometry
        parts = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in parts:
            rings = [part.exterior.coords] + [ring.coords for ring in part.interiors]
            records.append({ "polygon": [[[round(x, 5), round(y, 5)] for x, y, *_ in ring] for ring in rings], **record })

    return pd.DataFrame(records), intra_limit, intra_max


def inject_controls(path, view_info, fill_info, arc_style, default_view, summary, source):
    """Adds the control panel (view, commune fill, minimum flow, selected
    commune) and the script re-filtering / re-colouring the "flows" and
    "communes" layers."""
    options_html = "\n".join(
        '<option value="%s"%s>%s</option>' % (key, " selected" if key == default_view else "", html.escape(label))
        for key, label in views(source))
    fill_options_html = "\n".join(
        '<option value="%s">%s</option>' % (key, html.escape(info["label"])) for key, info in fill_info.items())

    panel = """
<div style="position:absolute;z-index:20;top:12px;left:12px;width:360px;padding:10px 12px;border-radius:7px;
            background:rgba(255,255,255,0.96);box-shadow:0 2px 10px rgba(0,0,0,0.28);color:#222;font:12px/1.35 Arial, sans-serif;">
  <div style="font-weight:700;font-size:13px;margin-bottom:6px;">Commuting flows (home commune &rarr; work commune)</div>
  <div style="color:#555;margin-bottom:8px;">%s</div>
  <label style="display:block;margin-bottom:6px;">Show<br/><select id="cf-view" style="width:100%%;">%s</select></label>
  <label style="display:block;margin-bottom:6px;">Hide arcs below (persons, absolute value) <input id="cf-threshold" type="number" min="0" step="10" value="%d" style="width:70px;"/></label>
  <div style="margin-bottom:6px;">Selected commune: <b id="cf-selected">none (click a commune)</b><br/>
    <label><input type="radio" name="cf-direction" value="both" checked/> from and to</label>
    <label><input type="radio" name="cf-direction" value="from"/> from (residents)</label>
    <label><input type="radio" name="cf-direction" value="to"/> to (workers)</label>
    <button id="cf-clear" style="margin-left:6px;">Clear</button></div>
  <div id="cf-count" style="color:#555;margin-bottom:8px;"></div>
  <div style="font-weight:700;">Arcs (between communes)</div>
  <div id="cf-arc-legend" style="margin-bottom:6px;"></div>
  <label style="display:block;font-weight:700;">Commune fill<br/><select id="cf-fill" style="width:100%%;font-weight:400;">%s</select></label>
  <div id="cf-fill-note" style="color:#555;"></div>
  <div id="cf-bar" style="width:220px;height:12px;border:1px solid #777;margin-top:3px;"></div>
  <div style="display:flex;justify-content:space-between;width:220px;margin-top:2px;"><span id="cf-low"></span><span id="cf-high"></span></div>
</div>
""" % (summary, options_html, DEFAULT_THRESHOLD, fill_options_html)

    script = """
<script>
(function () {
  const info = %s;
  const fills = %s;
  const style = %s;
  const state = { view: "%s", threshold: %d, selected: null, selectedName: null, direction: "both", fill: "intra" };
  let base = null;

  // Flow views: fixed orange (home) -> blue (work) arcs, width ~ sqrt(persons).
  // Difference views: diverging colour clipped at +/- diff_limit, width ~ sqrt(|difference|).
  function divergingColor(value) {
    const t = Math.max(-1, Math.min(1, value / style.diff_limit));
    const position = (t + 1) / 2 * (style.diff_stops.length - 1);
    const i = Math.min(Math.floor(position), style.diff_stops.length - 2), f = position - i;
    const a = style.diff_stops[i], b = style.diff_stops[i + 1];
    return [0, 1, 2].map(function (c) { return Math.round(a[c] + (b[c] - a[c]) * f); }).concat([220]);
  }
  function isFlow(key) { return style.flow_views.indexOf(key) >= 0; }
  function arcWidth(d, key) {
    return Math.max(1, style.max_width * Math.sqrt(Math.abs(d[key]) / (isFlow(key) ? style.flow_max : style.diff_max)));
  }
  function arcColor(d, key, end) {
    return isFlow(key) ? (end === "source" ? style.source_color : style.target_color) : divergingColor(d[key]);
  }

  function render() {
    if (typeof deckInstance === "undefined" || !deckInstance.props || !deckInstance.props.layers || !deckInstance.props.layers.length) {
      setTimeout(render, 200); return;
    }
    if (base === null) {
      base = {};
      deckInstance.props.layers.forEach(function (layer) { base[layer.id] = layer.props.data; });
      deckInstance.setProps({ onClick: function (picked) {
        if (picked && picked.object && picked.layer && picked.layer.id === "communes") {
          state.selected = picked.object.commune_id;
          state.selectedName = picked.object._tooltip_html.split("<br/>")[0].replace(/<[^>]+>/g, "");
          render();
        }
      } });
    }

    const key = state.view, o = info[key];
    const fillKey = state.fill === "intra" ? key + "_color" : state.fill + "_color";
    const legend = state.fill === "intra" ? o : fills[state.fill];
    const arcs = base["flows"].filter(function (d) {
      if (Math.abs(d[key]) < state.threshold) { return false; }
      if (state.selected === null) { return true; }
      return (state.direction !== "to" && d.origin_id === state.selected) || (state.direction !== "from" && d.destination_id === state.selected);
    });

    deckInstance.setProps({ layers: deckInstance.props.layers.map(function (layer) {
      if (layer.id === "flows") {
        return layer.clone({
          data: arcs,
          getSourceColor: function (d) { return arcColor(d, key, "source"); },
          getTargetColor: function (d) { return arcColor(d, key, "target"); },
          getWidth: function (d) { return arcWidth(d, key); },
          updateTriggers: { getSourceColor: key, getTargetColor: key, getWidth: key },
        });
      }
      if (layer.id === "communes") {
        return layer.clone({
          getFillColor: function (d) { return d[fillKey]; },
          getLineColor: function (d) { return d.commune_id === state.selected ? [0, 0, 0, 255] : [110, 110, 110, 140]; },
          getLineWidth: function (d) { return d.commune_id === state.selected ? 4 : 1; },
          updateTriggers: { getFillColor: fillKey, getLineColor: state.selected, getLineWidth: state.selected },
        });
      }
      return layer;
    }) });

    document.getElementById("cf-arc-legend").innerHTML = o.arc_legend;
    document.getElementById("cf-fill-note").textContent = fills[state.fill].note;
    document.getElementById("cf-bar").style.background = "linear-gradient(to right, " + legend.stops + ")";
    document.getElementById("cf-low").textContent = legend.low;
    document.getElementById("cf-high").textContent = legend.high;
    document.getElementById("cf-selected").textContent = state.selected === null ? "none (click a commune)" : state.selectedName;
    document.getElementById("cf-count").textContent = arcs.length + " of " + base["flows"].length + " commune pairs shown";
  }

  document.getElementById("cf-view").addEventListener("change", function (e) { state.view = e.target.value; render(); });
  document.getElementById("cf-fill").addEventListener("change", function (e) { state.fill = e.target.value; render(); });
  document.getElementById("cf-threshold").addEventListener("change", function (e) { state.threshold = Math.max(0, Number(e.target.value) || 0); render(); });
  document.getElementById("cf-clear").addEventListener("click", function () { state.selected = null; state.selectedName = null; render(); });
  document.querySelectorAll("input[name=cf-direction]").forEach(function (input) {
    input.addEventListener("change", function (e) { state.direction = e.target.value; render(); });
  });
  render();
})();
</script>
""" % (json.dumps(view_info), json.dumps(fill_info), json.dumps(arc_style), default_view, DEFAULT_THRESHOLD)

    with open(path, encoding = "utf-8") as f:
        page = f.read()
    page = page.replace("</body>", panel + "\n</body>", 1).replace("</html>", script + "\n</html>", 1)
    with open(path, "w", encoding = "utf-8") as f:
        f.write(page)


def plot_flow_map(context, df, names, scale, source, path):
    commune_ids = set(df["origin_id"]) | set(df["destination_id"])
    df_geometry = commune_geometries(context, commune_ids)
    positions = { row.commune_id: [round(row.x, 5), round(row.y, 5)] for row in df_geometry.itertuples(index = False) }

    df_known = df[df["origin_id"].isin(positions) & df["destination_id"].isin(positions)]
    missing = df.loc[~df.index.isin(df_known.index), ["mobpro", "synthetic"]].sum()
    if missing.sum() > 0:
        print("Commute flow map: no geometry for some communes, %.0f MOBPRO / %.0f %s not drawn" % (missing["mobpro"], missing["synthetic"], source["short"]))

    df_arcs, arc_scales = arc_records(df_known, names, positions, scale, source)
    df_communes, intra_limit, intra_max = commune_records(df, df_geometry, names, source)

    arc_style = dict(arc_scales,
        flow_views = FLOW_VIEWS, max_width = MAX_ARC_WIDTH,
        source_color = FLOW_SOURCE_COLOR + [200], target_color = FLOW_TARGET_COLOR + [200],
        diff_stops = [[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in DIFF_COLORMAP_COLORS])

    flow_legend = ("Width &prop; &radic;persons (widest = %s). Orange end = home commune, blue end = work commune." % format_count(arc_scales["flow_max"]))
    diff_legend = ("Red = %s too low, blue = %s too high (full colour at &plusmn;%s persons). Width &prop; &radic;|difference|." % (
        source["short"], source["short"], format_count(arc_scales["diff_limit"])))
    intra_stops = colormap_css_stops(INTRA_COLORMAP, 0, 1)
    diff_stops = ", ".join(DIFF_COLORMAP_COLORS)

    view_info = {}
    for key, _ in views(source):
        if key in FLOW_VIEWS:
            view_info[key] = { "arc_legend": flow_legend, "stops": intra_stops, "low": "0", "high": "%s persons (sqrt scale)" % format_count(intra_max) }
        else:
            view_info[key] = { "arc_legend": diff_legend, "stops": diff_stops, "low": "-%s" % format_count(intra_limit), "high": "+%s" % format_count(intra_limit) }

    resident_note = "Grey: fewer than %d MOBPRO workers. Red = %s too low, blue = too high." % (RESIDENT_MINIMUM, source["short"])
    fill_info = {
        "intra": { "label": "Living and working in the same commune (selected view)",
                   "note": "Same quantity as the selected view, for flows within the commune." },
        "resident_relative": { "label": "All workers living in the commune: (%s - MOBPRO) / MOBPRO" % source["short"],
                               "note": resident_note, "stops": diff_stops,
                               "low": "-%.0f%%" % (100 * RESIDENT_LIMIT), "high": "+%.0f%%" % (100 * RESIDENT_LIMIT) },
    }

    summary = ("MOBPRO: %s workers by usual workplace (all modes, incl. work from home). %s: %s %s. "
               "Hover for numbers.") % (
        format_count(df["mobpro"].sum()), html.escape(source["name"]), format_count(df["synthetic"].sum()),
        html.escape(source["description"]))

    layers = [
        pdk.Layer(
            "PolygonLayer", df_communes, id = "communes",
            pickable = True, auto_highlight = True, stroked = True, filled = True,
            get_polygon = "polygon", get_fill_color = "mobpro_color",
            get_line_color = [110, 110, 110, 140], get_line_width = 1, line_width_min_pixels = 1,
            highlight_color = [0, 200, 255, 120],
        ),
        pdk.Layer(
            "ArcLayer", df_arcs, id = "flows",
            pickable = True, auto_highlight = True,
            get_source_position = "source", get_target_position = "target",
            get_source_color = FLOW_SOURCE_COLOR, get_target_color = FLOW_TARGET_COLOR,
            get_width = 1, get_height = 0.5,
        ),
    ]

    minx, miny, maxx, maxy = df_geometry[df_geometry["commune_id"].isin(set(df["origin_id"]))].total_bounds
    view_state = compute_view([[minx, miny], [maxx, maxy]])
    view_state.pitch = 30

    deck = pdk.Deck(
        layers = layers,
        initial_view_state = view_state,
        tooltip = { "html": "{_tooltip_html}", "style": { "color": "white", "background-color": "black", "padding": "10px", "max-width": "520px" } },
        map_style = BASEMAP_STYLE,
    )
    deck.to_html(path, notebook_display = False)
    inject_controls(path, view_info, fill_info, arc_style, "mobpro", summary, source)


def write_outputs(context, df, scale, source, file_stem):
    """CSV, heatmap and interactive map, named <prefix><file_stem>[_heatmap.png|_map.html|.csv]."""
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")

    analysis_output_path = os.path.join(output_path, ANALYSIS_FOLDER)
    if not os.path.exists(analysis_output_path):
        os.mkdir(analysis_output_path)

    names = commune_names(context)

    df_csv = df.copy()
    df_csv.insert(1, "origin_name", df_csv["origin_id"].map(names))
    df_csv.insert(3, "destination_name", df_csv["destination_id"].map(names))
    df_csv = df_csv.rename(columns = { "synthetic": source["column"], "synthetic_scaled": source["column"] + "_scaled" })
    df_csv.sort_values("mobpro", ascending = False).to_csv(
        f"{analysis_output_path}/{output_prefix}{file_stem}.csv", sep = ";", index = None, float_format = "%.1f")

    plot_heatmap(aggregate_for_heatmap(df, names), scale, source, f"{analysis_output_path}/{output_prefix}{file_stem}_heatmap.png")
    plot_flow_map(context, df, names, scale, source, f"{analysis_output_path}/{output_prefix}{file_stem}_map.html")

    print("Commute flow comparison written to %s" % analysis_output_path)


def execute(context):
    df_home = home_communes(context)

    df_work, _, _ = context.stage("synthesis.population.spatial.primary.locations")
    df_work = df_work[["person_id", "commune_id"]].rename(columns = { "commune_id": "destination_id" })
    df_work["destination_id"] = df_work["destination_id"].astype(str)

    df, scale = build_flows(context, df_home.merge(df_work, on = "person_id"), set(df_home["origin_id"]), SYNTHETIC_SOURCE)
    write_outputs(context, df, scale, SYNTHETIC_SOURCE, "debug_commute_flows")
    return df
