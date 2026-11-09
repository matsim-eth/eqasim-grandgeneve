import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import branca

"""
Compares, per EDGT sector of residence (DTIR, "secteur de tirage", i.e. a
group of a few municipalities, or of IRIS within the largest ones), the share
of people who perform no mandatory activity (work/education) in Switzerland
but at least one secondary activity (shop, leisure, other) there, between the
EDGT74 survey (weighted) and the synthetic population (after location
assignment, so Swiss secondary locations are the ones actually chosen by
synthesis.population.spatial.secondary.locations).

Shares are computed over all residents of a sector, including people with no
trip at all. Cross-perimeter persons are excluded on both sides: the
synthetic population replaces their trip chain by a single "cross_perimeter"
activity (see synthesis.population.trips), so keeping them would bias the
synthetic share downwards.
"""

ANALYSIS_FOLDER = "analysis_cross_border_commuters"

MANDATORY_PURPOSES = ["work", "education"]
NON_SECONDARY_PURPOSES = ["home", "cross_perimeter"] + MANDATORY_PURPOSES

# The two surveys' own zoning; the "external" zones are coarse catch-all
# zones used to resolve trip destinations, not real home samples.
INTERNAL_ZONE_SOURCES = ["annemasse_zf", "annecy_zf"]

SHARE_COLORMAP = branca.colormap.LinearColormap(
    colors = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    vmin = 0, vmax = 30
)
DIFFERENCE_COLORMAP = branca.colormap.LinearColormap(
    colors = ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"],
    vmin = -10, vmax = 10
)


def configure(context):
    context.config("output_path")
    context.config("output_prefix", "ile_de_france_")

    context.stage("data.hts.edgt_74.adisp_merge.zones")
    context.stage("data.hts.edgt_74.adisp_merge.merge")

    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.locations")


def build_sectors(df_zones):
    """One polygon per DTIR of the surveys' own zoning."""
    df_sectors = df_zones[df_zones["source"].isin(INTERNAL_ZONE_SOURCES)].copy()
    df_sectors["geometry"] = df_sectors.geometry.make_valid()
    df_sectors = df_sectors.dissolve(by = "dtir_id", as_index = False, aggfunc = {
        "dtir_name": "first", "d30_name": "first", "source": "first",
        "commune_id": "nunique",
    }).rename(columns = { "commune_id": "nb_communes" })

    return df_sectors[["dtir_id", "dtir_name", "d30_name", "source", "nb_communes", "geometry"]]


def build_swiss_sectors(df_zones):
    """One polygon per Swiss DTIR of the surveys' external zoning (Geneva
    city, three parts of the canton of Geneva, Nyon, Lausanne-Morges, rest of
    Vaud, Valais, rest of Switzerland): the destination zones of the flows."""
    df_sectors = df_zones[df_zones["d10_name"] == "Suisse"].copy()
    df_sectors["geometry"] = df_sectors.geometry.make_valid()
    df_sectors = df_sectors.dissolve(by = "dtir_id", as_index = False, aggfunc = { "dtir_name": "first" })

    return df_sectors[["dtir_id", "dtir_name", "geometry"]]


def classify_persons(df_activities):
    """Per person (from a person_id, purpose, in_switzerland table of
    out-of-home activities): whether they have a mandatory activity in
    Switzerland, a secondary one, and the indicator itself (secondary but
    no mandatory activity in Switzerland)."""
    df = df_activities[df_activities["in_switzerland"]]
    is_mandatory = df["purpose"].isin(MANDATORY_PURPOSES)
    is_secondary = ~df["purpose"].isin(NON_SECONDARY_PURPOSES)

    mandatory_persons = set(df.loc[is_mandatory, "person_id"])
    secondary_persons = set(df.loc[is_secondary, "person_id"])

    return mandatory_persons, secondary_persons - mandatory_persons


def swiss_secondary_activities(df_activities, secondary_only_persons):
    """Swiss secondary activities (person_id, ch_dtir_id) of the persons
    counted in the indicator."""
    f = df_activities["in_switzerland"] & ~df_activities["purpose"].isin(NON_SECONDARY_PURPOSES)
    f &= df_activities["person_id"].isin(secondary_only_persons)
    return df_activities.loc[f, ["person_id", "ch_dtir_id"]]


def hts_persons(df_zones, df_persons, df_trips):
    """EDGT74 persons with their residence DTIR, weight and indicators, and
    the Swiss secondary activities of those counted in the indicator."""
    swiss_zone_to_dtir = df_zones[df_zones["d10_name"] == "Suisse"].set_index("zone_id")["dtir_id"]

    df_activities = pd.DataFrame({
        "person_id": df_trips["person_id"],
        "purpose": df_trips["following_purpose"].astype(str),
        "ch_dtir_id": df_trips["destination_zone_id"].map(swiss_zone_to_dtir),
    })
    df_activities["in_switzerland"] = df_activities["ch_dtir_id"].notna()
    mandatory_persons, secondary_only_persons = classify_persons(df_activities)

    zone_to_dtir = df_zones[df_zones["source"].isin(INTERNAL_ZONE_SOURCES)].set_index("zone_id")["dtir_id"]

    df = df_persons[~df_persons["is_crossperim_person"]].copy()
    df = pd.DataFrame({
        "person_id": df["person_id"].values,
        "dtir_id": df["residence_zone_id"].map(zone_to_dtir).values,
        "weight": df["person_weight"].values,
    })
    df["mandatory_ch"] = df["person_id"].isin(mandatory_persons)
    df["secondary_only_ch"] = df["person_id"].isin(secondary_only_persons)

    print("EDGT74: %d / %d non-cross-perimeter persons resolved to a residence DTIR" % (df["dtir_id"].notna().sum(), len(df)))
    return df.dropna(subset = ["dtir_id"]), swiss_secondary_activities(df_activities, secondary_only_persons)


def synthetic_persons(df_sectors, df_swiss_sectors, df_enriched, df_home, df_activities, df_locations):
    """Synthetic persons with their residence DTIR (home location spatially
    joined to the DTIR polygons) and indicators; weight 1. Also the Swiss
    secondary activities of those counted in the indicator, their location
    spatially joined to the Swiss DTIR polygons."""
    df_activities = pd.merge(
        df_activities[["person_id", "activity_index", "purpose"]],
        df_locations[["person_id", "activity_index", "commune_id", "geometry"]],
        on = ["person_id", "activity_index"], how = "left"
    )
    df_activities["purpose"] = df_activities["purpose"].astype(str)
    df_activities["in_switzerland"] = df_activities["commune_id"].astype(str).str.startswith("CH")
    mandatory_persons, secondary_only_persons = classify_persons(df_activities)

    # Swiss DTIR of the Swiss activities only (the nearest one for the few
    # falling just outside every polygon, e.g. on the lake shore)
    df_swiss = gpd.GeoDataFrame(df_activities[df_activities["in_switzerland"]], crs = df_locations.crs).to_crs(df_swiss_sectors.crs)
    df_swiss = gpd.sjoin_nearest(df_swiss, df_swiss_sectors[["dtir_id", "geometry"]], how = "left")
    df_swiss = df_swiss[~df_swiss.index.duplicated()]
    df_activities["ch_dtir_id"] = df_swiss["dtir_id"].reindex(df_activities.index)

    df_home = gpd.GeoDataFrame(df_home[["household_id", "geometry"]], crs = df_home.crs).to_crs(df_sectors.crs)
    df_home = gpd.sjoin(df_home, df_sectors[["dtir_id", "geometry"]], how = "left", predicate = "within")
    df_home = df_home.drop_duplicates("household_id")
    household_to_dtir = df_home.set_index("household_id")["dtir_id"]

    df = df_enriched
    if "is_crossperim_person" in df:
        df = df[~df["is_crossperim_person"].fillna(False).astype(bool)]

    df = pd.DataFrame({
        "person_id": df["person_id"].values,
        "dtir_id": df["household_id"].map(household_to_dtir).values,
        "weight": 1.0,
    })
    df["mandatory_ch"] = df["person_id"].isin(mandatory_persons)
    df["secondary_only_ch"] = df["person_id"].isin(secondary_only_persons)

    print("Synthetic: %d / %d non-cross-perimeter persons living within an EDGT74 DTIR" % (df["dtir_id"].notna().sum(), len(df)))
    return df.dropna(subset = ["dtir_id"]), swiss_secondary_activities(df_activities, secondary_only_persons)


def build_flows(df_persons, df_swiss_activities, prefix):
    """Weighted Swiss secondary activities per (residence DTIR, Swiss DTIR),
    and their share (%) among all Swiss secondary activities of the
    residence DTIR."""
    df = pd.merge(df_swiss_activities, df_persons[["person_id", "dtir_id", "weight"]], on = "person_id")
    df = df.groupby(["dtir_id", "ch_dtir_id"])["weight"].sum().rename(f"{prefix}_activities").reset_index()
    df[f"{prefix}_share"] = 100 * df[f"{prefix}_activities"] / df.groupby("dtir_id")[f"{prefix}_activities"].transform("sum")
    return df


def build_flow_comparison(df_hts, df_hts_swiss, df_synthetic, df_synthetic_swiss, df_sectors, df_swiss_sectors):
    df = pd.merge(
        build_flows(df_hts, df_hts_swiss, "hts"),
        build_flows(df_synthetic, df_synthetic_swiss, "synthetic"),
        on = ["dtir_id", "ch_dtir_id"], how = "outer"
    ).fillna(0.0)

    df["dtir_id"] = df["dtir_id"].astype(str)
    df["dtir_name"] = df["dtir_id"].map(df_sectors.assign(dtir_id = df_sectors["dtir_id"].astype(str)).set_index("dtir_id")["dtir_name"])
    df["ch_dtir_name"] = df["ch_dtir_id"].map(df_swiss_sectors.set_index("dtir_id")["dtir_name"])

    for column in ["hts_activities", "synthetic_activities"]:
        df[column] = df[column].round()
    for column in ["hts_share", "synthetic_share"]:
        df[column] = df[column].round(2)

    return df[[
        "dtir_id", "dtir_name", "ch_dtir_id", "ch_dtir_name",
        "hts_activities", "hts_share", "synthetic_activities", "synthetic_share",
    ]].sort_values(["dtir_id", "ch_dtir_id"])


def aggregate(df, prefix):
    """Weighted shares (%) per DTIR, plus a 95% confidence half-width for
    the indicator based on Kish's effective sample size (meaningful for the
    survey; very small for the synthetic population)."""
    df = df.assign(
        w_mandatory = df["weight"] * df["mandatory_ch"],
        w_secondary = df["weight"] * df["secondary_only_ch"],
        w2 = df["weight"] ** 2,
    )
    grouped = df.groupby("dtir_id").agg(
        nb_persons = ("person_id", "size"), weight = ("weight", "sum"), w2 = ("w2", "sum"),
        w_mandatory = ("w_mandatory", "sum"), w_secondary = ("w_secondary", "sum"),
    )

    share = grouped["w_secondary"] / grouped["weight"]
    effective_n = grouped["weight"] ** 2 / grouped["w2"]

    result = pd.DataFrame({
        "nb_persons": grouped["nb_persons"],
        "weighted_persons": grouped["weight"].round(),
        "mandatory_ch_share": (100 * grouped["w_mandatory"] / grouped["weight"]).round(2),
        "secondary_only_ch_share": (100 * share).round(2),
        "secondary_only_ch_ci95": (100 * 1.96 * np.sqrt(share * (1 - share) / effective_n)).round(2),
    })
    result.columns = [f"{prefix}_{c}" for c in result.columns]

    return result.reset_index()


def build_comparison(df_sectors, df_hts, df_synthetic):
    df = df_sectors.drop(columns = "geometry").copy()
    df["dtir_id"] = df["dtir_id"].astype(str)

    total = []
    for prefix, df_source in (("hts", df_hts), ("synthetic", df_synthetic)):
        df_source = df_source.assign(dtir_id = df_source["dtir_id"].astype(str))
        df = df.merge(aggregate(df_source, prefix), on = "dtir_id", how = "left")
        total.append(aggregate(df_source.assign(dtir_id = "all"), prefix).drop(columns = "dtir_id"))

    df["difference"] = (df["synthetic_secondary_only_ch_share"] - df["hts_secondary_only_ch_share"]).round(2)
    df = df.sort_values("dtir_id")

    df_total = pd.concat(total, axis = 1)
    df_total["dtir_id"] = "all"
    df_total["dtir_name"] = "Entire EDGT74 perimeter"
    df_total["difference"] = (df_total["synthetic_secondary_only_ch_share"] - df_total["hts_secondary_only_ch_share"]).round(2)

    return pd.concat([df, df_total], ignore_index = True)


def plot_comparison(df_comparison, path):
    df = df_comparison[df_comparison["dtir_id"] != "all"].dropna(subset = ["hts_secondary_only_ch_share", "synthetic_secondary_only_ch_share"])
    df_total = df_comparison[df_comparison["dtir_id"] == "all"].iloc[0]

    fig, ax = plt.subplots(figsize = (6, 6))

    ax.errorbar(
        df["hts_secondary_only_ch_share"], df["synthetic_secondary_only_ch_share"],
        xerr = df["hts_secondary_only_ch_ci95"], fmt = "none", ecolor = "#bbbbbb", elinewidth = 0.8, zorder = 1
    )
    ax.scatter(
        df["hts_secondary_only_ch_share"], df["synthetic_secondary_only_ch_share"],
        s = 10 + 0.15 * df["hts_nb_persons"], color = "#2171b5", alpha = 0.7, zorder = 2, label = "DTIR (size = EDGT sample)"
    )
    ax.scatter(
        [df_total["hts_secondary_only_ch_share"]], [df_total["synthetic_secondary_only_ch_share"]],
        marker = "*", s = 200, color = "#b2182b", zorder = 3, label = "Entire perimeter"
    )

    upper = 1.05 * max(df["hts_secondary_only_ch_share"].max(), df["synthetic_secondary_only_ch_share"].max(), 1.0)
    ax.plot([0, upper], [0, upper], color = "#555555", linestyle = "--", linewidth = 0.8)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)

    ax.set_xlabel("EDGT74 (weighted, %)")
    ax.set_ylabel("Synthetic population (%)")
    ax.set_title("Secondary activity in Switzerland without\nwork/education there, by residence DTIR", fontsize = 10)
    ax.grid(alpha = 0.3)
    ax.legend(loc = "upper left", fontsize = 8)

    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)


FLOW_COLORS = { "hts": "#d95f02", "synthetic": "#1b9e77" }


class FlowArrows(branca.element.MacroElement):
    """Clicking a French DTIR draws curved arrows from it to the Swiss DTIRs
    where its residents' Swiss secondary activities take place, one set per
    source (bent to opposite sides so that both stay readable), with a width
    proportional to the share of the origin's Swiss secondary activities;
    a panel lists the shares. Clicking elsewhere clears them. Arrows live in
    their own pane so that switching the choropleth layer doesn't hide them,
    and are redrawn on zoom so arrowheads keep a constant on-screen size. The
    pane uses its own SVG renderer and lets clicks through: with the map's
    canvas renderer, a full-map canvas would sit above the sectors and
    swallow every click after the first one."""

    _template = branca.element.Template("""
        {% macro html(this, kwargs) %}
        <style>
            .flow-panel { background: white; padding: 8px 10px; font: 12px sans-serif; box-shadow: 0 0 6px rgba(0,0,0,0.3); border-radius: 4px; max-width: 360px; }
            .flow-panel table { border-collapse: collapse; }
            .flow-panel td, .flow-panel th { padding: 1px 6px; text-align: right; }
            .flow-panel td:first-child, .flow-panel th:first-child { text-align: left; }
        </style>
        {% endmacro %}

        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            var layers = [{% for layer in this.layers %}{{ layer.get_name() }}{% if not loop.last %}, {% endif %}{% endfor %}];
            var data = {{ this.data }};
            var colors = {{ this.colors }};
            var sources = [["hts", "EDGT74", 0.2], ["synthetic", "Synthetic", -0.2]];

            map.createPane("flows");
            map.getPane("flows").style.zIndex = 450;
            map.getPane("flows").style.pointerEvents = "none";
            var renderer = L.svg({ pane: "flows" });
            var group = L.layerGroup().addTo(map);
            var current = null;

            var panel = L.control({ position: "bottomleft" });
            panel.onAdd = function() {
                this._div = L.DomUtil.create("div", "flow-panel");
                L.DomEvent.disableClickPropagation(this._div);
                return this._div;
            };
            panel.update = function(origin) {
                if (origin === null) {
                    this._div.innerHTML = "<b>Click on a French sector</b> to show where its residents' secondary activities in Switzerland take place " +
                        "(persons without work/education there).<br>" +
                        "<span style='color:" + colors.hts + "'>&#9632;</span> EDGT74 &nbsp; " +
                        "<span style='color:" + colors.synthetic + "'>&#9632;</span> Synthetic";
                    return;
                }
                var flows = data.flows[origin] || [];
                var html = "<b>" + data.origins[origin].name + "</b><br>Share of Swiss secondary activities (%)<table>" +
                    "<tr><th>Destination</th><th style='color:" + colors.hts + "'>EDGT74</th><th style='color:" + colors.synthetic + "'>Synthetic</th></tr>";
                flows.forEach(function(f) {
                    html += "<tr><td>" + data.destinations[f.destination].name + "</td><td>" + f.hts_share.toFixed(1) + "</td><td>" + f.synthetic_share.toFixed(1) + "</td></tr>";
                });
                if (flows.length === 0) {
                    html += "<tr><td colspan='3'>No secondary activity in Switzerland</td></tr>";
                }
                this._div.innerHTML = html + "</table>";
            };
            panel.addTo(map);
            panel.update(null);

            function curve(a, b, bend, zoom) {
                var p0 = map.project(a, zoom), p2 = map.project(b, zoom);
                var dx = p2.x - p0.x, dy = p2.y - p0.y;
                var cx = (p0.x + p2.x) / 2 - dy * bend, cy = (p0.y + p2.y) / 2 + dx * bend;
                var points = [];
                for (var i = 0; i <= 32; i++) {
                    var t = i / 32;
                    points.push(L.point(
                        (1 - t) * (1 - t) * p0.x + 2 * (1 - t) * t * cx + t * t * p2.x,
                        (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * cy + t * t * p2.y
                    ));
                }
                return points;
            }

            function draw(origin) {
                group.clearLayers();
                current = origin;
                panel.update(origin);
                if (origin === null) return;

                var zoom = map.getZoom();
                var o = data.origins[origin];
                L.circleMarker(o.location, { pane: "flows", renderer: renderer, interactive: false, radius: 5, color: "#222", weight: 2, fillColor: "#fff", fillOpacity: 1 }).addTo(group);

                (data.flows[origin] || []).forEach(function(f) {
                    var d = data.destinations[f.destination];
                    sources.forEach(function(s) {
                        var share = f[s[0] + "_share"];
                        if (!(share > 0)) return;

                        var width = 1.5 + 0.14 * share;
                        var size = 7 + 1.5 * width;
                        var points = curve(o.location, d.location, s[2], zoom);

                        // Stop the line short of the tip so it doesn't poke through the arrowhead
                        var tip = points[points.length - 1], previous = points[points.length - 3];
                        var ux = tip.x - previous.x, uy = tip.y - previous.y, norm = Math.sqrt(ux * ux + uy * uy) || 1;
                        ux /= norm; uy /= norm;
                        var back = L.point(tip.x - size * ux, tip.y - size * uy);
                        var line = points.slice(0, -2).concat([back]);

                        var label = s[1] + ": " + share.toFixed(1) + "% of Swiss secondary activities<br>" + o.name + " &rarr; " + d.name;
                        var style = { pane: "flows", renderer: renderer, color: colors[s[0]], opacity: 0.85 };

                        L.polyline(line.map(function(p) { return map.unproject(p, zoom); }), Object.assign({ weight: width, lineCap: "butt" }, style))
                            .bindTooltip(label, { sticky: true }).addTo(group);
                        L.polygon([
                            tip,
                            L.point(back.x - 0.6 * size * uy, back.y + 0.6 * size * ux),
                            L.point(back.x + 0.6 * size * uy, back.y - 0.6 * size * ux)
                        ].map(function(p) { return map.unproject(p, zoom); }), Object.assign({ weight: 1, fillColor: colors[s[0]], fillOpacity: 0.85 }, style))
                            .bindTooltip(label, { sticky: true }).addTo(group);
                    });
                });
            }

            layers.forEach(function(layer) {
                layer.on("click", function(e) {
                    L.DomEvent.stopPropagation(e);
                    draw(e.layer.feature.properties.dtir_id);
                });
            });
            map.on("click", function() { draw(null); });
            map.on("zoomend", function() { if (current !== null) draw(current); });
        })();
        {% endmacro %}
    """)

    def __init__(self, layers, data):
        super().__init__()
        self._name = "FlowArrows"
        self.layers = layers
        self.data = json.dumps(data)
        self.colors = json.dumps(FLOW_COLORS)


def build_flow_data(df_sectors, df_swiss_sectors, df_flows):
    """JSON-ready origins/destinations (name, representative point, lat/lon)
    and flows per origin, largest (over both sources) first."""
    def points(df):
        df = df.to_crs("EPSG:4326")
        locations = df.geometry.representative_point()
        return {
            str(dtir_id): { "name": name, "location": [point.y, point.x] }
            for dtir_id, name, point in zip(df["dtir_id"], df["dtir_name"], locations)
        }

    df_flows = df_flows.assign(order = df_flows[["hts_share", "synthetic_share"]].max(axis = 1))
    df_flows = df_flows.sort_values(["dtir_id", "order"], ascending = [True, False])

    flows = {
        str(dtir_id): [
            { "destination": str(row.ch_dtir_id), "hts_share": float(row.hts_share), "synthetic_share": float(row.synthetic_share) }
            for row in df.itertuples()
        ]
        for dtir_id, df in df_flows.groupby("dtir_id")
    }

    return { "origins": points(df_sectors), "destinations": points(df_swiss_sectors), "flows": flows }


def plot_map(df_sectors, df_swiss_sectors, df_flows, path):
    df_map = df_sectors.to_crs("EPSG:4326")

    centroid = df_map.geometry.union_all().centroid
    m = folium.Map(location = [centroid.y, centroid.x], zoom_start = 9, tiles = None, prefer_canvas = True)
    folium.TileLayer(
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr = "Esri, HERE, Garmin, © OpenStreetMap contributors", name = "Esri Light Gray", control = False,
    ).add_to(m)

    fields = [
        "dtir_id", "dtir_name", "d30_name",
        "hts_secondary_only_ch_share", "hts_secondary_only_ch_ci95", "hts_nb_persons",
        "synthetic_secondary_only_ch_share", "synthetic_nb_persons", "difference",
        "hts_mandatory_ch_share", "synthetic_mandatory_ch_share",
    ]
    aliases = [
        "DTIR:", "Name:", "D30:",
        "EDGT share (%):", "EDGT ±95% CI:", "EDGT persons:",
        "Synthetic share (%):", "Synthetic persons:", "Synthetic − EDGT (pp):",
        "EDGT work/educ. in CH (%):", "Synthetic work/educ. in CH (%):",
    ]

    layers = [
        ("EDGT74 (survey)", "hts_secondary_only_ch_share", SHARE_COLORMAP, True),
        ("Synthetic population", "synthetic_secondary_only_ch_share", SHARE_COLORMAP, False),
        ("Difference (synthetic − EDGT)", "difference", DIFFERENCE_COLORMAP, False),
    ]

    geojson_layers = []
    for name, column, colormap, show in layers:
        def style_function(feature, column = column, colormap = colormap):
            value = feature["properties"][column]
            if value is None or pd.isna(value):
                return { "fillColor": "#dddddd", "color": "#9a9a9a", "weight": 0.6, "dashArray": "3, 3", "fillOpacity": 0.5 }
            return { "fillColor": colormap(np.clip(value, colormap.vmin, colormap.vmax)), "color": "#555555", "weight": 0.6, "fillOpacity": 0.8 }

        geojson_layers.append(folium.GeoJson(
            df_map[fields + ["geometry"]], name = name, show = show, overlay = False,
            style_function = style_function,
            highlight_function = lambda f: { "weight": 2.5, "color": "#222222" },
            tooltip = folium.GeoJsonTooltip(fields = fields, aliases = aliases, sticky = True),
        ).add_to(m))

    folium.GeoJson(
        df_swiss_sectors.to_crs("EPSG:4326")[["dtir_name", "geometry"]], name = "Swiss destination sectors (DTIR)",
        style_function = lambda f: { "fillColor": "#6b4c7a", "color": "#6b4c7a", "weight": 1.2, "fillOpacity": 0.06 },
        tooltip = folium.GeoJsonTooltip(fields = ["dtir_name"], aliases = ["Swiss sector:"]),
    ).add_to(m)

    SHARE_COLORMAP.caption = "Share with secondary but no work/education activity in Switzerland (%)"
    SHARE_COLORMAP.add_to(m)
    DIFFERENCE_COLORMAP.caption = "Synthetic − EDGT (percentage points)"
    DIFFERENCE_COLORMAP.add_to(m)

    folium.LayerControl(collapsed = False).add_to(m)
    m.add_child(FlowArrows(geojson_layers, build_flow_data(df_sectors, df_swiss_sectors, df_flows)))
    m.save(path)


def execute(context):
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")

    analysis_output_path = os.path.join(output_path, ANALYSIS_FOLDER)
    if not os.path.exists(analysis_output_path):
        os.mkdir(analysis_output_path)

    df_zones, _, _ = context.stage("data.hts.edgt_74.adisp_merge.zones")
    _, df_hts_persons, df_hts_trips = context.stage("data.hts.edgt_74.adisp_merge.merge")
    df_sectors = build_sectors(df_zones)
    df_swiss_sectors = build_swiss_sectors(df_zones)

    df_hts, df_hts_swiss = hts_persons(df_zones, df_hts_persons, df_hts_trips)
    df_synthetic, df_synthetic_swiss = synthetic_persons(
        df_sectors, df_swiss_sectors,
        context.stage("synthesis.population.enriched"),
        context.stage("synthesis.population.spatial.home.locations"),
        context.stage("synthesis.population.activities"),
        context.stage("synthesis.population.spatial.locations")[0],
    )

    df_comparison = build_comparison(df_sectors, df_hts, df_synthetic)
    df_comparison.to_csv(f"{analysis_output_path}/{output_prefix}secondary_activities_ch_by_dtir.csv", sep = ";", index = None)

    df_flows = build_flow_comparison(df_hts, df_hts_swiss, df_synthetic, df_synthetic_swiss, df_sectors, df_swiss_sectors)
    df_flows.to_csv(f"{analysis_output_path}/{output_prefix}secondary_activities_ch_flows_by_dtir.csv", sep = ";", index = None)

    df_total = df_comparison[df_comparison["dtir_id"] == "all"].iloc[0]
    print("Share with secondary but no work/education activity in Switzerland: EDGT74 %.2f%% (±%.2f), synthetic %.2f%%" % (
        df_total["hts_secondary_only_ch_share"], df_total["hts_secondary_only_ch_ci95"], df_total["synthetic_secondary_only_ch_share"]
    ))

    df_sectors = df_sectors.assign(dtir_id = df_sectors["dtir_id"].astype(str)).merge(
        df_comparison.drop(columns = ["dtir_name", "d30_name", "source", "nb_communes"]), on = "dtir_id", how = "left"
    )

    gpkg_path = f"{analysis_output_path}/{output_prefix}secondary_activities_ch_by_dtir.gpkg"
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    df_sectors.to_file(gpkg_path, driver = "GPKG")

    plot_comparison(df_comparison, f"{analysis_output_path}/{output_prefix}secondary_activities_ch_by_dtir.png")
    plot_map(df_sectors, df_swiss_sectors, df_flows, f"{analysis_output_path}/{output_prefix}secondary_activities_ch_by_dtir_map.html")

    print("Secondary-activities-in-Switzerland comparison written to %s" % analysis_output_path)

    return dict(comparison = df_comparison, flows = df_flows)
