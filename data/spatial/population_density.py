import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
import logging

logger = logging.getLogger("synpp")

"""
Builds a population density index from IRIS-level aggregate population
counts (data.spatial.population) and IRIS polygons (data.spatial.iris) on
the French side, and (if generate_outbound_flows is enabled) from the ARE
SynPop point register (data.locations_CH.are_synpop.persons) on the Swiss
side.

Rather than sampling individual points (which does not scale well to large
populations), each IRIS is assigned an areal density (population / area).
The French density around a query point is estimated analytically as the
sum, over all IRIS zones intersecting a buffer of the requested radius
around the point, of intersection_area * iris_density. On the Swiss side,
ARE SynPop already gives one point per resident, so it is simply counted
with a radius query on a KDTree, like the SIRENE-based densities. Both
contributions are added together, giving an estimated resident count within
the radius, in the same unit as the SIRENE-based employee / company
densities.
"""


def configure(context):
    context.stage("data.spatial.population")
    context.stage("data.spatial.iris")

    context.config("generate_outbound_flows", "False")
    if context.config("generate_outbound_flows"):
        context.stage("data.locations_CH.are_synpop.persons")


def execute(context):
    df_population = context.stage("data.spatial.population")[["iris_id", "population"]]
    df_iris = context.stage("data.spatial.iris")[["iris_id", "geometry"]]

    df_iris = pd.merge(df_iris, df_population, on = "iris_id")
    df_iris = gpd.GeoDataFrame(df_iris, crs = "EPSG:2154")

    df_iris["density"] = df_iris["population"] / df_iris.geometry.area

    ch_kd_tree = None

    if context.config("generate_outbound_flows"):
        df_ch = context.stage("data.locations_CH.are_synpop.persons")[["x", "y"]]
        print("ARE SynPop residents in density index: %d" % len(df_ch))
        ch_kd_tree = KDTree(np.vstack([df_ch["x"], df_ch["y"]]).T)

    return {
        "df_iris": df_iris[["iris_id", "geometry", "density"]],
        "ch_kd_tree": ch_kd_tree
    }


def impute(context, density_data, df, x = "x", y = "y", radius = 500, point_type = "", chunk_size = 1e4):
    df_iris, ch_kd_tree = _unpack_density_data(density_data)

    logger.info("Imputing population density within %d m of %d %s coordinates...", radius, len(df), point_type)
    counts = []
    chunk_count = max(1, int(np.ceil(len(df) / chunk_size)))
    for chunk in context.progress(np.array_split(df, chunk_count),
                                  total = chunk_count,
                                  label = "Imputing population density..."):
        counts.extend(_query_density(df_iris, ch_kd_tree, chunk, x, y, radius))

    df["population_density"] = counts
    return df


def impute_parallel(context, density_data, df, x = "x", y = "y", radius = 500, point_type = "", chunk_size = 1e4, n_jobs = 10):
    df_iris, ch_kd_tree = _unpack_density_data(density_data)

    total_points = len(df)
    logger.info("Imputing population density within %d m of %d %s coordinates...", radius, total_points, point_type)

    chunk_count = max(1, int(np.ceil(total_points / chunk_size)))
    df_splits = np.array_split(df, chunk_count)

    results = Parallel(n_jobs = n_jobs)(
        delayed(_query_density)(df_iris, ch_kd_tree, chunk, x, y, radius)
        for chunk in context.progress(df_splits, total = chunk_count, label = "Imputing population density...")
    )

    counts = np.concatenate(results)
    df["population_density"] = counts
    return df


############# HELP FUNCTIONS #############

def _unpack_density_data(density_data):
    if isinstance(density_data, dict):
        return density_data["df_iris"], density_data.get("ch_kd_tree")
    return density_data, None


def _query_density(df_iris, ch_kd_tree, chunk, x, y, radius):
    coords = chunk[[x, y]].to_numpy(dtype = float)
    nan_mask = np.isnan(coords).any(axis = 1)

    result = np.zeros(len(coords), dtype = float)

    if (~nan_mask).any():
        valid_coords = coords[~nan_mask]

        df_buffers = gpd.GeoDataFrame({
            "row_id": np.arange(len(valid_coords))
        }, geometry = gpd.points_from_xy(valid_coords[:, 0], valid_coords[:, 1]).buffer(radius), crs = df_iris.crs)

        df_overlay = gpd.overlay(df_buffers, df_iris, how = "intersection")

        if len(df_overlay) > 0:
            df_overlay["weighted_count"] = df_overlay.geometry.area * df_overlay["density"]
            partial = df_overlay.groupby("row_id")["weighted_count"].sum()
            result[np.where(~nan_mask)[0][partial.index.to_numpy()]] += partial.to_numpy()

        if ch_kd_tree is not None:
            ch_counts = ch_kd_tree.query_radius(valid_coords, radius, count_only = True)
            result[~nan_mask] += ch_counts

    return result
