import geopandas as gpd
import pandas as pd
import data.gtfs.utils as gtfs
import os, pathlib

"""
This file reads GTFS schedules, cuts them to the scenario area (defined by the
selected regions and departments, plus the Swiss cantons/municipalities when
cross-border flows are enabled) and merges them together.
"""

def configure(context):
    context.config("data_path")
    context.config("gtfs_path", "gtfs_idf")
    context.config("gtfs_files", None)

    context.config("generate_outbound_flows", "False")

    context.stage("data.spatial.municipalities")

    if context.config("generate_outbound_flows"):
        context.stage("data.spatial.ch.spatial")


def execute(context):
    gtfs_files = context.config("gtfs_files")

    if gtfs_files:
        input_files = gtfs_files
    else:
        input_files = get_input_files("{}/{}".format(context.config("data_path"), context.config("gtfs_path")))

    # Prepare bounding area
    df_area = context.stage("data.spatial.municipalities")[["geometry"]]

    if context.config("generate_outbound_flows"):
        _, df_ch_municipalities = context.stage("data.spatial.ch.spatial")
        df_ch_area = df_ch_municipalities[["geometry"]].to_crs(df_area.crs)

        df_area = gpd.GeoDataFrame(
            pd.concat([df_area, df_ch_area], ignore_index = True), crs = df_area.crs)

    # Load and cut feeds
    feeds = []
    for path in input_files:
        feed = gtfs.read_feed(path)
        feed = gtfs.cut_feed(feed, df_area)
        feeds.append(feed)

    # Merge feeds
    merged_feed = gtfs.merge_feeds(feeds) if len(feeds) > 1 else feeds[0]

    # Write feed (not as a ZIP, but as files, for pt2matsim)
    gtfs.write_feed(merged_feed, "{}/gtfs.zip".format(context.path()))

    return "gtfs.zip"


def get_input_files(base_path):
    gtfs_paths = [
        str(child)
        for child in pathlib.Path(base_path).glob("*")
        if child.suffix.lower() == ".zip"
    ]

    if len(gtfs_paths) == 0:
        raise RuntimeError("Did not find any GTFS data (.zip) in {}".format(base_path))
    
    return gtfs_paths


def validate(context):
    gtfs_files = context.config("gtfs_files")

    if gtfs_files:
        input_files = gtfs_files
    else:
        input_files = get_input_files("{}/{}".format(context.config("data_path"), context.config("gtfs_path")))

    total_size = 0

    for path in input_files:
        total_size += os.path.getsize(path)

    return total_size
