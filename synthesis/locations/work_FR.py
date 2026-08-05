import pandas as pd
import geopandas as gpd
import data.id_utils


"""
This stage provides a list of work places that serve as potential locations for
work activities. It is derived from the SIRENE enterprise database.

Municipalities which do not have any registered enterprise receive a fake work
place at their centroid to be in line with INSEE OD data.
"""


def configure(context):
    context.stage("data.sirene.localized")
    context.stage("data.spatial.municipalities")


def execute(context):
    df_workplaces = context.stage("data.sirene.localized")[[
        "commune_id", "minimum_employees", "maximum_employees", "geometry"
    ]].copy()

    # Canonical id from SIRENE's own siret (the DataFrame index here),
    # base62-encoded. Keep the "FR_SIRET_" prefix/encoding in sync with
    # eqasim-switzerland's French-location stages.
    df_workplaces["location_id"] = "FR_SIRET_" + df_workplaces.index.astype(int).map(data.id_utils.to_base62)

    # Use minimum number of employees as weight
    df_workplaces["employees"] = df_workplaces["minimum_employees"]
    df_workplaces["fake"] = False

    ## Use centroids for municipalities where no work places exist
    df_zones = context.stage("data.spatial.municipalities")
    required_communes = set(df_zones["commune_id"].unique())
    missing_communes = required_communes - set(df_workplaces["commune_id"].unique())

    if len(missing_communes) > 0:
        print("Adding work places at the centroid of %d/%d communes without SIRENE observations" % (
            len(missing_communes), len(required_communes)))

        df_added = []

        for commune_id in missing_communes:
            centroid = df_zones[df_zones["commune_id"] == commune_id]["geometry"].centroid.iloc[0]

            # No SIRET here, so fall back to a commune-keyed id (base62 when
            # numeric; Corsica's "2A"/"2B" codes are kept as-is).
            commune_key = data.id_utils.to_base62(int(commune_id)) if str(commune_id).isdigit() else str(commune_id)
            df_added.append({
                "commune_id": commune_id, "employees": 1.0, "geometry": centroid,
                "location_id": "FR_WORK_CENTROID_" + commune_key,
            })

        df_added = gpd.GeoDataFrame(pd.DataFrame.from_records(df_added), crs = df_workplaces.crs)
        df_added["fake"] = True

        df_workplaces = pd.concat([df_workplaces, df_added])

    return df_workplaces[["location_id", "commune_id", "employees", "fake", "geometry"]]
