import io, gzip

import numpy as np
import pandas as pd

import matsim.writers as writers

def configure(context):
    context.stage("synthesis.locations.secondary")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")

HOME_FIELDS = [
    "household_id", "geometry"
]

PRIMARY_FIELDS = [
    "location_id", "geometry", "is_work"
]

SECONDARY_FIELDS = [
    "location_id", "geometry", "offers_leisure", "offers_shop", "offers_other"
]

def execute(context):
    output_path = "%s/facilities.xml.gz" % context.path()

    df_homes = context.stage("synthesis.population.spatial.home.locations")
    df_homes = df_homes[HOME_FIELDS]
    
    with gzip.open(output_path, 'wb+') as writer:
        with io.BufferedWriter(writer, buffer_size = 2 * 1024**3) as writer:
            writer = writers.FacilitiesWriter(writer)
            writer.start_facilities({
                "coordinateReferenceSystem": "Atlantis"#df_homes.crs
            })

            # Write home

            with context.progress(total = len(df_homes), label = "Writing home facilities ...") as progress:
                for item in df_homes.itertuples(index = False):
                    geometry = item[HOME_FIELDS.index("geometry")]

                    writer.start_facility(
                        "home_%s" % item[HOME_FIELDS.index("household_id")],
                        geometry.x, geometry.y
                    )

                    writer.add_activity("home")
                    writer.end_facility()

            # Write primary

            df_work, df_education = context.stage("synthesis.population.spatial.primary.locations")

            df_work = df_work.drop_duplicates("location_id").copy()
            df_education = df_education.drop_duplicates("location_id").copy()

            df_work["is_work"] = True
            df_education["is_work"] = False

            df_locations = pd.concat([df_work, df_education])
            df_locations = df_locations[PRIMARY_FIELDS]

            # A STATENT location can be chosen as both work and education;
            # merge into one facility offering both, since ids are now shared.
            has_work      = df_locations.groupby("location_id")["is_work"].transform("any")
            has_education = df_locations.groupby("location_id")["is_work"].transform(lambda s: (~s).any())
            df_locations   = df_locations.assign(has_work = has_work, has_education = has_education)
            df_locations   = df_locations.drop_duplicates("location_id")

            written_ids = set(df_locations["location_id"])

            with context.progress(total = len(df_locations), label = "Writing primary facilities ...") as progress:
                for item in df_locations.itertuples(index = False):
                    geometry = item[PRIMARY_FIELDS.index("geometry")]

                    writer.start_facility(
                        str(item[PRIMARY_FIELDS.index("location_id")]),
                        geometry.x, geometry.y
                    )

                    if item.has_work: writer.add_activity("work")
                    if item.has_education: writer.add_activity("education")
                    writer.end_facility()

            # Write secondary

            df_locations = context.stage("synthesis.locations.secondary")
            df_locations = df_locations[SECONDARY_FIELDS]

            # May already have been written above (shared STATENT id) - skip.
            df_locations = df_locations[~df_locations["location_id"].isin(written_ids)]

            with context.progress(total = len(df_locations), label = "Writing secondary facilities ...") as progress:
                for item in df_locations.itertuples(index = False):
                    geometry = item[SECONDARY_FIELDS.index("geometry")]

                    writer.start_facility(
                        item[SECONDARY_FIELDS.index("location_id")],
                        geometry.x, geometry.y
                    )

                    for purpose in ("shop", "leisure", "other"):
                        if item[SECONDARY_FIELDS.index("offers_%s" % purpose)]:
                            writer.add_activity(purpose)

                    writer.end_facility()
                    progress.update()

            writer.end_facilities()

    return "facilities.xml.gz"
