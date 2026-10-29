import numpy as np
import pandas as pd

"""
This stage samples home zones for all synthesized households. From the census
data we have several special cases that we need to cover:
- For people that live in municipalities without IRIS, only their departement is known
- For people that live in IRIS with less than 200 inhabitants, only their municipality is known

Based on these criteria, we can attach a random commune from the departement (which is not
covered by IRIS) to the first case, and we can attach a random IRIS within a commune that
has less than 200 inhabitants to the second case.

With home_zone_use_cantons, the commune of the first case is drawn among the
communes without IRIS of the household's canton (census CANTVILLE, matched
with data.spatial.canton_codes) instead of the whole departement.

For the first case, home_zone_method selects how the commune is drawn:
- "population": proportionally to the commune population
- "ilt": proportionally to the commune population times, for each household
  member, the probability of its type in the commune: not working, or working
  with a given place of work (census ILT: home commune, elsewhere in France,
  abroad) and commute mode, as counted in MOBPRO (data.od.workers_by_origin).
  A household with a cross-border commuter, for instance, then rather lands in
  a commune with many cross-border commuters.
"""

# Pseudo-count of the departement-wide type distribution added to the MOBPRO
# type counts of each candidate commune ("ilt")
HOME_ZONE_SMOOTHING = 20.0
HOME_ZONE_CHUNK_SIZE = 20000

NOT_WORKING = "not_working"


def configure(context):
    context.stage("synthesis.population.sampled_before_spatial_selection")

    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.iris")
    context.stage("data.spatial.population")

    context.config("random_seed")

    context.config("home_zone_method", "population")
    if context.config("home_zone_method") == "ilt":
        context.stage("data.od.workers_by_origin")

    context.config("home_zone_use_cantons", False)
    if context.config("home_zone_use_cantons"):
        context.stage("data.spatial.canton_codes")


def person_types(df_persons):
    """not working, or "<place of work>/<commute mode>" with the modes of
    data.od.workers_by_origin (motorcycles count as car, no mode = no transport)."""
    commute_mode = df_persons["commute_mode"].astype(object).replace({ "motorcycle": "car" }).fillna("no transport")
    place_of_work = df_persons["place_of_work"].astype(object).fillna("unknown")
    return (place_of_work + "/" + commute_mode).where(df_persons["employed"], NOT_WORKING)


def log_type_probabilities(df_workers, df_candidates, census_types):
    """log P(type | commune) as a (candidate commune x type) frame: MOBPRO
    workers by type, non-workers = population - workers, smoothed towards
    the distribution over all candidates (types never observed there keep a
    small, uniform probability)."""
    candidate_ids = df_candidates.index.astype(str)

    df_workers = df_workers[df_workers["origin_id"].isin(candidate_ids)]
    counts = df_workers.assign(type = df_workers["place_of_work"] + "/" + df_workers["commute_mode"]).pivot_table(
        index = "origin_id", columns = "type", values = "weight", aggfunc = "sum", fill_value = 0.0)

    types = sorted(set(counts.columns) | set(census_types) | { NOT_WORKING })
    counts = counts.reindex(index = candidate_ids, columns = types, fill_value = 0.0)

    population = pd.Series(df_candidates["population"].fillna(0.0).values, index = candidate_ids)
    counts[NOT_WORKING] = (population - counts.drop(columns = NOT_WORKING).sum(axis = 1)).clip(lower = 0.0)

    overall = (counts.sum() + 1.0) / (counts.values.sum() + len(types))
    smoothed = counts.add(HOME_ZONE_SMOOTHING * overall, axis = 1)
    return np.log(smoothed.div(smoothed.sum(axis = 1), axis = 0))


def sample_communes_by_types(random, df_composition, df_workers, df_candidates):
    """One commune per household (rows of df_composition: number of members
    per type), proportionally to population x prod P(member type | commune),
    by Gumbel-max sampling."""
    log_probabilities = log_type_probabilities(df_workers, df_candidates, df_composition.columns)
    composition = df_composition.reindex(columns = log_probabilities.columns, fill_value = 0).values.astype(float)
    log_prior = np.log(df_candidates["population"].fillna(0.0).values.astype(float))

    selected = np.empty(len(composition), dtype = int)
    for start in range(0, len(composition), HOME_ZONE_CHUNK_SIZE):
        chunk = composition[start:start + HOME_ZONE_CHUNK_SIZE]
        log_weights = chunk @ log_probabilities.values.T + log_prior
        selected[start:start + len(chunk)] = np.argmax(log_weights + random.gumbel(size = log_weights.shape), axis = 1)

    return df_candidates.index.values[selected]


def execute(context):
    random = np.random.default_rng(context.config("random_seed"))
    home_zone_method = context.config("home_zone_method")

    if not home_zone_method in ("population", "ilt"):
        raise RuntimeError("Unknown home_zone_method: %s" % home_zone_method)

    df_persons = context.stage("synthesis.population.sampled_before_spatial_selection")

    df_households = df_persons.drop_duplicates("household_id")[[
        "household_id", "commune_id", "iris_id", "canton_id", "departement_id"
    ]].copy().set_index("household_id")

    f_has_commune = df_households["commune_id"] != "undefined"
    f_has_iris = df_households["iris_id"] != "undefined"

    # Fix missing communes (we select from those without IRIS)
    df_municipalities = context.stage("data.spatial.municipalities").set_index("commune_id")
    df_municipalities["population"] = context.stage("data.spatial.population").groupby("commune_id")["population"].sum()

    df_households["commune_id"] = df_households["commune_id"].cat.add_categories(
        sorted(set(df_municipalities.index.unique()) - set(df_households["commune_id"].cat.categories)))

    # Candidate communes are those without IRIS of the household's canton
    # (census CANTVILLE) or, by default, of its departement
    if context.config("home_zone_use_cantons"):
        zone_column = "canton_id"
        df_cantons = context.stage("data.spatial.canton_codes").set_index("commune_id")["canton_id"]
        df_municipalities["canton_id"] = df_cantons.reindex(df_municipalities.index.astype(str)).values
    else:
        zone_column = "departement_id"

    zones = df_households[~f_has_commune][zone_column].unique()

    if home_zone_method == "ilt":
        df_workers = context.stage("data.od.workers_by_origin")

        # Number of members per type, for each household without commune
        df_target_persons = df_persons[df_persons["household_id"].isin(df_households.index[~f_has_commune])]
        df_composition = df_target_persons.assign(type = person_types(df_target_persons)).groupby(
            ["household_id", "type"]).size().unstack(fill_value = 0)

    for zone_id in context.progress(zones, label = "Fixing missing communes ..."):
        df_target = df_households[
            ~f_has_commune &
            (df_households[zone_column] == zone_id)].copy()

        df_candidates = df_municipalities[
            ~df_municipalities["has_iris"] &
            (df_municipalities[zone_column].astype(str) == zone_id)]

        if len(df_candidates) == 0:
            # e.g. a canton missing from data.spatial.canton_codes
            departement_id = str(df_target["departement_id"].iloc[0])
            print("No commune without IRIS in %s %s, placing its %d households in departement %s" % (
                zone_column, zone_id, len(df_target), departement_id))

            df_candidates = df_municipalities[
                ~df_municipalities["has_iris"] &
                (df_municipalities["departement_id"].astype(str) == departement_id)]

        if home_zone_method == "ilt":
            df_target["commune_id"] = sample_communes_by_types(
                random, df_composition.loc[df_target.index], df_workers, df_candidates)

        else:
            weights = df_candidates["population"].values.astype(float)
            weights /= np.sum(weights)

            indices = np.repeat(np.arange(weights.shape[0]), random.multinomial(len(df_target), weights))
            df_target["commune_id"] = df_candidates.reset_index()["commune_id"].iloc[indices].values

        df_households.loc[df_target.index, "commune_id"] = df_target["commune_id"]

    # Fix missing IRIS (we select from those with <200 inhabitants)
    df_iris = context.stage("data.spatial.iris").set_index("iris_id")
    df_iris["population"] = context.stage("data.spatial.population").set_index("iris_id")["population"]

    df_households["iris_id"] = df_households["iris_id"].cat.add_categories(
        sorted(set(df_iris.index.unique()) - set(df_households["iris_id"].cat.categories)))

    communes = df_households[~f_has_iris & f_has_commune]["commune_id"].unique()

    for commune_id in context.progress(communes, label = "Fixing missing IRIS ..."):
        df_candidates = df_iris[
            (df_iris["population"] <= 200) &
            (df_iris["commune_id"].astype(str) == commune_id)]

        df_target = df_households[
            f_has_commune & ~f_has_iris &
            (df_households["commune_id"] == commune_id)].copy()

        weights = df_candidates["population"].values.astype(float)
        if (weights == 0.0).all(): weights += 1.0
        weights /= np.sum(weights)

        indices = np.repeat(np.arange(weights.shape[0]), random.multinomial(len(df_target), weights))
        df_target["iris_id"] = df_candidates.reset_index()["iris_id"].iloc[indices].values

        df_households.loc[df_target.index, "iris_id"] = df_target["iris_id"]

    # Check that everybody has a commune now
    assert np.count_nonzero(df_households["commune_id"] == "undefined") == 0

    # Now there are some people left who don't have an IRIS, because the commune
    # is not covered in IRIS. Hence, we drive the commune-based IRIS for them.
    f = df_households["iris_id"] == "undefined"
    df_households.loc[f, "iris_id"] = df_households.loc[f, "commune_id"].astype(str) + "0000"

    # Finally, make sure that we have no invalid codes
    invalid_communes = set(df_households["commune_id"].unique()) - set(df_municipalities.index.unique())
    invalid_iris     = set(df_households["iris_id"].unique()) - set(df_iris.index.unique())

    assert len(invalid_communes) == 0
    assert len(invalid_iris) == 0
    assert np.count_nonzero(df_households["iris_id"] == "undefined") == 0

    return df_households.reset_index()[["household_id", "departement_id", "commune_id", "iris_id"]]
