import numpy as np
import pandas as pd
import numba

import data.hts.egt.cleaned
import data.hts.entd.cleaned


"""
This stage attaches obervations from the household travel survey to the synthetic
population sample. This is done by statistical matching.
"""

INCOME_CLASS = {
    "egt": data.hts.egt.cleaned.calculate_income_class,
    "entd": data.hts.entd.cleaned.calculate_income_class,
}

DEFAULT_MATCHING_ATTRIBUTES = [
    "sex", "age_class", "any_cars", "socioprofessional_class",
    #"departement_id"
]

DEFAULT_MANDATORY_MATCHING_ATTRIBUTES = [
    "sex", "age_class", "any_cars",
    #"departement_id"
]


def configure(context):
    context.config("processes", volatile = True)
    context.config("random_seed")
    context.config("matching_minimum_observations", 20)
    context.config("matching_attributes", DEFAULT_MATCHING_ATTRIBUTES)
    context.config("mandatory_matching_attributes", DEFAULT_MANDATORY_MATCHING_ATTRIBUTES)

    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.income.selected")

    context.stage("data.hts.selected", alias = "hts")


@numba.njit(cache = True)
def sample_indices(uniform, cdf, selected_indices):
    n = len(uniform)
    out = np.empty(n, dtype = np.int64)

    # Binary search over CDF for each random draw.
    for i in range(n):
        u = uniform[i]
        lo = 0
        hi = len(cdf)

        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid

        out[i] = selected_indices[lo]

    return out


def decrease_minimum_observation(N):
    # Any decreasing function can be implemented here.
    return N-1


def is_left_slice(list1, list2):
    return list1[:len(list2)] == list2


def recursive_iteration_statmatch(df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns=None,
                         rng=None, minimum_observations=0):
    
    # Reduce data frames
    df_source = df_source[[source_identifier, weight] + columns].copy()
    df_target = df_target[[target_identifier] + columns].copy()

    # Sort data frames
    df_source = df_source.sort_values(by=columns)
    df_target = df_target.sort_values(by=columns)

    # Perform matching
    weights = df_source[weight].values
    assigned_indices = np.ones((len(df_target),), dtype=int) * -1
    unassigned_mask = np.ones((len(df_target),), dtype=bool)
    assigned_levels = np.ones((len(df_target),), dtype=int) * -1
    uniform = rng.random_sample(size=(len(df_target),))

    if mandatory_columns:
        minimum_level = len(mandatory_columns)
    else:
        minimum_level = 1

    for level in range(len(columns), minimum_level - 1, -1):
        if not np.any(unassigned_mask):
            break

        level_columns = columns[:level]

        # Group indices once per level instead of evaluating all value combinations
        # against full-length boolean masks.
        source_groups = df_source.groupby(level_columns, sort=False, observed = True).indices
        target_groups = df_target.groupby(level_columns, sort=False, observed = True).indices

        for key, target_indices_all in target_groups.items():
            target_indices = target_indices_all[unassigned_mask[target_indices_all]]

            if len(target_indices) == 0:
                continue

            selected_indices = source_groups.get(key, None)
            if selected_indices is None:
                continue

            if len(selected_indices) < minimum_observations:
                continue

            selected_weights = weights[selected_indices]
            cdf = np.cumsum(selected_weights)
            cdf /= cdf[-1]

            assigned_indices[target_indices] = sample_indices(uniform[target_indices], cdf, selected_indices)
            assigned_levels[target_indices] = level
            unassigned_mask[target_indices] = False

    # Randomly assign unmatched observations
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    assigned_indices[unassigned_mask] = sample_indices(uniform[unassigned_mask], cdf, np.arange(len(weights), dtype=np.int64))
    assigned_levels[unassigned_mask] = 0

    # Write back indices
    df_target[source_identifier] = df_source[source_identifier].values[assigned_indices]

    return df_target, assigned_levels


def statistical_matching(progress, df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns=None,
                         random_seed=0, minimum_observations=0, percentage_matched = 0, initial_nb_of_agents = 0):
    
    # Columns check: mandatory columns should be a "left-slice" of columns.
    if mandatory_columns:
        if not is_left_slice(columns, mandatory_columns):
            raise RuntimeError("Mandatory columns must match the beginning of columns!")

    if initial_nb_of_agents == 0 and len(df_target) > 0:
        initial_nb_of_agents = len(df_target)

    assert(initial_nb_of_agents > 0)

    # Set up RNG
    rng = np.random.RandomState(random_seed)

    # Termination step
    if minimum_observations == 1:
        # At this point, the goal is to match everyone. So we do not consider the mandatory columns any longer.
        df_matching, assigned_levels = recursive_iteration_statmatch(df_source, source_identifier, weight, df_target, target_identifier, columns, None,
                         rng, minimum_observations)

        share_of_matched_agents = round(len(df_matching) / initial_nb_of_agents * 100,2) + percentage_matched
        
        print(f"{minimum_observations} obs required - {share_of_matched_agents:.2f}% of the population matched.")
        
        return df_matching, assigned_levels

    else:
        df_matching, assigned_levels = recursive_iteration_statmatch(df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns,
                         rng, minimum_observations)
        
        df_not_matching_on_mandatory = df_matching[assigned_levels < len(mandatory_columns)]
        df_matching_on_mandatory     = df_matching[assigned_levels >= len(mandatory_columns)]

        matched_levels               = assigned_levels[assigned_levels >= len(mandatory_columns)]

        next_minimum_observations    =  decrease_minimum_observation(minimum_observations)

        share_of_matched_agents = len(df_matching_on_mandatory) / initial_nb_of_agents * 100 + percentage_matched

        print(f"{minimum_observations} obs required - {share_of_matched_agents:.2f}% of the population matched.")
        
        matching_the_missing, levels = statistical_matching(progress, df_source, source_identifier, weight, df_not_matching_on_mandatory, target_identifier, columns, mandatory_columns, random_seed, next_minimum_observations, share_of_matched_agents, initial_nb_of_agents)
        
        return pd.concat([df_matching_on_mandatory, matching_the_missing]), np.concatenate((matched_levels, levels))


def _run_parallel_statistical_matching(context, args):
    # Pass arguments
    df_target, random_seed = args

    # Pass data
    df_source = context.data("df_source")
    source_identifier = context.data("source_identifier")
    weight = context.data("weight")
    target_identifier = context.data("target_identifier")
    columns = context.data("columns")
    minimum_observations = context.data("minimum_observations")

    return statistical_matching(context.progress, df_source, source_identifier, weight, df_target, target_identifier, columns, random_seed, minimum_observations)


def nonparallel_statistical_matching(context, df_source, source_identifier, weight, df_target, target_identifier, columns,
                                  mandatory_columns, minimum_observations=0):
    
    random_seed = context.config("random_seed")
    
    return statistical_matching(context.progress, df_source, source_identifier, weight, df_target, target_identifier,
                                columns, mandatory_columns, random_seed, minimum_observations)


def run_statistical_matching_extended(context, df_source, source_identifier, weight,
                                      df_population, target_identifier,
                                      columns, mandatory_columns,
                                      minimum_observations=0, population_selector=None,
                                      option = "person"):
    
    df_target = df_population.copy()
    
    if population_selector is not None:
        df_target = df_target.loc[population_selector].copy()

    df_assignment, levels = nonparallel_statistical_matching(
        context,
        df_source, source_identifier, weight,
        df_target, target_identifier,
        columns,
        mandatory_columns,
        minimum_observations=minimum_observations)
    
    df_target = pd.merge(
        df_target,
        df_assignment,
        on=target_identifier,
        validate="one_to_one",
    )

    assert len(df_target) == len(df_assignment)

    #context.set_info("matched_counts", {
    #    count: np.count_nonzero(levels >= count) for count in range(len(columns) + 1)
    #})

    for count in range(len(columns) + 1):
        matched_count = np.count_nonzero(levels >= count)
        matched_percent = 100 * matched_count / len(df_target) if len(df_target) > 0 else 0.0
        print(f"{count} matched levels: {matched_count} ({matched_percent:.2f}%)")
        
    # Remove and track unmatchable households (i.e. head of household)

    initial_population_length = len(df_population)
    initial_target_length     = len(df_target)
        
    if option == "household":

        unmatchable_household_selector = levels < 1
        umatchable_household_ids       = set(df_target.loc[unmatchable_household_selector, "household_id"].values)

        unmatchable_person_selector    = df_population["household_id"].isin(umatchable_household_ids)
        removed_person_ids             = set(df_population.loc[unmatchable_person_selector, "person_id"].values)

        removed_household_ids = set() | umatchable_household_ids

        df_target     = df_target.loc[~unmatchable_household_selector, :]
        df_population = df_population.loc[~unmatchable_person_selector, :]

        removed_households_count = sum(unmatchable_household_selector)
        removed_persons_count    = sum(unmatchable_person_selector)

        print("Unmatchable heads of household: %d", removed_households_count)
        print("  Removed households: %d", removed_households_count)
        print("  Removed persons: %d", removed_persons_count)
        print("")

        assert (len(df_target)     == initial_target_length     - removed_households_count)
        assert (len(df_population) == initial_population_length - removed_persons_count)

        return df_target, df_population, [removed_person_ids, removed_household_ids]
    
    elif option == "person":

        unmatchable_person_selector        = levels < 1
        unmatchable_person_ids             = set(df_target.loc[unmatchable_person_selector, "person_id"].values)
        unmatchable_person_selector        = df_population["person_id"].isin(unmatchable_person_ids) 
        unmatchable_person_selector_target = df_target["person_id"].isin(unmatchable_person_ids)  

        df_target     = df_target.loc[~unmatchable_person_selector_target, :]
        df_population = df_population.loc[~unmatchable_person_selector, :]  

        removed_persons_count = sum(unmatchable_person_selector)

        print("  Removed persons: %d", removed_persons_count)
        print("")

        assert (len(df_target)     == initial_target_length     - removed_persons_count)
        assert (len(df_population) == initial_population_length - removed_persons_count)

        return df_target, df_population, [unmatchable_person_ids, None]



def execute(context):
    hts = context.config("hts")

    # Load data
    df_source_households, df_source_persons, df_source_trips = context.stage("hts")
    df_source = pd.merge(df_source_persons, df_source_households)

    df_target = context.stage("synthesis.population.sampled")

    columns           = context.config("matching_attributes")
    mandatory_columns = context.config("mandatory_matching_attributes")

    if hts == "edgt_74":
        columns           = ["edgt_area"] + columns
        mandatory_columns = ["edgt_area"] + mandatory_columns

    try:
        default_index = columns.index("*default*")
        columns[default_index:default_index + 1] = DEFAULT_MATCHING_ATTRIBUTES
    except ValueError: pass

    # Define matching attributes
    AGE_BOUNDARIES = [14, 17, 29, 44, 59, 74, 1000]

    if "age_class" in columns:
        df_target["age_class"] = np.digitize(df_target["age"], AGE_BOUNDARIES, right = True)
        df_source["age_class"] = np.digitize(df_source["age"], AGE_BOUNDARIES, right = True)

    if "income_class" in columns:
        df_income = context.stage("synthesis.population.income.selected")[["household_id", "household_income"]]

        df_target = pd.merge(df_target, df_income)
        df_target["income_class"] = INCOME_CLASS[hts](df_target)

    if "any_cars" in columns:
        df_target["any_cars"] = df_target["number_of_vehicles"] > 0
        df_source["any_cars"] = df_source["number_of_vehicles"] > 0

    # Perform statistical matching
    df_source = df_source.rename(columns = { "person_id": "hts_id" })

    for column in columns:
        if not column in df_source:
            raise RuntimeError("Attribute not available in source (HTS) for matching: {}".format(column))

        if not column in df_target:
            raise RuntimeError("Attribute not available in target (census) for matching: {}".format(column))

    df_assignment, levels = nonparallel_statistical_matching(
        context,
        df_source, "hts_id", "person_weight",
        df_target, "person_id",
        columns,
        mandatory_columns,
        minimum_observations = context.config("matching_minimum_observations"))

    df_assignment = df_assignment[["person_id", "hts_id"]]

    df_target = pd.merge(df_target, df_assignment, on = "person_id")
    assert len(df_target) == len(df_assignment)

    context.set_info("matched_counts", {
        count: int(np.count_nonzero(levels >= count)) for count in range(len(columns) + 1)
    })

    for count in range(len(columns) + 1):
        print("%d matched levels:" % count, np.count_nonzero(levels >= count), "%.2f%%" % (100 * np.count_nonzero(levels >= count) / len(df_target),))

    return df_target[["person_id", "hts_id"]]
