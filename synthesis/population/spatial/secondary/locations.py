import numpy as np
import pandas as pd
import shapely.geometry as geo
import geopandas as gpd

from synthesis.population.spatial.secondary.problems import find_assignment_problems

def configure(context):
    context.stage("synthesis.population.trips")

    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")

    context.stage("synthesis.population.spatial.secondary.distance_distributions")
    context.stage("synthesis.locations.secondary")

    context.config("random_seed")
    context.config("processes", volatile = True)

    context.config("secondary_activities.maximum_iterations", np.inf)

    chain_solver = context.config("secondary_activities.chain_solver", "default")
    assert chain_solver in ("default", "force_model")

    if chain_solver == "force_model":
        context.stage("synthesis.population.spatial.secondary.force_model.force_field")

    # Penalty on secondary locations in another country than the chain's
    # fixed activities, calibrated in synthesis.population.spatial.secondary.border_penalty
    # (see BorderPenaltyAssignmentSolver)
    if context.config("secondary_activities.border_penalty", False):
        assert context.config("generate_outbound_flows"), "secondary_activities.border_penalty needs generate_outbound_flows"
        context.stage("synthesis.population.spatial.secondary.border_penalty")

    # Lakes as barriers for non-PT trips (see synthesis.population.spatial.secondary.lakes)
    if context.config("secondary_activities.lake_barrier", False):
        context.stage("data.spatial.ch.lakes")

    context.config("generate_outbound_flows")
    if context.config("generate_outbound_flows"):
        context.stage("data.locations_CH.secondary")

def prepare_locations(context):
    # Load persons and their primary locations
    df_home = context.stage("synthesis.population.spatial.home.locations")
    df_work, df_education, _ = context.stage("synthesis.population.spatial.primary.locations")
    crs = df_home.crs

    df_home = df_home.rename(columns = { "geometry": "home" })
    df_work = df_work.rename(columns = { "geometry": "work" })
    df_education = df_education.rename(columns = { "geometry": "education" })

    df_locations = context.stage("synthesis.population.sampled")[["person_id", "household_id"]]
    df_locations = pd.merge(df_locations, df_home[["household_id", "home"]], how = "left", on = "household_id")
    df_locations = pd.merge(df_locations, df_work[["person_id", "work"]], how = "left", on = "person_id")
    df_locations = pd.merge(df_locations, df_education[["person_id", "education"]], how = "left", on = "person_id")

    # Country of the fixed activities, for the border penalty (homes are
    # always in France)
    swiss_work = set(df_work.loc[df_work["commune_id"].astype(str).str.startswith("CH"), "person_id"])
    swiss_education = set(df_education.loc[df_education["commune_id"].astype(str).str.startswith("CH"), "person_id"])
    df_locations["work_in_ch"] = df_locations["person_id"].isin(swiss_work)
    df_locations["education_in_ch"] = df_locations["person_id"].isin(swiss_education)

    df_locations["border_acceptance"] = 1.0
    if context.config("secondary_activities.border_penalty"):
        df_locations["border_acceptance"] = home_border_acceptance(context, gpd.GeoDataFrame(df_locations, geometry = "home", crs = crs))

    return df_locations[["person_id", "home", "work", "education", "work_in_ch", "education_in_ch", "border_acceptance"]].sort_values(by = "person_id"), crs

def home_border_acceptance(context, df_locations):
    df_sectors, _ = context.stage("synthesis.population.spatial.secondary.border_penalty")

    df_homes = df_locations[["person_id", "home"]].drop_duplicates("person_id")
    df_homes = gpd.sjoin(df_homes, df_sectors.to_crs(df_locations.crs), how = "left", predicate = "within")
    df_homes = df_homes.drop_duplicates("person_id").set_index("person_id")

    acceptance = df_homes["border_acceptance"].fillna(1.0)
    return df_locations["person_id"].map(acceptance).values

def prepare_destinations(context):
    df_locations = context.stage("synthesis.locations.secondary")

    identifiers = df_locations["location_id"].values
    locations = np.vstack(df_locations["geometry"].apply(lambda x: np.array([x.x, x.y])).values)

    data = {}

    for purpose in ("shop", "leisure", "other"):
        f = df_locations["offers_%s" % purpose].values

        data[purpose] = dict(
            identifiers = identifiers[f],
            locations = locations[f]
        )

    return data

def resample_cdf(cdf, factor):
    if factor >= 0.0:
        cdf = cdf * (1.0 + factor * np.arange(1, len(cdf) + 1) / len(cdf))
    else:
        cdf = cdf * (1.0 + abs(factor) - abs(factor) * np.arange(1, len(cdf) + 1) / len(cdf))

    cdf /= cdf[-1]
    return cdf

def resample_distributions(distributions, factors):
    for mode, mode_distributions in distributions.items():
        for distribution in mode_distributions["distributions"]:
            distribution["cdf"] = resample_cdf(distribution["cdf"], factors[mode])

from synthesis.population.spatial.secondary.rda import AssignmentSolver, DiscretizationErrorObjective, GravityChainSolver, AngularTailSolver, GeneralRelaxationSolver
from synthesis.population.spatial.secondary.components import CustomDistanceSampler, CustomDiscretizationSolver, CandidateIndex, CustomFreeChainSolver
from synthesis.population.spatial.secondary.force_model.solver import ForceFieldChainSolver
from synthesis.population.spatial.secondary.lakes import LakeBarrier

class BorderPenaltyAssignmentSolver(AssignmentSolver):
    """AssignmentSolver that penalizes secondary locations in another
    country than the chain's fixed activities (e.g. home -> Geneva shop ->
    home, but not work in Geneva -> Geneva restaurant -> work in Geneva):
    the distance distributions don't know about the border, so without this
    a French resident living next to it does about half of their secondary
    activities abroad. A chain is allowed to cross the border with the
    person's probability `border_acceptance`, times the weight of its Swiss
    destination zone (`swiss_identifiers`, location_id -> weight).
    Otherwise the solver keeps iterating for a solution that doesn't cross,
    even one whose distances don't match the sampled ones, and only falls
    back to the crossing one if all attempts cross: next to the border, many
    chains never find a valid domestic solution and would otherwise keep the
    crossing whatever the acceptance. Chains with a trip blocked by
    `lake_barrier` (see synthesis.population.spatial.secondary.lakes) are
    rejected the same way, whatever their country."""

    def __init__(self, random, swiss_identifiers, anchors_in_ch, lake_barrier = None, **kwargs):
        AssignmentSolver.__init__(self, **kwargs)
        self.lake_barrier = lake_barrier
        self.random = random
        self.swiss_identifiers = swiss_identifiers
        self.anchors_in_ch = anchors_in_ch

    def crossing_weight(self, problem, discretization_result):
        """None if the chain stays in the country of its fixed activities,
        otherwise the lowest destination weight of its activities abroad
        (1.0 for activities in France)."""
        anchors_in_ch = self.anchors_in_ch[problem["person_id"]]
        countries = set(
            anchors_in_ch[purpose] for purpose in (problem["origin_purpose"], problem["destination_purpose"])
            if purpose is not None
        ) or { False } # free chains: residents of France

        weights = [
            self.swiss_identifiers.get(identifier, 1.0) for identifier in discretization_result["identifiers"]
            if (identifier in self.swiss_identifiers) not in countries
        ]

        return min(weights) if len(weights) > 0 else None

    def solve(self, problem):
        best_result, best_key = None, None
        # Drawn once per chain, not per attempt: otherwise a crossing gets
        # accepted on one of the attempts and ends the loop most of the time.
        acceptance = self.anchors_in_ch[problem["person_id"]]["border_acceptance"]
        draw = self.random.random()

        for assignment_iteration in range(self.maximum_iterations):
            distance_result = self.distance_sampler.sample(problem)

            relaxation_result = self.relaxation_solver.solve(problem, distance_result["distances"])
            discretization_result = self.discretization_solver.solve(problem, relaxation_result["locations"])

            assignment_result = self.objective.evaluate(problem, distance_result, relaxation_result, discretization_result)

            weight = self.crossing_weight(problem, discretization_result)
            rejected = weight is not None and draw >= acceptance * weight

            # Chains going around a lake (Lake Geneva) are rejected the same way
            if not rejected and self.lake_barrier is not None:
                rejected = self.lake_barrier.blocks_chain(problem, discretization_result["locations"])
            key = (rejected, not assignment_result["valid"], assignment_result["objective"])

            if best_result is None or key < best_key:
                best_result, best_key = assignment_result, key

                assignment_result["distance"] = distance_result
                assignment_result["relaxation"] = relaxation_result
                assignment_result["discretization"] = discretization_result
                assignment_result["iterations"] = assignment_iteration
                rejected_best = rejected

            if best_result["valid"] and not rejected_best:
                break

        return best_result

# Sentinel location_id for a discretized secondary activity whose geometry
# was forced onto a fixed (home/work/education) anchor by a "_loop" mode
# constraint below: there is no real facility for it in synthesis.locations.secondary,
# so we reuse the same sentinel already used for home in
# synthesis.population.spatial.locations.
LOOP_ANCHOR_LOCATION_ID = -1

def apply_loop_constraints(problem, result):
    """Trips tagged with a "_loop" mode (data.hts.edgt_74.adisp_merge.merge.
    tag_short_trip_loop_mode) represent very short "went around the block"
    trips rather than trips to a genuinely new place: their destination
    should coincide exactly with their origin. The discretization/relaxation
    solvers above don't know about this, so we force it here as a
    post-processing step on the already-solved chain."""
    modes = problem["modes"]

    if not any(mode.endswith("_loop") for mode in modes):
        return

    discretization = result["discretization"]
    locations = list(discretization["locations"])
    identifiers = list(discretization["identifiers"])

    origin, destination = problem["origin"], problem["destination"]

    sequence_locations = []
    sequence_identifiers = []

    if origin is not None:
        sequence_locations.append(np.asarray(origin).reshape(2))
        sequence_identifiers.append(LOOP_ANCHOR_LOCATION_ID)

    for location, identifier in zip(locations, identifiers):
        sequence_locations.append(np.asarray(location).reshape(2))
        sequence_identifiers.append(identifier)

    if destination is not None:
        sequence_locations.append(np.asarray(destination).reshape(2))
        sequence_identifiers.append(LOOP_ANCHOR_LOCATION_ID)

    if len(sequence_locations) - 1 != len(modes):
        # Chain structure doesn't match what we expect (e.g. an unusual free
        # chain); skip rather than risk corrupting the discretization.
        return

    changed = False

    for index, mode in enumerate(modes):
        if not mode.endswith("_loop"):
            continue

        destination_is_fixed_anchor = (destination is not None) and (index + 1 == len(sequence_locations) - 1)
        origin_is_fixed_anchor = (origin is not None) and (index == 0)

        if not destination_is_fixed_anchor:
            # Collapse the destination of this trip onto its origin.
            sequence_locations[index + 1] = sequence_locations[index].copy()
            sequence_identifiers[index + 1] = sequence_identifiers[index]
            changed = True

        elif not origin_is_fixed_anchor:
            # Destination is a fixed anchor (this is the chain's last trip):
            # can't move it, so pull the variable origin onto it instead.
            sequence_locations[index] = sequence_locations[index + 1].copy()
            sequence_identifiers[index] = sequence_identifiers[index + 1]
            changed = True

        else:
            # Both endpoints are fixed activities: a loop trip directly
            # between e.g. home and work can't be constrained here.
            print("Warning: cannot enforce loop constraint for person %s (both trip endpoints are fixed activities)" % problem["person_id"])

    if changed:
        start = 1 if origin is not None else 0
        end = len(sequence_locations) - (1 if destination is not None else 0)

        discretization["locations"] = np.vstack(sequence_locations[start:end])
        discretization["identifiers"] = sequence_identifiers[start:end]

def execute(context):
    # Load trips and primary locations
    df_trips = context.stage("synthesis.population.trips").sort_values(by = ["person_id", "trip_index"])
    df_trips["travel_time"] = df_trips["arrival_time"] - df_trips["departure_time"]
    df_primary, crs = prepare_locations(context)

    # in case it exists, replace motorcycle mode by car for this location algorithm where we don't need the distinction
    df_trips.loc[df_trips["mode"] == "motorcycle", "mode"] = "car"

    # Prepare data
    distance_distributions = context.stage("synthesis.population.spatial.secondary.distance_distributions")
    destinations = prepare_destinations(context)

    swiss_identifiers = {}
    if context.config("secondary_activities.border_penalty"):
        _, df_swiss = context.stage("synthesis.population.spatial.secondary.border_penalty")
        swiss_identifiers = dict(zip(df_swiss["location_id"], df_swiss["weight"]))

    lake_polygons = None
    if context.config("secondary_activities.lake_barrier"):
        lake_polygons = list(context.stage("data.spatial.ch.lakes").to_crs(crs)["geometry"])

    # Optional data for force model
    force_field_data = None
    if context.config("secondary_activities.chain_solver") == "force_model":
        force_field_data = context.stage("synthesis.population.spatial.secondary.force_model.force_field")

    # Resampling for calibration
    resample_factors = dict(
        car = 0.0, car_passenger = 0.1, pt = 0.5, bike = 0.0, walk = -0.5
    )
    resample_factors.update({
        "%s_loop" % mode: factor for mode, factor in resample_factors.items()
    })
    resample_distributions(distance_distributions, resample_factors)

    # Segment into subsamples
    processes = context.config("processes")

    unique_person_ids = df_trips["person_id"].unique()
    number_of_persons = len(unique_person_ids)
    unique_person_ids = np.array_split(unique_person_ids, processes)

    random = np.random.default_rng(context.config("random_seed"))
    random_seeds = random.integers(10000, size = processes)

    # Create batch problems for parallelization
    batches = []

    for index in range(processes):
        batches.append((
            df_trips[df_trips["person_id"].isin(unique_person_ids[index])],
            df_primary[df_primary["person_id"].isin(unique_person_ids[index])],
            random_seeds[index], crs
        ))

    # Run algorithm in parallel
    with context.progress(label = "Assigning secondary locations to persons", total = number_of_persons):
        with context.parallel(processes = processes, data = dict(
            distance_distributions = distance_distributions,
            destinations = destinations,
            force_field_data = force_field_data,
            swiss_identifiers = swiss_identifiers,
            lake_polygons = lake_polygons
        )) as parallel:
            df_locations, df_convergence = [], []

            for df_locations_item, df_convergence_item in parallel.imap_unordered(process, batches):
                df_locations.append(df_locations_item)
                df_convergence.append(df_convergence_item)

    df_locations = pd.concat(df_locations).sort_values(by = ["person_id", "activity_index"])
    df_convergence = pd.concat(df_convergence)

    print("Success rate:", df_convergence["valid"].mean())

    return df_locations, df_convergence

def process(context, arguments):
  df_trips, df_primary, random_seed, crs = arguments

  # Set up RNG
  random = np.random.default_rng(random_seed)
  maximum_iterations = context.config("secondary_activities.maximum_iterations")

  # Set up discretization solver
  destinations = context.data("destinations")
  candidate_index = CandidateIndex(destinations)
  discretization_solver = CustomDiscretizationSolver(candidate_index)

  # Set up distance sampler
  distance_distributions = context.data("distance_distributions")
  distance_sampler = CustomDistanceSampler(
        maximum_iterations = min(1000, maximum_iterations),
        random = random,
        distributions = distance_distributions)

  # Set up relaxation solver; currently, we do not consider tail problems.
  chain_solver_option = context.config("secondary_activities.chain_solver")
  if chain_solver_option == "default":
    chain_solver = GravityChainSolver(
      random = random, eps = 10.0, lateral_deviation = 10.0, alpha = 0.1,
      maximum_iterations = min(1000, maximum_iterations))
  elif chain_solver_option == "force_model":
    grid_parameters, force_field_data = context.data("force_field_data")
    chain_solver = ForceFieldChainSolver(grid_parameters, force_field_data, 
      random = random, maximum_iterations = min(1000, maximum_iterations))

  tail_solver = AngularTailSolver(random = random)
  free_solver = CustomFreeChainSolver(random, candidate_index)

  relaxation_solver = GeneralRelaxationSolver(chain_solver, tail_solver, free_solver)

  # Set up assignment solver
  thresholds = dict(
    car = 200.0, car_passenger = 200.0, pt = 200.0,
    bike = 100.0, walk = 100.0
  )
  thresholds.update({
    "%s_loop" % mode: threshold for mode, threshold in thresholds.items()
  })

  assignment_objective = DiscretizationErrorObjective(thresholds = thresholds)
  solver_arguments = dict(
      distance_sampler = distance_sampler,
      relaxation_solver = relaxation_solver,
      discretization_solver = discretization_solver,
      objective = assignment_objective,
      maximum_iterations = min(20, maximum_iterations)
      )

  lake_polygons = context.data("lake_polygons")

  if context.config("secondary_activities.border_penalty") or lake_polygons is not None:
    anchors_in_ch = {
      person_id: dict(home = False, work = work_in_ch, education = education_in_ch, border_acceptance = border_acceptance)
      for person_id, work_in_ch, education_in_ch, border_acceptance
      in df_primary[["person_id", "work_in_ch", "education_in_ch", "border_acceptance"]].itertuples(index = False)
    }
    assignment_solver = BorderPenaltyAssignmentSolver(
      random = random,
      swiss_identifiers = context.data("swiss_identifiers"), anchors_in_ch = anchors_in_ch,
      lake_barrier = None if lake_polygons is None else LakeBarrier(lake_polygons),
      **solver_arguments)
  else:
    assignment_solver = AssignmentSolver(**solver_arguments)

  df_locations = []
  df_convergence = []

  last_person_id = None

  for problem in find_assignment_problems(df_trips, df_primary):
      result = assignment_solver.solve(problem)
      apply_loop_constraints(problem, result)

      starting_activity_index = problem["activity_index"]

      for index, (identifier, location) in enumerate(zip(result["discretization"]["identifiers"], result["discretization"]["locations"])):
          df_locations.append((
              problem["person_id"], starting_activity_index + index, identifier, geo.Point(location)
          ))

      df_convergence.append((
          result["valid"], problem["size"], result["iterations"], result["relaxation"]["iterations"], problem["person_id"]
      ))

      if problem["person_id"] != last_person_id:
          last_person_id = problem["person_id"]
          context.progress.update()

  df_locations = pd.DataFrame.from_records(df_locations, columns = ["person_id", "activity_index", "location_id", "geometry"])
  df_locations = gpd.GeoDataFrame(df_locations, crs = crs)
  assert not df_locations["geometry"].isna().any()

  df_convergence = pd.DataFrame.from_records(df_convergence, columns = ["valid", "size", "assignment_iterations", "relaxation_iterations", "person_id"])
  return df_locations, df_convergence
