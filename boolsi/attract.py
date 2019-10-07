"""
Finding attractors.
"""
import logging
from sys import getsizeof
import persistent
import transaction
from BTrees import OOBTree, Length
from functools import partial

from .constants import inf
from .model import simulate_step, encode_state
from .db import batch_cache_reset
from .mpi import generic_master, generic_worker
from .utils import TickingProgressBar, lcm


class AggregatedAttractor(persistent.Persistent):
    """
    Persistent class for storing attractors in database.
    """
    def __init__(self, min_state_code, state_codes, states, trajectory_l):
        min_state_code_index = state_codes.index(min_state_code)
        # Rotate states to put key-inducing state first.
        self.states = states[min_state_code_index:] + states[:min_state_code_index]
        self.frequency = 1
        self.trajectory_l_mean = trajectory_l
        self.trajectory_l_variation_sum = 0

    def __eq__(self, aggregated_attractor):
        return self.states, self.frequency, self.trajectory_l_mean, self.trajectory_l_variation_sum == \
               aggregated_attractor.states, aggregated_attractor.frequency, \
               aggregated_attractor.trajectory_l_mean, aggregated_attractor.trajectory_l_variation_sum

    def update(self, frequency, trajectory_l_mean, trajectory_l_variation_sum):
        new_frequency = self.frequency + frequency
        # Use parallel one-pass algorithm to update trajectory length
        # mean and variance sum.
        trajectory_l_mean_delta = trajectory_l_mean - self.trajectory_l_mean
        self.trajectory_l_mean += trajectory_l_mean_delta * frequency / new_frequency
        self.trajectory_l_variation_sum += \
            trajectory_l_variation_sum + trajectory_l_mean_delta ** 2 * \
            self.frequency * frequency / new_frequency
        # Update frequency.
        self.frequency = new_frequency


def init_attractor_db_structure(db_conn):
    """
    Init database structure for storing aggregated attractors.

    :param db_conn: database connection
    :return None
    """
    db_conn.root.aggregated_attractors = OOBTree.BTree()
    db_conn.root.aggregated_attractor_keys_by_batch_index = OOBTree.BTree()
    db_conn.root.sorted_aggregated_attractors = OOBTree.BTree()
    # To store summary statistics of attractors.
    db_conn.root.n_aggregated_attractors = Length.Length()
    db_conn.root.total_frequency = Length.Length()
    db_conn.root.n_aggregated_attractor_batches = Length.Length()
    # To store least common multiple of sizes of aggregated attractor
    # batches.
    db_conn.root.aggregated_attractor_batch_sizes_lcm = Length.Length(1)


def attract_master(
        mpi_comm, n_simulation_problem_batches_per_process, origin_simulation_problem,
        simulation_problem_variations, predecessor_node_lists, truth_tables, max_t,
        max_attractor_l, n_simulation_problems, storing_all_states, db_conn, packing_db,
        output_dirpath):
    """
    Top-level function of Attract mode for master. Performs simulations for every
    combination of initial state, fixed nodes and perturbations until attractor is
    found or the limit of steps reached (if given).

    :param mpi_comm: MPI communicator object
    :param n_simulation_problem_batches_per_process: number of batches
        to split single process load into
    :param origin_simulation_problem: simulation problem to start variations from
    :param simulation_problem_variations: variations of simulation problems
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :param max_t: maximum number of time steps to simulate for
    :param max_attractor_l: maximum length of attractor to search for
    :param n_simulation_problems: total number of combinations
    :param storing_all_states: whether to store all previous states
        to speed up attractor detection
    :param db_conn: database connection to store aggregated attractors
    :param packing_db: whether to pack DB on each write
    :param output_dirpath: output directory path
    :return number of aggregated attractors in database
    """
    n_processes_text = '{} worker processes'.format(mpi_comm.n_workers) if \
        mpi_comm.n_workers > 0 else 'Single process'
    logging.getLogger().info(
        '{} will be used to find attractors from {} initial conditions...'.format(
            n_processes_text, n_simulation_problems))

    if storing_all_states:
        simulate_until_attractor_or_max_t = partial(
            simulate_until_attractor_or_max_t_storing_all_states, max_attractor_l)
    else:
        simulate_until_attractor_or_max_t = partial(
            simulate_until_attractor_or_max_t_using_reference_points,
            max_t, max_attractor_l, partial(encode_state, set()))

    _write_aggregated_attractors_to_db = partial(write_aggregated_attractors_to_db, packing_db)

    generic_master(
        mpi_comm, n_simulation_problems, n_simulation_problem_batches_per_process,
        simulate_until_attractor_or_max_t, origin_simulation_problem,
        simulation_problem_variations, predecessor_node_lists, truth_tables, set(), None,
        store_attractor, dict(), _write_aggregated_attractors_to_db, db_conn, storing_all_states,
        max_t, output_dirpath, inf)

    aggregated_attractors_found_text_parts = \
        ["Found {} attractors.".format(db_conn.root.n_aggregated_attractors())]

    if db_conn.root.total_frequency() < n_simulation_problems:
        no_attractor_text_parts = ['No attractor']
        if max_attractor_l < inf:
            no_attractor_text_parts.append('of length {} or less'.format(max_attractor_l))
        no_attractor_text_parts.append('can be')
        if max_t < inf:
            no_attractor_text_parts.append('detected in')
            if max_t == 1:
                max_simulation_t_text = '1 time step'
            else:
                max_simulation_t_text = '{} or less time steps'.format(max_t)
            no_attractor_text_parts.append(max_simulation_t_text)
        else:
            no_attractor_text_parts.append('reached')
        no_attractor_text_parts.append('from {:.2%} initial conditions.'.format(
            1 - db_conn.root.total_frequency() / n_simulation_problems))
        aggregated_attractors_found_text_parts.append(' '.join(no_attractor_text_parts))
    logging.getLogger().info(' '.join(aggregated_attractors_found_text_parts))

    if db_conn.root.n_aggregated_attractors() > 0:
        # Sort attractors by their basin of attraction sizes, descending,
        # and find biggest sizes of individual attractor and batch
        # attractors, to avoid running out of memory when reassigning
        # sorted attractors to batches.
        max_memory_for_aggregated_attractor = 0
        max_memory_for_aggregated_attractor_batch = 0
        total_memory_for_aggregated_attractor_batches = 0
        previous_aggregated_attractor_batch_index = None
        with TickingProgressBar((db_conn.root.n_aggregated_attractors(),
                                 db_conn.root.n_aggregated_attractor_batches() *
                                 db_conn.root.aggregated_attractor_batch_sizes_lcm()),
                                1, output_dirpath, mpi_comm.n_workers == 0,
                                stage_label='Sorting attractors...') as progressbar:
            for aggregated_attractor_batch_index, (aggregated_attractor_key, aggregated_attractor) \
                    in read_bundled_aggregated_attractors_from_db(db_conn, resetting_cache=False):
                if aggregated_attractor_batch_index != previous_aggregated_attractor_batch_index:
                    if previous_aggregated_attractor_batch_index is not None:
                        total_memory_for_aggregated_attractor_batches += memory_for_aggregated_attractor_batch
                        max_memory_for_aggregated_attractor_batch = max(
                            memory_for_aggregated_attractor_batch, max_memory_for_aggregated_attractor_batch)
                        transaction.commit()

                    previous_aggregated_attractor_batch_index = aggregated_attractor_batch_index
                    aggregated_attractor_batch_progress = \
                        db_conn.root.aggregated_attractor_batch_sizes_lcm() // len(
                            db_conn.root.aggregated_attractor_keys_by_batch_index[
                                aggregated_attractor_batch_index])
                    memory_for_aggregated_attractor_batch = 0

                aggregated_attractor_key_for_sorting = (-aggregated_attractor.frequency,
                                                        aggregated_attractor_key)
                db_conn.root.sorted_aggregated_attractors[aggregated_attractor_key_for_sorting] = \
                    aggregated_attractor
                memory_for_aggregated_attractor = calculate_memory_for_aggregated_attractor(
                    aggregated_attractor, aggregated_attractor_key_for_sorting)
                if memory_for_aggregated_attractor > max_memory_for_aggregated_attractor:
                    max_memory_for_aggregated_attractor = memory_for_aggregated_attractor
                memory_for_aggregated_attractor_batch += memory_for_aggregated_attractor
                progressbar.update((1, aggregated_attractor_batch_progress))

            if previous_aggregated_attractor_batch_index is not None:
                total_memory_for_aggregated_attractor_batches += memory_for_aggregated_attractor_batch
                max_memory_for_aggregated_attractor_batch = max(
                    memory_for_aggregated_attractor_batch, max_memory_for_aggregated_attractor_batch)
                db_conn.root.aggregated_attractors = db_conn.root.sorted_aggregated_attractors
                transaction.commit()

        # Reassign attractors to batches to account for new order.
        db_conn.root.n_aggregated_attractor_batches.set(0)
        db_conn.root.aggregated_attractor_batch_sizes_lcm.set(1)
        db_conn.root.aggregated_attractor_keys_by_batch_index.clear()
        aggregated_attractor_batch_index = 0
        memory_for_aggregated_attractor_batch = 0
        aggregated_attractor_batch_keys = []
        with TickingProgressBar((db_conn.root.n_aggregated_attractors(),
                                 total_memory_for_aggregated_attractor_batches),
                                1, output_dirpath, mpi_comm.n_workers == 0,
                                stage_label='Restructuring attractors...') as progressbar:
            for aggregated_attractor_index, (aggregated_attractor_key, aggregated_attractor) in \
                    enumerate(db_conn.root.aggregated_attractors.items()):

                memory_for_aggregated_attractor = calculate_memory_for_aggregated_attractor(
                    aggregated_attractor, aggregated_attractor_key)
                memory_for_aggregated_attractor_batch += memory_for_aggregated_attractor
                aggregated_attractor_batch_keys.append(aggregated_attractor_key)
                if memory_for_aggregated_attractor_batch + max_memory_for_aggregated_attractor > \
                        max_memory_for_aggregated_attractor_batch:
                    db_conn.root.aggregated_attractor_keys_by_batch_index[
                        aggregated_attractor_batch_index] = aggregated_attractor_batch_keys
                    db_conn.root.n_aggregated_attractor_batches.change(1)
                    db_conn.root.aggregated_attractor_batch_sizes_lcm.set(
                        lcm(len(aggregated_attractor_batch_keys),
                            db_conn.root.aggregated_attractor_batch_sizes_lcm()))
                    aggregated_attractor_batch_index += 1
                    memory_for_aggregated_attractor_batch = 0
                    aggregated_attractor_batch_keys = []
                    transaction.commit()

                progressbar.update((1, memory_for_aggregated_attractor))

            if aggregated_attractor_batch_keys:
                db_conn.root.aggregated_attractor_keys_by_batch_index[
                    aggregated_attractor_batch_index] = aggregated_attractor_batch_keys
                db_conn.root.n_aggregated_attractor_batches.change(1)
                db_conn.root.aggregated_attractor_batch_sizes_lcm.set(
                    lcm(len(aggregated_attractor_batch_keys),
                        db_conn.root.aggregated_attractor_batch_sizes_lcm()))
                transaction.commit()

    return db_conn.root.n_aggregated_attractors()


def attract_worker(mpi_comm, max_t, max_attractor_l, storing_all_states, db_conn, packing_db):
    """
    Top-level function of Attract mode for worker.

    :param mpi_comm: MPI communicator object
    :param max_t: maximum number of time steps to simulate for
    :param max_attractor_l: maximum length of attractor to search for
    :param storing_all_states: whether to store all previous states
        to speed up attractor detection
    :param db_conn: database connection to store aggregated attractors
    :param packing_db: whether to pack DB on each write
    :return: None
    """
    if storing_all_states:
        simulate_until_attractor_or_max_t = partial(
            simulate_until_attractor_or_max_t_storing_all_states, max_attractor_l)
    else:
        simulate_until_attractor_or_max_t = partial(
            simulate_until_attractor_or_max_t_using_reference_points,
            max_t, max_attractor_l, partial(encode_state, set()))

    _write_aggregated_attractors_to_db = partial(write_aggregated_attractors_to_db, packing_db)

    generic_worker(
        mpi_comm, simulate_until_attractor_or_max_t, store_attractor, dict(),
        read_aggregated_attractor_batches_from_db, _write_aggregated_attractors_to_db,
        db_conn, storing_all_states, max_t, inf)


def simulate_until_attractor_or_max_t_storing_all_states(
        max_attractor_l, _simulate_until_attractor_or_target_substate_or_max_t, initial_state,
        perturbed_nodes_by_t, predecessor_node_lists, truth_tables):
    """
    Simulates until attractor is found (or one of optionally given constrains is exceeded).

    State is not considered as part of the attractor until all the perturbations are carried out. Initial state
    can be considered as part of the attractor only if no perturbations are present.

    :param max_attractor_l: maximum length of attractor to search for
    :param _simulate_until_attractor_or_target_substate_or_max_t: [function]
    to perform simulation accounting for the time cap (if any)
    :param initial_state: initial state of the network
    :param perturbed_nodes_by_t: dict (by time step) of dicts (by node) of
        node states
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :return attractor or None if not found
    """
    attractor = None
    states, state_codes_since_last_perturbation, _, attractor_is_found, *_ = \
        _simulate_until_attractor_or_target_substate_or_max_t(
            initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

    if attractor_is_found:
        # Find attractor states.
        attractor_first_state_code = state_codes_since_last_perturbation[-1]
        # Trajectory length is counted from when all perturbations are carried out.
        attractor_trajectory_l = state_codes_since_last_perturbation.index(attractor_first_state_code)
        attractor_l = len(state_codes_since_last_perturbation) - (attractor_trajectory_l + 1)
        # Check if attractor is compliant with length cap (if any).
        if attractor_l <= max_attractor_l:
            attractor_state_codes = state_codes_since_last_perturbation[-attractor_l:]
            attractor_min_state_code = min(attractor_state_codes)
            attractor_states = states[-attractor_l:]
            attractor_trajectory_l = len(states) - (attractor_l + 1)
            attractor = (attractor_min_state_code, attractor_state_codes,
                         attractor_states, attractor_trajectory_l)

    return attractor


def simulate_until_attractor_or_max_t_using_reference_points(
        max_t, max_attractor_l, _encode_state, _simulate_until_attractor_or_target_substate_or_max_t,
        initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables):
    """
    Simulates until attractor is found (or one of optionally given constrains is exceeded).

    State is not considered as part of the attractor until all the perturbations are carried out. Initial state
    can be considered as part of the attractor only if no perturbations are present.

    :param max_t: time after which to stop the simulation
    :param max_attractor_l: maximum length of attractor to search for
    :param _simulate_until_attractor_or_target_substate_or_max_t: [function]
        to perform simulation accounting for the time cap (if any)
    :param _encode_state: [function] to encode state and substate
    :param initial_state: initial state of the network
    :param perturbed_nodes_by_t: dict (by time step) of dicts (by node) of
        node states
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :return attractor or None if not found
    """
    attractor = None
    states, state_codes_since_last_perturbation, t, _, _, reference_points = \
        _simulate_until_attractor_or_target_substate_or_max_t(
            initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)
    # Learn attractor length.
    last_reference_t, _, _ = reference_points[-1]
    attractor_l = t - last_reference_t
    # Check if attractor is compliant with length cap (if any).
    if attractor_l <= max_attractor_l:
        state = states[-1]
        state_code = state_codes_since_last_perturbation[-1]
        # Find all states of attractor instance by simulating from one of them.
        attractor_states = [state]
        attractor_state_codes = [state_code]
        for _ in range(attractor_l - 1):
            state = simulate_step(state, predecessor_node_lists, truth_tables)
            state_code, _ = _encode_state(state)
            attractor_states.append(state)
            attractor_state_codes.append(state_code)

        # Find the last reference point that is not a part of
        # attractor, to learn its trajectory length by simulating
        # from there. If attractor has been reached already at the
        # last perturbation time, then it is its trajectory length.
        attractor_state_code_set = set(attractor_state_codes)
        try:
            t, state, state_code = next(
                (t, state, state_code) for (t, state, state_code) in reversed(reference_points)
                if state_code not in attractor_state_code_set)
        except StopIteration:
            attractor_trajectory_l, _, _ = reference_points[0]
        else:
            while state_code not in attractor_state_code_set:
                t += 1
                state = simulate_step(state, predecessor_node_lists, truth_tables)
                state_code, _ = _encode_state(state)
            attractor_trajectory_l = t

        # Check if attractor is compliant with time cap (if any).
        if attractor_trajectory_l + attractor_l <= max_t:
            attractor_min_state_code = min(attractor_state_codes)
            attractor = (attractor_min_state_code, attractor_state_codes,
                         attractor_states, attractor_trajectory_l)

    return attractor


def store_attractor(attractor, fixed_nodes, perturbed_nodes_by_t, aggregated_attractors):
    """
    Adds attractor to aggregated attractors.

    :param attractor: key, state codes, states, trajectory length
    :param fixed_nodes: fixed nodes under which the attractor was found
        (not needed, for uniformity purposes)
    :param perturbed_nodes_by_t: perturbations under which the attractor was
        found (not needed, for uniformity purposes)
    :param aggregated_attractors: [dict] currently stored attractors
    :return: number of attractors added
    """
    attractor_min_state_code, attractor_state_codes, attractor_states, attractor_trajectory_l = attractor
    try:
        # Check if this attractor was found before.
        stored_aggregated_attractor = aggregated_attractors[attractor_min_state_code]
    except KeyError:
        # Store attractor.
        aggregated_attractors[attractor_min_state_code] = AggregatedAttractor(
            attractor_min_state_code, attractor_state_codes, attractor_states, attractor_trajectory_l)

        n_aggregated_attractors_added = 1
    else:
        # Update stored attractor.
        stored_aggregated_attractor.update(1, attractor_trajectory_l, 0)

        n_aggregated_attractors_added = 0

    return n_aggregated_attractors_added


def write_aggregated_attractors_to_db(packing_db, db_conn, aggregated_attractors,
                                      aggregated_attractor_batch_index):
    """
    Write aggregated attractors from memory to database as a batch.
    Batches are used as units that are guaranteed to fit in memory.

    :param packing_db: whether to pack DB on each write
    :param db_conn: database connection
    :param aggregated_attractors: [dict] attractors to write
    :param aggregated_attractor_batch_index: index of batch with new aggregated attractors
    :return: number of aggregated attractors added to database
    """
    n_aggregated_attractors_added = 0
    total_frequency_added = 0
    # Initialize list of aggregated attractors linked to this task.
    aggregated_attractor_batch_keys = []
    # Write aggregated attractors.
    for aggregated_attractor_key, aggregated_attractor in aggregated_attractors.items():
        try:
            stored_aggregated_attractor = db_conn.root.aggregated_attractors[aggregated_attractor_key]
        except KeyError:
            # Write new aggregated attractor to database.
            db_conn.root.aggregated_attractors[aggregated_attractor_key] = aggregated_attractor
            aggregated_attractor_batch_keys.append(aggregated_attractor_key)
            n_aggregated_attractors_added += 1

        else:
            # Update aggregated attractor in database.
            stored_aggregated_attractor.update(
                aggregated_attractor.frequency, aggregated_attractor.trajectory_l_mean,
                aggregated_attractor.trajectory_l_variation_sum)

        total_frequency_added += aggregated_attractor.frequency

    # Update summary statistics of aggregated attractors in database.
    db_conn.root.total_frequency.change(total_frequency_added)
    if n_aggregated_attractors_added > 0:
        db_conn.root.n_aggregated_attractors.change(n_aggregated_attractors_added)
        # Update # of attractor batches in database.
        db_conn.root.n_aggregated_attractor_batches.change(1)
        db_conn.root.aggregated_attractor_keys_by_batch_index[aggregated_attractor_batch_index] = \
            aggregated_attractor_batch_keys
        # Update least common multiple of aggregated attractor batch sizes.
        db_conn.root.aggregated_attractor_batch_sizes_lcm.set(
            lcm(n_aggregated_attractors_added, db_conn.root.aggregated_attractor_batch_sizes_lcm()))

    # Remove stale aggregated attractors from DB.
    if packing_db:
        db_conn.db().pack()

    return n_aggregated_attractors_added


def read_aggregated_attractor_batches_from_db(db_conn):
    """
    Generate aggregated attractors from database in batches.

    :param db_conn: database connection
    :return: (batch index, aggregated attractor batch)
    """
    previous_aggregated_attractor_batch_index = None
    for aggregated_attractor_batch_index, (aggregated_attractor_key, aggregated_attractor) in \
            read_bundled_aggregated_attractors_from_db(db_conn, resetting_cache=False):
        if aggregated_attractor_batch_index != previous_aggregated_attractor_batch_index:
            # Yield attractor batch and unload it from memory.
            if previous_aggregated_attractor_batch_index is not None:

                yield previous_aggregated_attractor_batch_index, aggregated_attractor_batch

                del aggregated_attractor_batch
                db_conn.cacheMinimize()

            aggregated_attractor_batch = dict()
            previous_aggregated_attractor_batch_index = aggregated_attractor_batch_index

        aggregated_attractor_batch[aggregated_attractor_key] = aggregated_attractor

    # Yield last attractor batch and unload it from memory.
    if previous_aggregated_attractor_batch_index is not None:

        yield previous_aggregated_attractor_batch_index, aggregated_attractor_batch

        del aggregated_attractor_batch
        db_conn.cacheMinimize()


@batch_cache_reset
def read_bundled_aggregated_attractors_from_db(db_conn):
    """
    Generate aggregated attractors from database.

    :param db_conn: database connection
    :return: generated aggregated attractors
    """
    for aggregated_attractor_batch_index, aggregated_attractor_batch_keys in \
            db_conn.root.aggregated_attractor_keys_by_batch_index.items():

        for aggregated_attractor_key in aggregated_attractor_batch_keys:
            yield aggregated_attractor_batch_index, (aggregated_attractor_key, \
                  db_conn.root.aggregated_attractors[aggregated_attractor_key])


def calculate_memory_for_aggregated_attractor(aggregated_attractor, aggregated_attractor_key):
    """
    Estimate memory taken by aggregated attractor.

    :param aggregated_attractor: aggregated attractor
    :param aggregated_attractor_key: attractor key
    :return: memory size in bytes
    """
    return getsizeof(aggregated_attractor_key) + getsizeof(aggregated_attractor) + \
           getsizeof(aggregated_attractor.states) + \
           sum(getsizeof(state) for state in aggregated_attractor.states) + \
           getsizeof(aggregated_attractor.frequency) + \
           getsizeof(aggregated_attractor.trajectory_l_mean) + \
           getsizeof(aggregated_attractor.trajectory_l_variation_sum)
