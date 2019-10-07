"""
Performing simulations of given length.
"""
import logging
import persistent
from functools import partial
from BTrees import OOBTree, Length

from .constants import inf
from .model import count_perturbations
from .db import batch_cache_reset
from .mpi import generic_master, generic_worker
from .utils import lcm


class Simulation(persistent.Persistent):
    """
    Persistent class for storing simulations in database.
    """
    def __init__(self, states, fixed_nodes, perturbed_nodes_by_t):
        self.states = states
        self.fixed_nodes = fixed_nodes
        self.perturbed_nodes_by_t = perturbed_nodes_by_t
        self.n_perturbations = count_perturbations(perturbed_nodes_by_t)

    def __eq__(self, simulation):
        return self.states, self.fixed_nodes, self.perturbed_nodes_by_t == \
               simulation.states, simulation.fixed_nodes, simulation.perturbed_nodes_by_t


def init_simulation_db_structure(db_conn):
    """
    Init database structure for storing simulations.

    :param db_conn: DB connection
    :return: None
    """
    db_conn.root.simulations = OOBTree.BTree()
    # To store summary statistics of simulations.
    db_conn.root.n_simulations = Length.Length()
    db_conn.root.n_simulation_batches = Length.Length()
    # To store least common multiple of sizes of simulation batches.
    db_conn.root.simulation_batch_sizes_lcm = Length.Length(1)


def simulate_master(
        mpi_comm, n_simulation_problem_batches_per_process, origin_simulation_problem,
        simulation_problem_variations, predecessor_node_lists, truth_tables, max_t,
        n_simulation_problems, db_conn, output_directory):
    """
    Top-level function of Simulate mode for master. For every combination of initial state, fixed nodes,
    and perturbations simulates for requested number of steps.

    :param mpi_comm: MPI communicator object
    :param n_simulation_problem_batches_per_process: number of batches to split single process load into
    :param origin_simulation_problem: simulation problem to start variations from
    :param simulation_problem_variations: variations of simulation problems
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :param max_t: maximum simulation time
    :param n_simulation_problems: number of simulations to perform
    :param db_conn: database connection, for storing simulations
    :param output_directory: output directory path
    :return: None
    """
    n_processes_text = '{} worker processes'.format(mpi_comm.n_workers) if \
        mpi_comm.n_workers > 0 else 'Single process'
    logging.getLogger().info('{} will be used to perform {} simulations...'.format(
        n_processes_text, n_simulation_problems))

    _simulate_until_max_t = partial(simulate_until_max_t, max_t)

    generic_master(
        mpi_comm, n_simulation_problems, n_simulation_problem_batches_per_process,
        _simulate_until_max_t, origin_simulation_problem, simulation_problem_variations,
        predecessor_node_lists, truth_tables, set(), None, store_simulation, [],
        write_simulations_to_db, db_conn, True, max_t, output_directory, inf)


def simulate_worker(mpi_comm, max_t, db_conn):
    """
    Top-level function of Simulate mode for worker.

    :param mpi_comm: MPI communicator object
    :param max_t: maximum simulation time
    :param db_conn: database connection, for storing simulations
    :return: None
    """
    _simulate_until_max_t = partial(simulate_until_max_t, max_t)

    generic_worker(
        mpi_comm, _simulate_until_max_t, store_simulation, [],
        read_simulation_batches_from_db, write_simulations_to_db, db_conn, True, max_t, inf)


def simulate_until_max_t(
        max_t, _simulate_until_attractor_or_target_substate_or_max_t, initial_state,
        perturbed_nodes_by_t, predecessor_node_lists, truth_tables):

    """
    Simulate for given number of time steps.

    :param max_t: time after which to stop the simulation
    :param _simulate_until_attractor_or_target_substate_or_max_t: [function] to simulate
        until attractor or max_t
    :param initial_state: initial network state
    :param perturbed_nodes_by_t: perturbations of network nodes
    :param predecessor_node_lists: predecessor nodes for each node
    :param truth_tables: truth tables of node update rules
    :return: simulation states
    """
    states, state_codes_since_last_perturbation, *_ = \
        _simulate_until_attractor_or_target_substate_or_max_t(
            initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)
    last_simulation_t = len(states) - 1

    if last_simulation_t < max_t:
        # Find attractor states.
        attractor_instance_first_state_code = state_codes_since_last_perturbation[-1]
        attractor_l = len(state_codes_since_last_perturbation) - \
            state_codes_since_last_perturbation.index(attractor_instance_first_state_code) - 1
        attractor_instance_states = states[-attractor_l:]
        # Deduce remaining simulation states.
        remaining_l_to_simulate = max_t - last_simulation_t
        n_attractor_instance_repetitions, n_leftover_states = divmod(remaining_l_to_simulate,
                                                                     len(attractor_instance_states))
        states.extend(attractor_instance_states * n_attractor_instance_repetitions +
                      attractor_instance_states[:n_leftover_states])

    return states


def store_simulation(states, fixed_nodes, perturbed_nodes_by_t, simulations):
    """
    Store simulation in memory.

    :param states: simulation states
    :param fixed_nodes: fixed nodes of the simulation
    :param perturbed_nodes_by_t: perturbations of the simulation
    :param simulations: [list] currently stored simulations
    :return: number of simulations added
    """
    simulations.append(Simulation(states, fixed_nodes, perturbed_nodes_by_t))

    return 1


def write_simulations_to_db(db_conn, simulations, simulation_batch_index,
                            n_simulation_results_to_find=inf):
    """
    Write simulation batch to a database on disk.

    :param db_conn: database connection
    :param simulations: simulations to write
    :param simulation_batch_index: batch to associate simulations with
    :param n_simulation_results_to_find: not to store more than this many simulations
    :return: number of simulations added to database
    """
    n_simulations_added = 0

    # No need to store more simulations than requested.
    if db_conn.root.n_simulations() + len(simulations) > n_simulation_results_to_find:
        simulations = simulations[:n_simulation_results_to_find - db_conn.root.n_simulations()]

    for simulation_index_in_batch, simulation in enumerate(simulations):
        # Write simulation to database.
        db_conn.root.simulations[(simulation_batch_index, simulation_index_in_batch)] = simulation

    if simulations:
        # Update statistics of simulations in database.
        db_conn.root.n_simulations.change(len(simulations))
        db_conn.root.n_simulation_batches.change(1)
        # Update least common multiple of simulation batch sizes.
        db_conn.root.simulation_batch_sizes_lcm.set(
            lcm(len(simulations), db_conn.root.simulation_batch_sizes_lcm()))

    return n_simulations_added


def read_simulation_batches_from_db(db_conn):
    """
    Generate simulation batches from database.

    :param db_conn: database connection
    :return: (batch index, batch)
    """
    previous_simulation_batch_index = None
    for simulation_batch_index, simulation in \
            read_bundled_simulations_from_db(db_conn, resetting_cache=False):
        if simulation_batch_index != previous_simulation_batch_index:
            if previous_simulation_batch_index is not None:

                yield previous_simulation_batch_index, simulation_batch

                del simulation_batch
                db_conn.cacheMinimize()

            simulation_batch = []
            previous_simulation_batch_index = simulation_batch_index

        simulation_batch.append(simulation)

    if previous_simulation_batch_index is not None:

        yield previous_simulation_batch_index, simulation_batch

        del simulation_batch
        db_conn.cacheMinimize()


@batch_cache_reset
def read_bundled_simulations_from_db(db_conn):
    """
    Generate simulations from database.

    :param db_conn: database connection
    :return: (batch index, simulation)
    """
    for (simulation_batch_index, _), simulation in db_conn.root.simulations.items():
        yield simulation_batch_index, simulation
