"""
Finding transitions.
"""
import logging
from functools import partial

from .constants import inf
from .simulate import store_simulation, read_simulation_batches_from_db, \
    write_simulations_to_db
from .mpi import generic_master, generic_worker


def target_master(mpi_comm,
                  n_simulation_problem_batches_per_process,
                  origin_simulation_problem,
                  simulation_problem_variations,
                  target_substate_code,
                  target_node_set,
                  predecessor_node_lists,
                  truth_tables,
                  n_simulations_to_reach_target_substate,
                  max_t,
                  n_simulation_problems,
                  db_conn,
                  output_dirpath):
    """
    Top-level function of Target mode for master. Returns the states of
    simulation only if it reaches target substate (after all the perturbations).

    :param mpi_comm: MPI communicator object
    :param n_simulation_problem_batches_per_process: number of batches to split single process load into
    :param origin_simulation_problem: simulation problem to start variations from
    :param simulation_problem_variations: variations of simulation problems
    :param target_substate_code: code of the substate to reach
    :param target_node_set: nodes of the substate to reach
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states, value: resulting node state)
    :param n_simulations_to_reach_target_substate: stop searching after this many simulations
        that reached target substate
    :param max_t: maximum number of time steps to simulate for
    :param n_simulation_problems: total number of combinations
    :param db_conn: database connection, for storing simulations
    :return: number of simulations that reached target substate
    """
    if n_simulations_to_reach_target_substate is not inf and \
            n_simulations_to_reach_target_substate > n_simulation_problems:
        logging.getLogger().warning(
            'Requested {} simulations that reach target state, but only {} simulation '
            'problems provided. Will look for all simulations that reach target state.'.format(
                n_simulations_to_reach_target_substate, n_simulation_problems))
        n_simulations_to_reach_target_substate = inf

    n_processes_text = '{} worker processes'.format(mpi_comm.n_workers) if \
        mpi_comm.n_workers > 0 else 'Single process'
    n_simulations_to_reach_target_substate_text = \
        'all' if n_simulations_to_reach_target_substate is inf else n_simulations_to_reach_target_substate

    logging.getLogger().info(
        '{} will be used to perform {} simulations and find {} that reach target states.'.format(
            n_processes_text, n_simulation_problems, n_simulations_to_reach_target_substate_text))

    _write_simulations_to_db = partial(
        write_simulations_to_db, n_simulation_results_to_find=n_simulations_to_reach_target_substate)

    generic_master(
        mpi_comm, n_simulation_problems, n_simulation_problem_batches_per_process,
        simulate_until_target_substate_or_max_t, origin_simulation_problem,
        simulation_problem_variations, predecessor_node_lists, truth_tables, target_node_set,
        target_substate_code, store_simulation, [], _write_simulations_to_db, db_conn, True, max_t,
        output_dirpath, n_simulations_to_reach_target_substate)

    if n_simulations_to_reach_target_substate is inf:
        n_simulations_to_reach_target_substate = n_simulation_problems

    if db_conn.root.n_simulations() > 0:
        if db_conn.root.n_simulations() < n_simulations_to_reach_target_substate:
            simulation_word = 'Only'
        elif n_simulations_to_reach_target_substate < n_simulation_problems:
            simulation_word = 'At least'
        else:
            simulation_word = 'All'
        n_simulations_text = '{} {} simulations'.format(simulation_word, db_conn.root.n_simulations())
    else:
        n_simulations_text = "No simulations"

    logging.getLogger().info("{} reach target state.".format(n_simulations_text))

    return db_conn.root.n_simulations()


def target_worker(mpi_comm, max_t, n_simulations_to_reach_target_substate, db_conn):
    """
    Top-level function of Target mode for worker.

    :param mpi_comm: MPI communicator object
    :param max_t: maximum number of time steps to simulate for
    :param db_conn: DB connection
    :return: None
    """
    _write_simulations_to_db = partial(
        write_simulations_to_db, n_simulation_results_to_find=n_simulations_to_reach_target_substate)

    generic_worker(
        mpi_comm, simulate_until_target_substate_or_max_t, store_simulation, [],
        read_simulation_batches_from_db, _write_simulations_to_db, db_conn, True, max_t,
        n_simulations_to_reach_target_substate)


def simulate_until_target_substate_or_max_t(
        _simulate_until_attractor_or_target_substate_or_max_t, initial_state, perturbed_nodes_by_t,
        predecessor_node_lists, truth_tables):

    """
    Perform simulation to figure whether it reaches target substate.
    Does not return states of simulations that don't reach target substate.

    Target substate is not considered as reached until all the
    perturbations are carried out. Initial state can be considered as
    reached target substate if no perturbations are present.

    :param _simulate_until_attractor_or_target_substate_or_max_t: [function] to perform simulation
    :param initial_state: initial state of the network
    :param perturbed_nodes_by_t: dict (by time steps) of dicts (by nodes) of node states
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states, value: resulting node state)
    :return: list of states where last state contains target substate,
        or None if target substate was not reached
    """

    states, *_, target_substate_is_reached, _ = _simulate_until_attractor_or_target_substate_or_max_t(
        initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

    return states if target_substate_is_reached else None
