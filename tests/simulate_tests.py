import ZODB

from boolsi.constants import NodeStateRange
from boolsi.testing_tools import build_predecessor_nodes_lists_and_truth_tables, \
    count_simulation_problems, configure_encode_and_simulate, generate_test_description, \
    UPDATE_RULES_A, UPDATE_RULES_B
from boolsi.simulate import Simulation, init_simulation_db_structure, simulate_master, simulate_until_max_t, store_simulation
from boolsi.mpi import MPICommWrapper


def test_simulate_master_A():
    """
    `simulate_master`
    Feature A: performing simulations irrespectively of performance tuning.
    """
    predecessor_node_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, True, False, False, False]
    fixed_nodes = dict()
    perturbed_nodes_by_t = dict()
    max_t = 101
    initial_state_variations = []
    fixed_nodes_variations = []
    perturbed_nodes_by_t_variations = [(40, 0, NodeStateRange.MAYBE_TRUE)]
    n_simulation_problems = count_simulation_problems(
        initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations)
    # Test for {single batch per process, multiple batches per process}.
    n_simulation_problem_batches_per_process_1 = 1
    n_simulation_problem_batches_per_process_2 = 5

    expected_simulation_states_1 = \
        [initial_state] + 25 * [[True, False, False, True, True, False],
                                [True, True, False, False, True, True],
                                [False, True, False, False, False, True],
                                [False, False, True, False, False, False]] + \
        [[True, False, False, True, True, False]]
    expected_simulation_1 = Simulation(expected_simulation_states_1, dict(), dict())
    expected_simulation_states_2 = \
        expected_simulation_states_1[:40] + [[True, False, True, False, False, False],
                                             [True, True, False, True, False, False]] + \
        60 * [[True, True, False, False, False, False]]
    expected_simulation_2 = Simulation(expected_simulation_states_2, dict(), {40: {0: True}})
    expected_simulations = [expected_simulation_1, expected_simulation_2]

    for n_simulation_problem_batches_per_process in [n_simulation_problem_batches_per_process_1,
                                                     n_simulation_problem_batches_per_process_2]:
        db_conn = ZODB.connection(None)
        init_simulation_db_structure(db_conn)
        simulate_master(
            MPICommWrapper(), n_simulation_problem_batches_per_process,
            (initial_state, fixed_nodes, perturbed_nodes_by_t),
            (initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations),
            predecessor_node_lists, truth_tables, max_t, n_simulation_problems, db_conn, None)

        test_description = generate_test_description(
            locals(), 'n_simulation_problem_batches_per_process')
        assert list(db_conn.root.simulations.values()) == expected_simulations, test_description
        assert db_conn.root.n_simulations() == len(expected_simulations), test_description


def test_simulate_A():
    """
    `simulate_until_max_t`
    Feature A: simulating until the time cap.
    """
    predecessor_node_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_A)
    initial_state = [False, True, True, False, False]
    perturbed_nodes_by_t = dict()
    # Test for {not reaching attractor, reaching attractor}.
    max_t_1 = 3
    max_t_2 = 20
    expected_simulation_states_1 = \
        [initial_state, [False, False, True, False, True], [True, False, False, False, True],
         [True, True, False, True, True]]
    expected_simulation_states_2 = \
        4 * [initial_state, [False, False, True, False, True], [True, False, False, False, True],
             [True, True, False, True, True], [True, True, True, True, False]] + \
        [initial_state]

    for max_t, expected_simulation_states in zip(
            [max_t_1, max_t_2], [expected_simulation_states_1, expected_simulation_states_2]):
        _, _simulate_until_attractor_or_target_substate_or_max_t = \
            configure_encode_and_simulate(max_t=max_t)
        simulation_states = simulate_until_max_t(
            max_t, _simulate_until_attractor_or_target_substate_or_max_t, initial_state,
            perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(locals(), 'max_t')
        assert expected_simulation_states == simulation_states, test_description

