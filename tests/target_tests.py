import ZODB
from itertools import product
from math import inf

from boolsi.testing_tools import build_predecessor_nodes_lists_and_truth_tables, \
    configure_encode_and_simulate, count_simulation_problems, generate_test_description, \
    UPDATE_RULES_B
from boolsi.simulate import Simulation, init_simulation_db_structure
from boolsi.target import target_master, simulate_until_target_substate_or_max_t
from boolsi.mpi import MPICommWrapper


def test_simulate_until_target_substate_or_max_t_A():
    """
    `simulate_until_target_substate_or_max_t`
    Feature A: finding target substate that is not part of attractor.
    """
    predecessor_node_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    substate_node_set = {0, 1, 2, 3, 4, 5}
    _encode_state, _ = configure_encode_and_simulate(substate_node_set=substate_node_set)
    _, target_substate_code = _encode_state([True, True, False, False, False, True])
    # Test for {no time cap, time cap}.
    max_t_1 = inf
    max_t_2 = 10
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {3: False}}

    expected_simulation_states = [initial_state] + [[True, False, False, False, True, False],
                                                    [True, True, False, False, False, True]]

    for max_t, perturbed_nodes_by_t in product(
            [max_t_1, max_t_2], [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]):
        _, _simulate_until_attractor_or_target_substate_or_max_t = \
            configure_encode_and_simulate(max_t=max_t, substate_node_set=substate_node_set,
                                          target_substate_code=target_substate_code)

        simulation_states = simulate_until_target_substate_or_max_t(
            _simulate_until_attractor_or_target_substate_or_max_t, initial_state,
            perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(locals(), 'max_t', 'perturbed_nodes_by_t')
        assert expected_simulation_states == simulation_states


def test_simulate_until_target_substate_or_max_t_B():
    """
    `simulate_until_target_substate_or_max_t`
    Feature B: finding target substate that is part of attractor.
    """
    predecessor_node_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    substate_node_set = {0, 1, 2, 3, 4, 5}
    _encode_state, _ = configure_encode_and_simulate(substate_node_set=substate_node_set)
    _, target_substate_code = _encode_state([True, True, False, False, False, False])
    # Test for {no time cap, time cap}.
    max_t_1 = inf
    max_t_2 = 10
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {3: False}}

    expected_simulation_states = [initial_state] + [[True, False, False, False, True, False],
                                                    [True, True, False, False, False, True],
                                                    [False, True, False, False, False, False],
                                                    [True, False, True, False, False, False],
                                                    [True, True, False, True, False, False],
                                                    [True, True, False, False, False, False]]

    for max_t, perturbed_nodes_by_t in product(
            [max_t_1, max_t_2], [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]):
        _, _simulate_until_attractor_or_target_substate_or_max_t = \
            configure_encode_and_simulate(max_t=max_t, substate_node_set=substate_node_set,
                                          target_substate_code=target_substate_code)

        simulation_states = simulate_until_target_substate_or_max_t(
            _simulate_until_attractor_or_target_substate_or_max_t, initial_state,
            perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(locals(), 'max_t', 'perturbed_nodes_by_t')
        assert expected_simulation_states == simulation_states


def test_simulate_until_target_substate_or_max_t_C():
    """
    `simulate_until_target_substate_or_max_t`
    Feature C: not finding target substate if time cap is violated.
    """
    predecessor_node_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    substate_node_set = {0, 1, 2, 3, 4, 5}
    _encode_state, _ = configure_encode_and_simulate(substate_node_set=substate_node_set)
    _, target_substate_code = _encode_state([True, True, False, False, False, True])
    max_t = 1
    _, _simulate_until_attractor_or_target_substate_or_max_t = \
        configure_encode_and_simulate(max_t=max_t, substate_node_set=substate_node_set,
                                      target_substate_code=target_substate_code)
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {3: False}}

    for perturbed_nodes_by_t in [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]:
        simulation_states = simulate_until_target_substate_or_max_t(
            _simulate_until_attractor_or_target_substate_or_max_t, initial_state,
            perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(locals(), 'max_t', 'perturbed_nodes_by_t')
        assert simulation_states is None


def test_simulate_until_target_substate_or_max_t_D():
    """
    `simulate_until_target_substate_or_max_t`
    Feature D: not finding target substate if attractor is reached.
    """
    predecessor_node_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    substate_node_set = {0, 1, 2, 3, 4, 5}
    _encode_state, _ = configure_encode_and_simulate(substate_node_set=substate_node_set)
    _, target_substate_code = _encode_state([True, True, True, True, True, True])
    max_t = 100
    _, _simulate_until_attractor_or_target_substate_or_max_t = \
        configure_encode_and_simulate(max_t=max_t, substate_node_set=substate_node_set,
                                      target_substate_code=target_substate_code)
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {3: False}}

    for perturbed_nodes_by_t in [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]:
        simulation_states = simulate_until_target_substate_or_max_t(
            _simulate_until_attractor_or_target_substate_or_max_t, initial_state,
            perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(locals(), 'max_t', 'perturbed_nodes_by_t')
        assert simulation_states is None


def test_target_master_A():
    """
    `target_master`
    Feature A: finding target states irrespectively of performance tuning.
    """
    predecessor_node_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    substate_node_set = {0, 1, 2, 3, 4, 5}
    _encode_state, _ = configure_encode_and_simulate(substate_node_set=substate_node_set)
    _, target_substate_code = _encode_state([True, True, False, False, False, True])
    max_t = inf
    n_simulations_to_reach_target_substate = inf
    initial_state_variations = [0, 1, 2, 3, 4, 5]
    fixed_nodes_variations = []
    perturbed_nodes_by_t_variations = []
    fixed_nodes = {0: True, 1: True, 2: False, 3: False, 4: False, 5: True}
    perturbed_nodes_by_t = dict()
    n_simulation_problems = count_simulation_problems(
        initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations)
    # Test for {single batch per process, multiple batches per process}.
    n_simulation_problem_batches_per_process_1 = 1
    n_simulation_problem_batches_per_process_2 = 5

    expected_simulations = \
        [Simulation([list(initial_state_tuple)] + [[True, True, False, False, False, True]],
                    fixed_nodes, perturbed_nodes_by_t)
         for initial_state_tuple in product([False, True], repeat=6)]

    for n_simulation_problem_batches_per_process in [n_simulation_problem_batches_per_process_1,
                                                     n_simulation_problem_batches_per_process_2]:
        db_conn = ZODB.connection(None)
        init_simulation_db_structure(db_conn)
        target_master(
            MPICommWrapper(), n_simulation_problem_batches_per_process,
            (initial_state, fixed_nodes, perturbed_nodes_by_t),
            (initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations),
            target_substate_code, substate_node_set, predecessor_node_lists, truth_tables,
            n_simulations_to_reach_target_substate, max_t, n_simulation_problems, db_conn, None)

        test_description = generate_test_description(
            locals(), 'n_simulation_problem_batches_per_process')
        assert list(db_conn.root.simulations.values()) == expected_simulations, test_description
        assert db_conn.root.n_simulations() == len(expected_simulations), test_description


def test_target_master_B():
    """
    `target_master`
    Feature B: finding no more than requested number of simulations
    reaching target state.
    """
    predecessor_node_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    substate_node_set = {0, 1, 2, 3, 4, 5}
    _encode_state, _ = configure_encode_and_simulate(substate_node_set=substate_node_set)
    _, target_substate_code = _encode_state([True, True, False, False, False, True])
    max_t = inf
    n_simulations_to_reach_target_substate = 1
    initial_state_variations = [0, 1, 2, 3, 4, 5]
    fixed_nodes_variations = []
    perturbed_nodes_by_t_variations = []
    fixed_nodes = {0: True, 1: True, 2: False, 3: False, 4: False, 5: True}
    perturbed_nodes_by_t = dict()
    n_simulation_problems = count_simulation_problems(
        initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations)
    # Test for {single batch per process, multiple batches per process}.
    n_simulation_problem_batches_per_process_1 = 1
    n_simulation_problem_batches_per_process_2 = 5

    expected_simulations = [Simulation([initial_state] + [[True, True, False, False, False, True]],
                                       fixed_nodes, perturbed_nodes_by_t)]

    for n_simulation_problem_batches_per_process in [n_simulation_problem_batches_per_process_1,
                                                     n_simulation_problem_batches_per_process_2]:
        db_conn = ZODB.connection(None)
        init_simulation_db_structure(db_conn)
        target_master(
            MPICommWrapper(), n_simulation_problem_batches_per_process,
            (initial_state, fixed_nodes, perturbed_nodes_by_t),
            (initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations),
            target_substate_code, substate_node_set, predecessor_node_lists, truth_tables,
            n_simulations_to_reach_target_substate, max_t, n_simulation_problems, db_conn, None)

        test_description = generate_test_description(
            locals(), 'n_simulation_problem_batches_per_process')
        assert list(db_conn.root.simulations.values()) == expected_simulations, test_description
        assert db_conn.root.n_simulations() == len(expected_simulations), test_description
