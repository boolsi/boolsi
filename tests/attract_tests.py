import ZODB
from itertools import product
from math import inf

from boolsi.testing_tools import build_predecessor_nodes_lists_and_truth_tables, count_simulation_problems, \
    generate_test_description, configure_encode_and_simulate, construct_aggregated_attractor, \
    UPDATE_RULES_A, UPDATE_RULES_B
from boolsi.attract import init_attractor_db_structure, \
    simulate_until_attractor_or_max_t_storing_all_states, \
    simulate_until_attractor_or_max_t_using_reference_points, attract_master
from boolsi.model import encode_state
from boolsi.mpi import MPICommWrapper


def test_simulate_until_attractor_or_max_t_storing_all_states_A():
    """
    `simulate_until_attractor_or_max_t_storing_all_states`
    Feature A: finding attractor.
    """
    initial_state = [False, False, True, False, False, False]
    # Test for {no time cap, time cap}.
    max_t_1 = inf
    max_t_2 = 10
    # Test for {no length constraint, length constraint}.
    max_attractor_l_1 = inf
    max_attractor_l_2 = 10
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {3: False}}

    attractor_states_1 = [[True, False, False, True, True, False],
                          [True, True, False, False, True, True],
                          [False, True, False, False, False, True],
                          [False, False, True, False, False, False]]
    attractor_states_2 = [[True, True, False, False, False, False]]
    expected_attractor_states_list = [attractor_states_1, attractor_states_2] * 4
    expected_trajectory_l_list = [0, 6] * 4

    for (max_t, max_attractor_l, perturbed_nodes_by_t), \
        (expected_attractor_states, expected_trajectory_l) in zip(
            product([max_t_1, max_t_2],
                    [max_attractor_l_1, max_attractor_l_2],
                    [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]),
        zip(expected_attractor_states_list, expected_trajectory_l_list)):

        predecessor_node_lists, truth_tables = \
            build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
        _, _simulate_until_attractor_or_target_substate_or_max_t = \
                configure_encode_and_simulate(max_t=max_t)
        expected_attractor_state_codes = \
            [encode_state(set(), state)[0] for state in expected_attractor_states]
        expected_attractor_key = min(expected_attractor_state_codes)

        attractor_key, attractor_state_codes, attractor_states, trajectory_l = \
            simulate_until_attractor_or_max_t_storing_all_states(
                max_attractor_l, _simulate_until_attractor_or_target_substate_or_max_t,
                initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'max_t', 'max_attractor_l', 'perturbed_nodes_by_t')
        assert expected_attractor_key == attractor_key, test_description
        assert expected_attractor_state_codes == attractor_state_codes, test_description
        assert expected_attractor_states == attractor_states, test_description
        assert expected_trajectory_l == trajectory_l, test_description


def test_simulate_until_attractor_or_max_t_storing_all_states_B():
    """
    `simulate_until_attractor_or_max_t_storing_all_states`
    Feature B: not finding attractor if time cap is violated.
    """
    initial_state = [False, False, True, False, False, False]
    max_t = 3
    # Test for {no length constraint, length constraint}.
    max_attractor_l_1 = inf
    max_attractor_l_2 = 100
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {3: False}}

    for max_attractor_l, perturbed_nodes_by_t in product(
            [max_attractor_l_1, max_attractor_l_2],
            [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]):
        predecessor_node_lists, truth_tables = \
            build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
        _, _simulate_until_attractor_or_target_substate_or_max_t = \
            configure_encode_and_simulate(max_t=max_t)

        attractor = simulate_until_attractor_or_max_t_storing_all_states(
            max_attractor_l, _simulate_until_attractor_or_target_substate_or_max_t,
            initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'max_attractor_l', 'perturbed_nodes_by_t')
        assert attractor is None, test_description


def test_simulate_until_attractor_or_max_t_storing_all_states_C():
    """
    `simulate_until_attractor_or_max_t_storing_all_states`
    Feature C: not finding attractor if length constraint is violated.
    """
    initial_state = [False, False, True, False, False, False]
    max_attractor_l = 3
    # Test for {no time cap, time cap}.
    max_t_1 = inf
    max_t_2 = 6
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {2: False}}

    for max_simulation_t, perturbed_nodes_by_t in product(
            [max_t_1, max_t_2],
            [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]):
        predecessor_node_lists, truth_tables = \
            build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
        _, _simulate_until_attractor_or_target_substate_or_max_t = \
                configure_encode_and_simulate(max_t=max_simulation_t)

        attractor = simulate_until_attractor_or_max_t_storing_all_states(
            max_attractor_l, _simulate_until_attractor_or_target_substate_or_max_t,
            initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'max_simulation_t', 'perturbed_nodes_by_t')
        assert attractor is None, test_description


def test_simulate_until_attractor_or_max_t_using_reference_points_A():
    """
    `simulate_until_attractor_or_max_t_using_reference_points`
    Feature A: finding attractor.
    """
    initial_state = [False, False, True, False, False, False]
    # Test for {no time cap, time cap}.
    max_simulation_t_1 = inf
    max_simulation_t_2 = 10
    # Test for {no length constraint, length constraint}.
    max_attractor_l_1 = inf
    max_attractor_l_2 = 10
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {3: False}}

    attractor_states_1 = [[True, True, False, False, True, True],
                          [False, True, False, False, False, True],
                          [False, False, True, False, False, False],
                          [True, False, False, True, True, False]]
    attractor_states_2 = [[True, True, False, False, False, False]]
    expected_attractor_states_list = [attractor_states_1, attractor_states_2] * 4
    expected_trajectory_l_list = [0, 6] * 4

    for (max_t, max_attractor_l, perturbed_nodes_by_t), \
        (expected_attractor_states, expected_trajectory_l) in zip(
        product([max_simulation_t_1, max_simulation_t_2],
                [max_attractor_l_1, max_attractor_l_2],
                [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]),
        zip(expected_attractor_states_list, expected_trajectory_l_list)):

        predecessor_node_lists, truth_tables = \
            build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
        _encode_state, _simulate_until_attractor_or_target_substate_or_max_t = \
            configure_encode_and_simulate(storing_all_states=False)
        expected_attractor_state_codes = \
            [encode_state(set(), state)[0] for state in expected_attractor_states]
        expected_attractor_key = min(expected_attractor_state_codes)

        attractor_key, attractor_state_codes, attractor_states, trajectory_l = \
            simulate_until_attractor_or_max_t_using_reference_points(
                max_t, max_attractor_l, _encode_state, _simulate_until_attractor_or_target_substate_or_max_t,
                initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'max_t', 'max_attractor_l', 'perturbed_nodes_by_t')
        assert expected_attractor_key == attractor_key, test_description
        assert expected_attractor_state_codes == attractor_state_codes, test_description
        assert expected_attractor_states == attractor_states, test_description
        assert expected_trajectory_l == trajectory_l, test_description


def test_simulate_until_attractor_or_max_t_using_reference_points_B():
    """
    `simulate_until_attractor_or_max_t_using_reference_points`
    Feature B: not finding attractor if time cap is violated.
    """
    initial_state = [False, False, True, False, False, False]
    max_t = 3
    # Test for {no length constraint, length constraint}.
    max_attractor_l_1 = inf
    max_attractor_l_2 = 100
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {3: False}}

    for max_attractor_l, perturbed_nodes_by_t in product(
            [max_attractor_l_1, max_attractor_l_2],
            [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]):
        predecessor_node_lists, truth_tables = \
            build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
        _encode_state, _simulate_until_attractor_or_target_substate_or_max_t = \
            configure_encode_and_simulate(storing_all_states=False)

        attractor = simulate_until_attractor_or_max_t_using_reference_points(
            max_t, max_attractor_l, _encode_state, _simulate_until_attractor_or_target_substate_or_max_t,
            initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'max_attractor_l', 'perturbed_nodes_by_t')
        assert attractor is None, test_description


def test_simulate_until_attractor_or_max_t_using_reference_points_C():
    """
    `simulate_until_attractor_or_max_t_using_reference_points`
    Feature C: not finding attractor if length constraint is violated.
    """
    initial_state = [False, False, True, False, False, False]
    max_attractor_l = 3
    # Test for {no time cap, time cap}.
    max_t_1 = inf
    max_t_2 = 6
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {2: False}}

    for max_t, perturbed_nodes_by_t in product(
            [max_t_1, max_t_2], [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]):
        predecessor_node_lists, truth_tables = \
            build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
        _encode_state, _simulate_until_attractor_or_target_substate_or_max_t = \
            configure_encode_and_simulate(storing_all_states=False)

        attractor = simulate_until_attractor_or_max_t_using_reference_points(
            max_t, max_attractor_l, _encode_state, _simulate_until_attractor_or_target_substate_or_max_t,
            initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'max_t', 'perturbed_nodes_by_t')
        assert attractor is None, test_description


def test_attract_master_A():
    """
    `attract_master`
    Feature A: finding attractors irrespectively of performance tuning.
    """
    predecessor_node_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_A)
    initial_state = [False, False, False, False, False]
    max_t = inf
    max_attractor_l = inf
    initial_state_variations = [0, 1, 2, 3, 4]
    fixed_nodes_variations = []
    perturbed_nodes_by_t_variations = []
    fixed_nodes = {2: True}
    perturbed_nodes_by_t = {5: {2: True}, 1000: {2: False}}
    n_simulation_problems = count_simulation_problems(
        initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations)
    # Test for {single batch per process, multiple batches per process}.
    n_simulation_problem_batches_per_process_1 = 1
    n_simulation_problem_batches_per_process_2 = 5
    # Test for {not storing all states, storing all states}.
    storing_all_states_1 = False
    storing_all_states_2 = True
    # Test for {not packing DB, packing DB}.
    packing_db_1 = False
    packing_db_2 = True

    expected_attractors = \
        {(-32, 23): construct_aggregated_attractor([[True, True, True, False, True]], 32, 1005, 0)}
    expected_total_frequency = sum(attractor.frequency for attractor in expected_attractors.values())

    for n_simulation_problem_batches_per_process, storing_all_states, packing_db in product(
            [n_simulation_problem_batches_per_process_1, n_simulation_problem_batches_per_process_2],
            [storing_all_states_1, storing_all_states_2], [packing_db_1, packing_db_2]):
        db_conn = ZODB.connection(None)
        init_attractor_db_structure(db_conn)
        attract_master(
            MPICommWrapper(), n_simulation_problem_batches_per_process,
            (initial_state, fixed_nodes, perturbed_nodes_by_t), (initial_state_variations, [], []),
            predecessor_node_lists, truth_tables, max_t, max_attractor_l,
            n_simulation_problems, storing_all_states, db_conn, packing_db, None)

        test_description = generate_test_description(
            locals(), 'n_simulation_problem_batches_per_process', 'storing_all_states')
        assert dict(db_conn.root.aggregated_attractors.items()) == expected_attractors, \
            test_description
        assert db_conn.root.n_aggregated_attractors() == len(expected_attractors), test_description
        assert db_conn.root.total_frequency() == expected_total_frequency, test_description
