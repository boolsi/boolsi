from itertools import product
from functools import partial
from math import inf

from boolsi.model import majority, apply_update_rules, adjust_update_rules_for_fixed_nodes, \
    simulate_step, simulate_n_steps, encode_state, \
    simulate_until_attractor_or_target_substate_or_max_t
from boolsi.testing_tools import generate_test_description, \
    build_predecessor_nodes_lists_and_truth_tables, UPDATE_RULES_A, UPDATE_RULES_B


def test_majority_A():
    """
    `majority`
    Feature A: evaluating output correctly.
    """
    arguments_1 = [False, True, False]
    arguments_2 = [False, True, True]
    arguments_3 = [False, True]

    expected_outcome_1 = False
    expected_outcome_2 = True
    expected_outcome_3 = False

    for arguments, expected_outcome in zip(
            [arguments_1, arguments_2, arguments_3],
            [expected_outcome_1, expected_outcome_2, expected_outcome_3]):
        outcome = majority(*arguments)

        test_description = generate_test_description(locals(), 'arguments')
        assert expected_outcome == outcome, test_description


def test_apply_update_rules_A():
    """
    `apply_update_rules`
    Feature A: properly hashing predecessor node states and updating
    network state correctly as a result.
    """
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    current_state = [True, False, True, False, False, False]

    expected_next_state = [True, True, False, True, False, False]

    next_state = apply_update_rules(current_state, predecessor_nodes_lists, truth_tables)

    assert expected_next_state == next_state


def test_adjust_update_rules_to_fixed_nodes_A():
    """
    `adjust_update_rules_to_fixed_nodes`
    Feature A: properly adjusting truth tables for fixed nodes.
    """
    update_rule_dict = {'A': 'not A'}
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(update_rule_dict)
    fixed_node_state_1 = False
    fixed_node_state_2 = False

    expected_adjusted_predecessor_node_lists = [[]]

    for fixed_node_state in [fixed_node_state_1, fixed_node_state_2]:
        fixed_nodes = {0: fixed_node_state}

        expected_adjusted_truth_tables = [{(): fixed_node_state}]

        adjusted_predecessor_node_lists, adjusted_truth_tables = \
            adjust_update_rules_for_fixed_nodes(predecessor_nodes_lists, truth_tables, fixed_nodes)

        test_description = generate_test_description(locals(), 'fixed_node_state')
        assert expected_adjusted_predecessor_node_lists == adjusted_predecessor_node_lists, \
            test_description
        assert expected_adjusted_truth_tables == adjusted_truth_tables, test_description


def test_simulate_time_step_A():
    """
    `simulate_time_step`
    Feature A: evaluating next state.
    """
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)

    current_state = [False, False, False, False, False, False]
    # Test for {no perturbations, perturbations}.
    current_perturbations_1 = dict()
    current_perturbations_2 = {1: True}

    for current_perturbations in [current_perturbations_1, current_perturbations_2]:
        expected_next_state = [True, bool(current_perturbations), False, False, True, False]

        next_state = simulate_step(
            current_state, predecessor_nodes_lists, truth_tables, current_perturbations)

        test_description = generate_test_description(locals(), 'current_perturbations')
        assert expected_next_state == next_state


def test_simulate_n_steps_A():
    """
    `simulate_n_steps`
    Feature A: evaluating next n states.
    """
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    n_steps = 3
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {1: True}}

    for perturbed_nodes_by_t in [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]:
        expected_states = [initial_state,
                           [True, bool(perturbed_nodes_by_t), False, False, True, False],
                           [True, True, False, False, False, True],
                           [False, True, False, False, False, False]]

        states = simulate_n_steps(
            initial_state, perturbed_nodes_by_t, predecessor_nodes_lists, truth_tables, n_steps)

        test_description = generate_test_description(locals(), 'perturbed_nodes_by_t')
        assert expected_states == states


def test_encode_state_A():
    """
    `encode_state`
    Feature A: injectivity.
    """
    n_nodes = 20
    n_substate_nodes = 10
    substate_nodes = set(range(2 * n_substate_nodes)[::2])
    states_codes = set()
    substates_codes = set()
    substates_tuples = set()

    for state_tuple in product([False, True], repeat=n_nodes):
        state_code, substate_code = encode_state(substate_nodes, list(state_tuple))
        substate_tuple = tuple(state_tuple[node] for node in substate_nodes)
        assert state_code not in states_codes
        assert (substate_code in substates_codes) == (substate_tuple in substates_tuples)
        states_codes.add(state_code)
        substates_codes.add(substate_code)
        substates_tuples.add(substate_tuple)


def test_simulate_until_max_t_or_attractor_or_target_substate_A():
    """
    `simulate_until_attractor_or_target_substate_or_max_t`
    Feature A: stopping at the time cap.
    """
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    perturbed_nodes_by_t = dict()
    max_t = 6
    # Test for {not storing all states, storing all states}.
    storing_all_states_1 = False
    storing_all_states_2 = True
    # Test for {no target substate, target substate}.
    target_node_set_1 = set()
    target_node_set_2 = {0, 1, 2, 3, 4, 5}

    expected_t = 6

    for storing_all_states, target_node_set in product([storing_all_states_1, storing_all_states_2],
                                                       [target_node_set_1, target_node_set_2]):
        _encode_state = partial(encode_state, target_node_set)
        _, target_substate_code = _encode_state([True, True, True, True, True, True])
        target_substate_code = target_substate_code or None

        _, _, t, *_ = simulate_until_attractor_or_target_substate_or_max_t(
            storing_all_states, max_t, _encode_state, target_substate_code, initial_state,
            perturbed_nodes_by_t, predecessor_nodes_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'storing_all_states', 'target_node_set')
        assert expected_t == t, test_description


def test_simulate_until_max_t_or_attractor_or_target_substate_B():
    """
    `simulate_until_attractor_or_target_substate_or_max_t`
    Feature B: finding attractor at expected time step.
    """
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    perturbed_nodes_by_t = dict()
    # Test for {not storing all states, storing all states}.
    storing_all_states_1 = False
    storing_all_states_2 = True
    # Test for {no time cap, not reaching time cap, reaching time cap}.
    n_steps_before_max_t_1 = inf
    n_steps_before_max_t_2 = 1
    n_steps_before_max_t_3 = 0
    # Test for {no target substate, target substate}.
    target_node_set_1 = set()
    target_node_set_2 = {0, 1, 2, 3, 4, 5}

    expected_attractor_is_found = True
    expected_target_substate_is_reached = False

    for storing_all_states, n_steps_before_max_t, target_node_set in product(
            [storing_all_states_1, storing_all_states_2],
            [n_steps_before_max_t_1, n_steps_before_max_t_2, n_steps_before_max_t_3],
            [target_node_set_1, target_node_set_2]):
        if storing_all_states:
            update_rules = UPDATE_RULES_B
            initial_state = [False, False, False, False, False, False]
            target_state = [True, True, False, False, True, False]
        else:
            update_rules = UPDATE_RULES_A
            initial_state = [False, True, True, False, False]
            target_state = [True, True, True, True, True]

        predecessor_nodes_lists, truth_tables = \
            build_predecessor_nodes_lists_and_truth_tables(update_rules)
        _encode_state = partial(encode_state, target_node_set)
        _, target_substate_code = _encode_state(target_state)
        target_substate_code = target_substate_code or None

        expected_t = 7 if storing_all_states else 11

        max_t = expected_t + n_steps_before_max_t
        _, _, t, attractor_is_found, target_substate_is_reached, _ = \
            simulate_until_attractor_or_target_substate_or_max_t(
                storing_all_states, max_t, _encode_state, target_substate_code, initial_state,
                perturbed_nodes_by_t, predecessor_nodes_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'storing_all_states', 'n_steps_before_max_t', 'target_node_set')
        assert expected_t == t, test_description
        assert expected_attractor_is_found == attractor_is_found, test_description
        assert expected_target_substate_is_reached == target_substate_is_reached, test_description


def test_simulate_until_max_t_or_attractor_or_target_substate_C():
    """
    `simulate_until_attractor_or_target_substate_or_max_t`
    Feature C: reaching target state at expected time step.
    """
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    perturbed_nodes_by_t = dict()
    target_node_set = {0, 1, 2, 3, 4, 5}
    # Test for {not storing all states, storing all states}.
    storing_all_states_1 = False
    storing_all_states_2 = True
    # Test for {no time cap, not reaching time cap, reaching time cap}.
    n_steps_before_max_t_1 = inf
    n_steps_before_max_t_2 = 1
    n_steps_before_max_t_3 = 0

    expected_attractor_is_found = False
    expected_target_substate_is_reached = True

    for storing_all_states, n_steps_before_max_t in product(
            [storing_all_states_1, storing_all_states_2],
            [n_steps_before_max_t_1, n_steps_before_max_t_2, n_steps_before_max_t_3]):
        if storing_all_states:
            update_rules = UPDATE_RULES_B
            initial_state = [False, False, False, False, False, False]
            target_state = [True, True, False, False, False, False]
        else:
            update_rules = UPDATE_RULES_A
            initial_state = [False, True, True, False, False]
            target_state = [True, True, True, True, False]

        predecessor_nodes_lists, truth_tables = \
            build_predecessor_nodes_lists_and_truth_tables(update_rules)
        _encode_state = partial(encode_state, target_node_set)
        _, target_substate_code = _encode_state(target_state)
        target_substate_code = target_substate_code or None

        expected_t = 6 if storing_all_states else 4

        max_t = expected_t + n_steps_before_max_t
        _, _, t, attractor_is_found, target_substate_is_reached, _ = \
            simulate_until_attractor_or_target_substate_or_max_t(
                storing_all_states, max_t, _encode_state, target_substate_code, initial_state,
                perturbed_nodes_by_t, predecessor_nodes_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'storing_all_states', 'n_steps_before_max_t')
        assert expected_t == t, test_description
        assert expected_attractor_is_found == attractor_is_found, test_description
        assert expected_target_substate_is_reached == target_substate_is_reached, test_description


def test_simulate_until_max_t_or_attractor_or_target_substate_D():
    """
     `simulate_until_attractor_or_target_substate_or_max_t`
    Feature D: simulating expected states.
    """
    predecessor_nodes_lists, truth_tables = build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_A)
    initial_state = [False, True, True, False, False]
    max_t = 1
    target_node_set = set()
    _encode_state = partial(encode_state, target_node_set)
    target_substate_code = None
    # Test for {not storing all states, storing all states}.
    storing_all_states_1 = False
    storing_all_states_2 = True
    # Test for {stopping at last perturbation, stopping not at last
    # perturbation}.
    perturbed_nodes_by_t_1 = {1: {1: True}}
    perturbed_nodes_by_t_2 = dict()

    for storing_all_states, perturbed_nodes_by_t in product(
            [storing_all_states_1, storing_all_states_2],
            [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2]):
        expected_last_state = [False, bool(perturbed_nodes_by_t), True, False, True]
        expected_states = [initial_state, expected_last_state]
        expected_state_codes_since_last_perturbation = \
            [_encode_state(state)[0]
             for state in expected_states[int(bool(perturbed_nodes_by_t)):]]
        if storing_all_states:
            expected_attractor_reference_points = None
        elif not perturbed_nodes_by_t:
            expected_states.insert(1, None)
            expected_state_codes_since_last_perturbation.insert(1, None)

        states, state_codes_since_last_perturbation, *_ = \
            simulate_until_attractor_or_target_substate_or_max_t(
                storing_all_states, max_t, _encode_state, target_substate_code, initial_state,
                perturbed_nodes_by_t, predecessor_nodes_lists, truth_tables)

        test_description = generate_test_description(
            locals(), 'storing_all_states', 'perturbed_nodes_by_t')
        assert expected_states == states, test_description
        assert expected_state_codes_since_last_perturbation == state_codes_since_last_perturbation, \
            test_description


def test_simulate_until_max_t_or_attractor_or_target_substate_E():
    """
    `simulate_until_attractor_or_target_substate_or_max_t`
    Feature E: ignoring attractors before last perturbation.
    """
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [True, True, False, True, False, False]
    perturbed_nodes_by_t = {10: {1: True}}
    max_t = inf
    target_node_set = set()
    _encode_state = partial(encode_state, target_node_set)
    target_substate_code = None
    # Test for {not storing all states, storing all states}.
    storing_all_states_1 = False
    storing_all_states_2 = True

    for storing_all_states in [storing_all_states_1, storing_all_states_2]:
        _, _, t, *_ = simulate_until_attractor_or_target_substate_or_max_t(
            storing_all_states, max_t, _encode_state, target_substate_code, initial_state,
            perturbed_nodes_by_t, predecessor_nodes_lists, truth_tables)

        test_description = generate_test_description(locals(), 'storing_all_states')
        assert t > 10, test_description


def test_simulate_until_max_t_or_attractor_or_target_substate_F():
    """
    `simulate_until_attractor_or_target_substate_or_max_t`
    Feature F: ignoring target substates before last perturbation.
    """
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    perturbed_nodes_by_t = {2: {1: True}}
    max_t = inf
    target_node_set = {0, 1, 2, 3, 4, 5}
    target_state = [True, False, False, False, True, False]
    _encode_state = partial(encode_state, target_node_set)
    _, target_substate_code = _encode_state(target_state)
    # Test for {not storing all states, storing all states}.
    storing_all_states_1 = False
    storing_all_states_2 = True

    for storing_all_states in [storing_all_states_1, storing_all_states_2]:
        _, _, t, *_ = simulate_until_attractor_or_target_substate_or_max_t(
            storing_all_states, max_t, _encode_state, target_substate_code, initial_state,
            perturbed_nodes_by_t, predecessor_nodes_lists, truth_tables)

        test_description = generate_test_description(locals(), 'storing_all_states')
        assert t >= 2, test_description


def test_simulate_until_max_t_or_attractor_or_target_substate_G():
    """
    `simulate_until_attractor_or_target_substate_or_max_t`
    Feature G: recognizing target substate in initial state.
    """
    predecessor_nodes_lists, truth_tables = \
        build_predecessor_nodes_lists_and_truth_tables(UPDATE_RULES_B)
    initial_state = [False, False, False, False, False, False]
    perturbed_nodes_by_t = dict()
    max_t = inf
    target_node_set = {1, 2, 4, 5}
    _encode_state = partial(encode_state, target_node_set)
    _, target_substate_code = _encode_state(initial_state)
    # Test for {not storing all states, storing all states}.
    storing_all_states_1 = False
    storing_all_states_2 = True

    for storing_all_states in [storing_all_states_1, storing_all_states_2]:
        _, _, t, *_ = simulate_until_attractor_or_target_substate_or_max_t(
            storing_all_states, max_t, _encode_state, target_substate_code, initial_state,
            perturbed_nodes_by_t, predecessor_nodes_lists, truth_tables)

        test_description = generate_test_description(locals(), 'storing_all_states')
        assert t == 0, test_description


def test_simulate_until_max_t_or_attractor_or_target_substate_H():
    """
    `simulate_until_attractor_or_target_substate_or_max_t`
    Feature H: storing expected reference points when looking for
    attractors.
    """
    perturbed_nodes_by_t = dict()
    max_t = inf
    target_node_set = set()
    _encode_state = partial(encode_state, target_node_set)
    target_substate_code = None
    storing_all_states = False
    # Test for {attractor detected at a reference point, attractor
    # detected not at a reference point}.
    update_rules_1 = UPDATE_RULES_A
    initial_state_1 = [True, True, True, True, True]
    expected_attractor_reference_points_dict_1 = \
        {0: initial_state_1, 3: [True, False, True, False, True],
         6: [True, True, True, False, False]}
    update_rules_2 = {'A': 'A'}
    initial_state_2 = [False]
    expected_attractor_reference_points_dict_2 = {0: initial_state_2}

    for update_rules, initial_state, expected_attractor_reference_points_dict in zip(
            [update_rules_1, update_rules_2], [initial_state_1, initial_state_2],
            [expected_attractor_reference_points_dict_1, expected_attractor_reference_points_dict_2]):
        expected_attractor_reference_points = \
            [(t, state, _encode_state(state)[0])
             for t, state in sorted(expected_attractor_reference_points_dict.items())]

        predecessor_nodes_lists, truth_tables = \
            build_predecessor_nodes_lists_and_truth_tables(update_rules)
        *_, attractor_reference_points = simulate_until_attractor_or_target_substate_or_max_t(
            storing_all_states, max_t, _encode_state, target_substate_code, initial_state,
            perturbed_nodes_by_t, predecessor_nodes_lists, truth_tables)

        test_description = generate_test_description(locals(), 'update_rules', 'initial_state')
        assert expected_attractor_reference_points == attractor_reference_points, test_description
