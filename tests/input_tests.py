import pytest
import os
from itertools import product
from heapq import heappush
from math import inf

from boolsi.constants import Mode, NodeStateRange
from boolsi.testing_tools import generate_test_description
from boolsi.input import parse_input, parse_raw_input_node_names, parse_raw_input_perturbations, \
    parse_raw_input_fixed_nodes, parse_raw_input_target_state, parse_raw_input_update_rules, \
    parse_raw_input_initial_state, parse_raw_input_time_steps, parse_input_update_rules, \
    parse_predecessor_node_names_from_update_rule, \
    build_truth_table_from_safe_update_rule, generate_safe_node_names, count_simulation_problems


def test_parse_input_A():
    """
    `parse_input`
    Feature A: parsing configuration from proper input
    """
    input_file_name = "BoolSi_test_input"
    baseline_input_text = \
        'nodes :\n - A\n - B\nupdate rules:\n A: not B\n B: A\ninitial state:\n A: 0\n B: 1\n'
    expected_config = \
        {'node names': ['A', 'B'], 'origin simulation problem': ([False, True], {}, {}),
         'simulation problem variations': ([], [], []), 'incoming node lists': [[1], [0]],
         'truth tables': [{(False, ): True, (True, ): False}, {(False, ): False, (True, ): True}],
         'target substate code': None, 'target node set': None, 'total combination count': 1}
    # Test for {no fixed nodes, fixed nodes}.
    fixed_nodes_input_text_1 = ''
    fixed_nodes_input_text_2 = 'fixed nodes:\n \n'#'fixed nodes:\n A: 0\n'
    # Test for {no perturbations, perturbations}.
    perturbations_input_text_1 = ''
    perturbations_input_text_2 = 'perturbations:\n \n'#'perturbations:\n A:\n  0: 5\n'
    # Test for {no target state, target state}.
    target_state_input_text_1 = ''
    target_state_input_text_2 = 'target state:\n A: any\n B: any\n'
    # Test for {no unknown sections, unknown sections}.
    unknown_section_input_text_1 = ''
    unknown_section_input_text_2 = 'unknown section:\n this is fine\n'

    for fixed_nodes_input_text, perturbations_input_text, target_state_input_text, \
        unknown_section_input_text in product(
            [fixed_nodes_input_text_1, fixed_nodes_input_text_2],
            [perturbations_input_text_1, perturbations_input_text_2],
            [target_state_input_text_1, target_state_input_text_2],
            [unknown_section_input_text_1, unknown_section_input_text_2]):

        input_text = baseline_input_text + fixed_nodes_input_text + perturbations_input_text + \
                     target_state_input_text + unknown_section_input_text
        with open(input_file_name, "w") as input_file:
            input_file.write(input_text)

        config = parse_input(input_file_name, 10, Mode.SIMULATE)
        os.remove(input_file_name)

        test_description = generate_test_description(
            locals(), 'fixed_nodes_input_text', 'perturbations_input_text',
            'target_state_input_text', 'unknown_section_input_text')
        assert expected_config == config


def test_parse_raw_input_node_names_A():
    """
    `parse_raw_input_node_names`
    Feature A: parsing proper node names from proper input.
    """
    raw_input_node_names = [' A', 'Beta\t', 'c']

    expected_node_names = ['A', 'Beta', 'c']

    assert expected_node_names == parse_raw_input_node_names(raw_input_node_names)


def test_parse_raw_input_node_names_B():
    """
    `parse_raw_input_node_names`
    Feature B: raising exception when no node names specified.
    """
    raw_input_node_names = []

    with pytest.raises(ValueError):
        parse_raw_input_node_names(raw_input_node_names)


def test_parse_raw_input_node_names_C():
    """
    `parse_raw_input_node_names`
    Feature C: raising exception when node names are not a list.
    """
    raw_input_node_names = ' - A\n - B'

    with pytest.raises(ValueError):
        parse_raw_input_node_names(raw_input_node_names)


def test_parse_raw_input_node_names_D():
    """
    `parse_raw_input_node_names`
    Feature D: raising exception for non-string node names.
    """
    raw_input_node_names = ['A', {'B': 'C'}]

    with pytest.raises(ValueError):
        parse_raw_input_node_names(raw_input_node_names)


def test_parse_raw_input_node_names_E():
    """
    `parse_raw_input_node_names`
    Feature E: raising exception for duplicate node names.
    """
    raw_input_node_names = ['A', 'A']

    with pytest.raises(ValueError):
        parse_raw_input_node_names(raw_input_node_names)


def test_parse_raw_input_node_names_F():
    """
    `parse_raw_input_node_names`
    Feature F: raising exception for reserved node names.
    """
    # Test for {'0', '1', 'and', 'or', 'not', 'majority'}.
    raw_input_node_names_1 = ['0']
    raw_input_node_names_2 = ['1']
    raw_input_node_names_3 = ['aNd']
    raw_input_node_names_4 = ['oR']
    raw_input_node_names_5 = ['nOt']
    raw_input_node_names_6 = ['maJority']

    for raw_input_node_names in \
            [raw_input_node_names_1, raw_input_node_names_2, raw_input_node_names_3,
             raw_input_node_names_4, raw_input_node_names_5, raw_input_node_names_6]:

        with pytest.raises(ValueError):
            parse_raw_input_node_names(raw_input_node_names)


def test_parse_raw_input_node_names_G():
    """
    `parse_raw_input_node_names`
    Feature G: raising exception for node names with forbidden symbols.
    """
    node_names = ['a', 'b']
    # Test for {' ', '\t', ',', '(', ')'}
    forbidden_symbol_1 = ' '
    forbidden_symbol_2 = '\t'
    forbidden_symbol_3 = ','
    forbidden_symbol_4 = '('
    forbidden_symbol_5 = ')'
    # Test for forbidden symbol {in the beginning, in the middle, in the end}.
    forbidden_symbol_index_1 = 0
    forbidden_symbol_index_2 = 1
    forbidden_symbol_index_3 = 2

    for forbidden_symbol, forbidden_symbol_index in product(
            [forbidden_symbol_1, forbidden_symbol_2, forbidden_symbol_3, forbidden_symbol_4,
             forbidden_symbol_5],
            [forbidden_symbol_index_1, forbidden_symbol_index_2, forbidden_symbol_index_3]):

        if forbidden_symbol.isspace() and forbidden_symbol_index != 1:
            continue

        symbols = node_names.copy()
        symbols.insert(forbidden_symbol_index, forbidden_symbol)
        raw_input_node_names = ''.join(symbols)

        test_description = generate_test_description(
            locals(), 'forbidden_symbol', 'forbidden_symbol_index')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_node_names(raw_input_node_names)


def test_parse_raw_input_node_names_H():
    """
    `parse_raw_input_node_names`
    Feature H: raising exception for missing node names.
    """
    with pytest.raises(ValueError):
        parse_raw_input_node_names(None)


def test_parse_raw_input_update_rules_A():
    """
    `parse_raw_input_update_rules`
    Feature A: parsing predecessor node lists and truth tables from
        proper input.
    """
    raw_input_update_rules = {'A': 'A'}
    node_names = ['A']

    expected_incoming_node_lists = [[0]]
    expected_truth_tables = [{(False, ): False, (True, ): True}]

    incoming_node_lists, truth_tables = parse_raw_input_update_rules(
        raw_input_update_rules, node_names)

    assert expected_incoming_node_lists == incoming_node_lists
    assert expected_truth_tables == truth_tables


def test_parse_raw_input_update_rules_B():
    """
    `parse_raw_input_update_rules`
    Feature B: raising exception when update rules are not a dict.
    """
    raw_input_update_rules = 'A'
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_update_rules(raw_input_update_rules, node_names)


def test_parse_raw_input_update_rules_C():
    """
    `parse_raw_input_update_rules`
    Feature C: raising exception for duplicate (modulo whitespace)
        node names.
    """
    raw_input_update_rules = {'A': 'A', 'A ': 'A'}
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_update_rules(raw_input_update_rules, node_names)


def test_parse_raw_input_update_rules_D():
    """
    `parse_raw_input_update_rules`
    Feature D: raising exception for missing update rules.
    """
    with pytest.raises(ValueError):
        parse_raw_input_update_rules(None, ['A'])



def test_parse_raw_input_initial_state_A():
    """
    `parse_input_raw_initial_state`
    Feature A: parsing initial state from proper input.
    """
    raw_input_initial_state = {'A': '0', 'B': 'aNy'}
    node_names = ['A', 'B']

    expected_initial_state = [False, False]
    expected_initial_variations = [1]

    initial_state, initial_variations = parse_raw_input_initial_state(
        raw_input_initial_state, node_names)

    assert expected_initial_state == initial_state
    assert expected_initial_variations == initial_variations


def test_parse_raw_input_initial_state_B():
    """
    `parse_input_raw_initial_state`
    Feature B: raising exception when initial state is not a dict.
    """
    raw_input_initial_state = ['A']
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_initial_state(raw_input_initial_state, node_names)


def test_parse_raw_input_initial_state_C():
    """
    `parse_input_raw_initial_state`
    Feature C: raising exception for duplicate (modulo whitespace) node
        names.
    """
    raw_input_initial_state = {'A': '0', ' A': '0'}
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_initial_state(raw_input_initial_state, node_names)


def test_parse_raw_input_initial_state_D():
    """
    `parse_input_raw_initial_state`
    Feature D: raising exception for unknown node names.
    """
    raw_input_initial_state = {'A': '0'}
    node_names = ['B']

    with pytest.raises(ValueError):
        parse_raw_input_initial_state(raw_input_initial_state, node_names)


def test_parse_raw_input_initial_state_E():
    """
    `parse_input_raw_initial_state`
    Feature E: raising exception for missing node names.
    """
    raw_input_initial_state = {'A': '0'}
    node_names = ['A', 'B']

    with pytest.raises(ValueError):
        parse_raw_input_initial_state(raw_input_initial_state, node_names)

def test_parse_raw_input_initial_state_F():
    """
    `parse_input_raw_initial_state`
    Feature F: raising exception when node state is not a string.
    """
    raw_input_initial_state = {'A': ['0']}
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_initial_state(raw_input_initial_state, node_names)


def test_parse_raw_input_initial_state_G():
    """
    `parse_input_raw_initial_state`
    Feature G: raising exception for node state with '?'.
    """
    node_names = ['A']
    # Test for {'0?', '1?', 'any?'}
    node_state_1 = '0?'
    node_state_2 = '1?'
    node_state_3 = 'any?'

    for node_state in [node_state_1, node_state_2, node_state_3]:
        raw_input_initial_state = {'A': node_state_1}

        test_description = generate_test_description(locals(), 'node_state')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_initial_state(raw_input_initial_state, node_names)


def test_parse_raw_input_initial_state_H():
    """
    `parse_input_raw_initial_state`
    Feature G: raising exception for unrecognized node state.
    """
    raw_input_initial_state = {'A': 'a'}
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_initial_state(raw_input_initial_state, node_names)


def test_parse_raw_input_initial_state_I():
    """
    `parse_raw_input_initial_state`
    Feature I: raising exception for missing initial state.
    """
    with pytest.raises(ValueError):
        parse_raw_input_initial_state(None, ['A'])


def test_parse_raw_input_perturbations_A():
    """
    `parse_raw_input_perturbations`
    Feature A: parsing perturbations from proper input
    """
    node_names = ['A']
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET
    # Test for {no simulation time cap, simulation time cap}.
    max_t_1 = inf
    max_t_2 = 15
    # Test for {no '0' perturbations, '0' perturbations}.
    zero_node_state_perturbations_kvpairs_1 = []
    zero_node_state_perturbations_kvpairs_2 = [('0', '1')]
    # Test for {no '1' perturbations, '1' perturbations}.
    one_node_state_perturbations_kvpairs_1 = []
    one_node_state_perturbations_kvpairs_2 = [('1', '2')]
    # Test for {no '0?' perturbations, '0?' perturbations}.
    maybe_zero_node_state_perturbations_kvpairs_1 = []
    maybe_zero_node_state_perturbations_kvpairs_2 = [('0?', '3')]
    # Test for {no '1?' perturbations, '1?' perturbations}.
    maybe_one_node_state_perturbations_kvpairs_1 = []
    maybe_one_node_state_perturbations_kvpairs_2 = [('1?', '4')]
    # Test for {no 'any' perturbations, 'any' perturbations}.
    any_node_state_perturbations_kvpairs_1 = []
    any_node_state_perturbations_kvpairs_2 = [('any', '5')]
    # Test for {no 'any' perturbations, 'any?' perturbations}.
    maybe_any_node_state_perturbations_kvpairs_1 = []
    maybe_any_node_state_perturbations_kvpairs_2 = [('any?', '6')]

    for mode, max_t, zero_node_state_perturbations_kvpairs, one_node_state_perturbations_kvpairs, \
        maybe_zero_node_state_perturbations_kvpairs, maybe_one_node_state_perturbations_kvpairs, \
        any_node_state_perturbations_kvpairs, maybe_any_node_state_perturbations_kvpairs in product(
        [mode_1, mode_2, mode_3], [max_t_1, max_t_2],
        [zero_node_state_perturbations_kvpairs_1, zero_node_state_perturbations_kvpairs_2],
        [one_node_state_perturbations_kvpairs_1, one_node_state_perturbations_kvpairs_2],
        [maybe_zero_node_state_perturbations_kvpairs_1, maybe_zero_node_state_perturbations_kvpairs_2],
        [maybe_one_node_state_perturbations_kvpairs_1, maybe_one_node_state_perturbations_kvpairs_2],
        [any_node_state_perturbations_kvpairs_1, any_node_state_perturbations_kvpairs_2],
        [maybe_any_node_state_perturbations_kvpairs_1, maybe_any_node_state_perturbations_kvpairs_2]):

        if mode == Mode.ATTRACT and \
                (maybe_zero_node_state_perturbations_kvpairs or maybe_one_node_state_perturbations_kvpairs
                 or any_node_state_perturbations_kvpairs or maybe_any_node_state_perturbations_kvpairs):
            continue

        expected_perturbed_nodes_by_t = dict()
        expected_perturbations_variations = []
        if zero_node_state_perturbations_kvpairs:
            expected_perturbed_nodes_by_t[1] = {0: False}
        if one_node_state_perturbations_kvpairs:
            expected_perturbed_nodes_by_t[2] = {0: True}
        if maybe_zero_node_state_perturbations_kvpairs:
            heappush(expected_perturbations_variations, (3, 0, NodeStateRange.MAYBE_FALSE))
        if maybe_one_node_state_perturbations_kvpairs:
            heappush(expected_perturbations_variations, (4, 0, NodeStateRange.MAYBE_TRUE))
        if any_node_state_perturbations_kvpairs:
            expected_perturbed_nodes_by_t[5] = {0: False}
            heappush(expected_perturbations_variations, (5, 0, NodeStateRange.TRUE_OR_FALSE))
        if maybe_any_node_state_perturbations_kvpairs:
            heappush(expected_perturbations_variations, (6, 0, NodeStateRange.MAYBE_TRUE_OR_FALSE))

        perturbations_kvpairs = \
            zero_node_state_perturbations_kvpairs + one_node_state_perturbations_kvpairs + \
            maybe_zero_node_state_perturbations_kvpairs + maybe_one_node_state_perturbations_kvpairs + \
            any_node_state_perturbations_kvpairs + maybe_any_node_state_perturbations_kvpairs
        if perturbations_kvpairs:
            raw_input_perturbations['A'] = dict()
            raw_input_perturbations['A'].update(perturbations_kvpairs)
        else:
            raw_input_perturbations = dict()

        perturbed_nodes_by_t, perturbations_variations = parse_raw_input_perturbations(
            raw_input_perturbations, node_names, mode, max_t)

        test_description = generate_test_description(
            locals(), 'mode', 'max_t', 'zero_node_state_perturbations_kvpairs',
            'one_node_state_perturbations_kvpairs', 'maybe_zero_node_state_perturbations_kvpairs',
            'maybe_one_node_state_perturbations_kvpairs', 'any_node_state_perturbations_kvpairs',
            'maybe_any_node_state_perturbations_kvpairs')
        assert expected_perturbed_nodes_by_t == perturbed_nodes_by_t, test_description
        assert expected_perturbations_variations == perturbations_variations, test_description


def test_parse_raw_input_perturbations_B():
    """
    `parse_raw_input_perturbations`
    Feature B: raising exception when perturbations are not a dict.
    """
    raw_input_perturbations = '5'
    node_names = ['A']
    max_t = None
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_C():
    """
    `parse_raw_input_perturbations`
    Feature C: raising exception for duplicate (modulo whitespace) node
        name.
    """
    raw_input_perturbations = {'A': {'1': '5'}, ' A': {'1': '15'}}
    node_names = ['A']
    max_t = None
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_D():
    """
    `parse_raw_input_perturbations`
    Feature D: raising exception for unknown node names.
    """
    raw_input_perturbations = {'A': {'1': '5'}, 'B': {'1': '15'}}
    node_names = ['A']
    max_t = inf
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_E():
    """
    `parse_raw_input_perturbations`
    Feature E: raising exception when node perturbations are not a dict.
    """
    raw_input_perturbations = {'A': '5'}
    node_names = ['A']
    max_t = inf
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_F():
    """
    `parse_raw_input_perturbations`
    Feature F: raising exception for duplicate (modulo whitespace) node
        states.
    """
    raw_input_perturbations = {'A': {'1': '5', ' 1': '15'}}
    node_names = ['A']
    max_t = inf
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_G():
    """
    `parse_raw_input_perturbations`
    Feature G: raising exception when time steps are not a string.
    """
    raw_input_perturbations = {'A': {'1': [5]}}
    node_names = ['A']
    # Test for {no simulation time cap, simulation time cap}.
    max_t_1 = inf
    max_t_2 = 666
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode, max_t in product([mode_1, mode_2, mode_3], [max_t_1, max_t_2]):
        test_description = generate_test_description(locals(), 'mode', 'max_t')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_H():
    """
    `parse_raw_input_perturbations`
    Feature H: raising exception when time is not positive.
    """
    raw_input_perturbations = {'A': {'1': '0'}}
    node_names = ['A']
    # Test for {no simulation time cap, simulation time cap}.
    max_t_1 = inf
    max_t_2 = 666
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode, max_t in product([mode_1, mode_2, mode_3], [max_t_1, max_t_2]):
        test_description = generate_test_description(locals(), 'mode', 'max_t')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_I():
    """
    `parse_raw_input_perturbations`
    Feature I: raising exception when time step exceeds simulation time cap.
    """
    raw_input_perturbations = {'A': {'1': '11'}}
    node_names = ['A']
    max_t = 10
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_J():
    """
    `parse_raw_input_perturbations`
    Feature J: raising exception when times overlap.
    """
    node_names = ['A']
    # Test for {no simulation time cap, simulation time cap}.
    max_t_1 = inf
    max_t_2 = 666
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET
    # Test for {'0', '1', '0?', '1?', 'any', 'any?'}.
    node_state_1 = '0'
    node_state_2 = '1'
    node_state_3 = '0?'
    node_state_4 = '1?'
    node_state_5 = 'any'
    node_state_6 = 'any?'
    node_states = [node_state_1, node_state_2, node_state_3, node_state_4, node_state_5, node_state_6]

    for mode, max_t, node_state in product(
            [mode_1, mode_2, mode_3], [max_t_1, max_t_2], node_states):
        # Test for {'0', '1', '0?', '1?', 'any', 'any?'} - {node_state}.
        for another_node_state in set(node_states) - {node_state}:
            if mode != Mode.ATTRACT or {node_state, another_node_state}.issubset({'0', '1'}):
                raw_input_perturbations = {'A': {node_state: '5', another_node_state: '5'}}

                test_description = generate_test_description(
                    locals(), 'mode', 'max_t', 'node_state', 'another_node_state')
                with pytest.raises(ValueError, message=test_description):
                    parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_K():
    """
    `parse_raw_input_perturbations`
    Feature K: raising exception for non-constant perturbation node
        states in Attract mode.
    """
    node_names = ['A']
    # Test for {no simulation time cap, simulation time cap}.
    max_t_1 = inf
    max_t_2 = 666
    # Test for {'0?', '1?', 'any', 'any?'}.
    node_state_1 = '0?'
    node_state_2 = '1?'
    node_state_3 = 'any'
    node_state_4 = 'any?'

    for max_t, node_state in product(
            [max_t_1, max_t_2], [node_state_1, node_state_2, node_state_3, node_state_4]):
        raw_input_perturbations = {'A': {node_state: '5'}}

        test_description = generate_test_description(locals(), 'max_t', 'node_state')
        with pytest.raises(ValueError):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, Mode.ATTRACT, max_t)


def test_parse_raw_input_perturbations_L():
    """
    `parse_raw_input_perturbations`
    Feature L: raising exception for unrecognized perturbation node
        state.
    """
    raw_input_perturbations = {'A': {'a': '5'}}
    node_names = ['A']
    max_t = 10
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t)


def test_parse_raw_input_perturbations_X():
    """
    `parse_raw_input_perturbations`
    Feature X: parsing empty perturbations from absent input.
    """
    raw_input_perturbations = None
    node_names = ['A']
    max_t = 10
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    expected_perturbed_nodes_by_t = dict()
    expected_perturbations_variations = []

    for mode in [mode_1, mode_2, mode_3]:
        perturbed_nodes_by_t, perturbations_variations = parse_raw_input_perturbations(
            raw_input_perturbations, node_names, mode, max_t)

        test_description = generate_test_description(locals(), 'mode')
        assert expected_perturbed_nodes_by_t == perturbed_nodes_by_t, test_description
        assert expected_perturbations_variations == perturbations_variations, test_description


def test_parse_raw_input_fixed_nodes_A():
    """
    `parse_raw_input_fixed_nodes`
    Feature A: parsing fixed nodes from proper input.
    """
    node_names = ['A', 'B', 'C', 'D', 'E', 'F']
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET
    # Test for {no '0' fixed nodes, '0' fixed nodes}.
    zero_node_state_fixed_nodes_kvpairs_1 = []
    zero_node_state_fixed_nodes_kvpairs_2 = [('A', '0')]
    # Test for {no '1' fixed nodes, '1' fixed nodes}.
    one_node_state_fixed_nodes_kvpairs_1 = []
    one_node_state_fixed_nodes_kvpairs_2 = [('B', '1')]
    # Test for {no '0?' fixed nodes, '0?' fixed nodes}.
    maybe_zero_node_state_fixed_nodes_kvpairs_1 = []
    maybe_zero_node_state_fixed_nodes_kvpairs_2 = [('C', '0?')]
    # Test for {no '1?' fixed nodes, '1?' fixed nodes}.
    maybe_one_node_state_fixed_nodes_kvpairs_1 = []
    maybe_one_node_state_fixed_nodes_kvpairs_2 = [('D', '1?')]
    # Test for {no 'any' fixed nodes, 'any' fixed nodes}.
    any_node_state_fixed_nodes_kvpairs_1 = []
    any_node_state_fixed_nodes_kvpairs_2 = [('E', 'any')]
    # Test for {no 'any' fixed nodes, 'any?' fixed nodes}.
    maybe_any_node_state_fixed_nodes_kvpairs_1 = []
    maybe_any_node_state_fixed_nodes_kvpairs_2 = [('F', 'any?')]

    for mode, zero_node_state_fixed_nodes_kvpairs, one_node_state_fixed_nodes_kvpairs, \
        maybe_zero_node_state_fixed_nodes_kvpairs, maybe_one_node_state_fixed_nodes_kvpairs, \
        any_node_state_fixed_nodes_kvpairs, maybe_any_node_state_fixed_nodes_kvpairs in product(
        [mode_1, mode_2, mode_3],
        [zero_node_state_fixed_nodes_kvpairs_1, zero_node_state_fixed_nodes_kvpairs_2],
        [one_node_state_fixed_nodes_kvpairs_1, one_node_state_fixed_nodes_kvpairs_2],
        [maybe_zero_node_state_fixed_nodes_kvpairs_1, maybe_zero_node_state_fixed_nodes_kvpairs_2],
        [maybe_one_node_state_fixed_nodes_kvpairs_1, maybe_one_node_state_fixed_nodes_kvpairs_2],
        [any_node_state_fixed_nodes_kvpairs_1, any_node_state_fixed_nodes_kvpairs_2],
        [maybe_any_node_state_fixed_nodes_kvpairs_1, maybe_any_node_state_fixed_nodes_kvpairs_2]):

        if mode == Mode.ATTRACT and \
                (maybe_zero_node_state_fixed_nodes_kvpairs or maybe_one_node_state_fixed_nodes_kvpairs
                 or any_node_state_fixed_nodes_kvpairs or maybe_any_node_state_fixed_nodes_kvpairs):
            continue

        expected_fixed_nodes = dict()
        expected_fixed_nodes_variations = []
        if zero_node_state_fixed_nodes_kvpairs:
            expected_fixed_nodes[0] = False
        if one_node_state_fixed_nodes_kvpairs:
            expected_fixed_nodes[1] = True
        if maybe_zero_node_state_fixed_nodes_kvpairs:
            heappush(expected_fixed_nodes_variations, (2, NodeStateRange.MAYBE_FALSE))
        if maybe_one_node_state_fixed_nodes_kvpairs:
            heappush(expected_fixed_nodes_variations, (3, NodeStateRange.MAYBE_TRUE))
        if any_node_state_fixed_nodes_kvpairs:
            expected_fixed_nodes[4] = False
            heappush(expected_fixed_nodes_variations, (4, NodeStateRange.TRUE_OR_FALSE))
        if maybe_any_node_state_fixed_nodes_kvpairs:
            heappush(expected_fixed_nodes_variations, (5, NodeStateRange.MAYBE_TRUE_OR_FALSE))

        fixed_nodes_pairs = \
            zero_node_state_fixed_nodes_kvpairs + one_node_state_fixed_nodes_kvpairs + \
            maybe_zero_node_state_fixed_nodes_kvpairs + maybe_one_node_state_fixed_nodes_kvpairs + \
            any_node_state_fixed_nodes_kvpairs + maybe_any_node_state_fixed_nodes_kvpairs
        raw_input_fixed_nodes = dict()
        raw_input_fixed_nodes.update(fixed_nodes_pairs)

        fixed_nodes, fixed_nodes_variations = parse_raw_input_fixed_nodes(
            raw_input_fixed_nodes, node_names, mode)

        test_description = generate_test_description(
            locals(), 'mode', 'zero_node_state_fixed_nodes_kvpairs',
            'one_node_state_fixed_nodes_kvpairs', 'maybe_zero_node_state_fixed_nodes_kvpairs',
            'maybe_one_node_state_fixed_nodes_kvpairs', 'any_node_state_fixed_nodes_kvpairs',
            'maybe_any_node_state_fixed_nodes_kvpairs')
        assert expected_fixed_nodes == fixed_nodes, test_description
        assert expected_fixed_nodes_variations == fixed_nodes_variations, test_description


def test_parse_raw_input_fixed_nodes_B():
    """
    `parse_raw_input_fixed_nodes`
    Feature B: raising exception when fixed nodes are not a dict.
    """
    raw_input_fixed_nodes = '5'
    node_names = ['A']
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_fixed_nodes(raw_input_fixed_nodes, node_names, mode)


def test_parse_raw_input_fixed_nodes_C():
    """
    `parse_raw_input_fixed_nodes`
    Feature C: raising exception for duplicate (modulo whitespace) node.
    """
    raw_input_fixed_nodes = {'A': '1', ' A': '1'}
    node_names = ['A']
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_fixed_nodes(raw_input_fixed_nodes, node_names, mode)


def test_parse_raw_input_fixed_nodes_D():
    """
    `parse_raw_input_fixed_nodes`
    Feature D: raising exception for unknown node name.
    """
    raw_input_fixed_nodes = {'B': '1'}
    node_names = ['A']
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_fixed_nodes(raw_input_fixed_nodes, node_names, mode)


def test_parse_raw_input_fixed_nodes_E():
    """
    `parse_raw_input_fixed_nodes`
    Feature E: raising exception when fixed node state is not a string.
    """
    raw_input_fixed_nodes = {'A': [1]}
    node_names = ['A']
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_fixed_nodes(raw_input_fixed_nodes, node_names, mode)


def test_parse_raw_input_fixed_nodes_F():
    """
    `parse_raw_input_fixed_nodes`
    Feature F: raising exception for non-constant fixed node states in
        Attract mode.
    """
    node_names = ['A']
    # Test for {'0?', '1?', 'any', 'any?'}.
    node_state_1 = '0?'
    node_state_2 = '1?'
    node_state_3 = 'any'
    node_state_4 = 'any?'

    for node_state in [node_state_1, node_state_2, node_state_3, node_state_4]:
        raw_input_fixed_nodes = {'A': node_state}

        test_description = generate_test_description(locals(), 'node_state')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_fixed_nodes(raw_input_fixed_nodes, node_names, Mode.ATTRACT)


def test_parse_raw_input_fixed_nodes_G():
    """
    `parse_raw_input_fixed_nodes`
    Feature G: raising exception for unrecognized fixed node state.
    """
    raw_input_fixed_nodes = {'A': 'a'}
    node_names = ['A']
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    for mode in [mode_1, mode_2, mode_3]:
        test_description = generate_test_description(locals(), 'mode')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_fixed_nodes(raw_input_fixed_nodes, node_names, mode)


def test_parse_raw_input_fixed_nodes_X():
    """
    `parse_raw_input_fixed_nodes`
    Feature X: parsing empty fixed nodes from absent input.
    """
    raw_input_fixed_nodes = None
    node_names = ['A']
    # Test for {Simulate mode, Attract mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT
    mode_3 = Mode.TARGET

    expected_fixed_nodes = dict()
    expected_fixed_nodes_variations = []

    for mode in [mode_1, mode_2, mode_3]:
        fixed_nodes, fixed_nodes_variations = parse_raw_input_fixed_nodes(
            raw_input_fixed_nodes, node_names, mode)

        test_description = generate_test_description(locals(), 'mode')
        assert expected_fixed_nodes == fixed_nodes, test_description
        assert expected_fixed_nodes_variations == fixed_nodes_variations, test_description


def test_parse_raw_input_target_state_A():
    """
    `parse_input_raw_target_state`
    Feature A: parsing target state from proper input.
    """
    raw_input_target_state = {'A': '0', 'B': 'aNy'}
    node_names = ['A', 'B']

    expected_target_subcode = 0
    expected_target_node_set = {0}

    target_subcode, target_node_set = parse_raw_input_target_state(raw_input_target_state,
                                                                   node_names, Mode.TARGET)

    assert expected_target_subcode == target_subcode
    assert expected_target_node_set == target_node_set


def test_parse_raw_input_target_state_B():
    """
    `parse_input_raw_target_state`
    Feature A: ignoring target state outside Target mode.
    """
    raw_input_target_state = {'A': '0', 'B': 'aNy'}
    node_names = ['A', 'B']
    # Test for {Simulate mode, Target mode}.
    mode_1 = Mode.SIMULATE
    mode_2 = Mode.ATTRACT

    expected_target_subcode = None
    expected_target_node_set = None

    for mode in [mode_1, mode_2]:
        target_subcode, target_node_set = parse_raw_input_target_state(
            raw_input_target_state, node_names, mode)

        test_description = generate_test_description(locals(), 'mode')
        assert expected_target_subcode == target_subcode
        assert expected_target_node_set == target_node_set


def test_parse_raw_input_target_state_C():
    """
    `parse_input_raw_target_state`
    Feature C: raising exception when target state is not a dict.
    """
    raw_input_target_state = ['A']
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_target_state(raw_input_target_state, node_names, Mode.TARGET)


def test_parse_raw_input_target_state_D():
    """
    `parse_input_raw_target_state`
    Feature D: raising exception for duplicate (modulo whitespace) node
        names.
    """
    raw_input_target_state = {'A': '0', ' A': '0'}
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_target_state(raw_input_target_state, node_names, Mode.TARGET)


def test_parse_raw_input_target_state_E():
    """
    `parse_input_raw_target_state`
    Feature E: raising exception for unknown node names.
    """
    raw_input_target_state = {'B': '0'}
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_target_state(raw_input_target_state, node_names, Mode.TARGET)


def test_parse_raw_input_target_state_F():
    """
    `parse_input_raw_target_state`
    Feature F: raising exception for missing node names.
    """
    raw_input_target_state = {'B': '0'}
    node_names = ['A', 'B']

    with pytest.raises(ValueError):
        parse_raw_input_target_state(raw_input_target_state, node_names, Mode.TARGET)


def test_parse_raw_input_target_state_G():
    """
    `parse_input_raw_target_state`
    Feature G: raising exception when node state is not a string.
    """
    raw_input_target_state = {'A': ['0']}
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_target_state(raw_input_target_state, node_names, Mode.TARGET)


def test_parse_raw_input_target_state_H():
    """
    `parse_input_raw_target_state`
    Feature H: raising exception for node state with '?'.
    """
    node_names = ['A']
    # Test for {'0?', '1?', 'any?'}
    node_state_1 = '0?'
    node_state_2 = '1?'
    node_state_3 = 'any?'

    for node_state in [node_state_1, node_state_2, node_state_3]:
        raw_input_target_state = {'A': node_state_1}

        test_description = generate_test_description(locals(), 'node_state')
        with pytest.raises(ValueError, message=test_description):
            parse_raw_input_target_state(raw_input_target_state, node_names, Mode.TARGET)


def test_parse_raw_input_target_state_I():
    """
    `parse_input_raw_target_state`
    Feature I: raising exception for unrecognized node state.
    """
    raw_input_target_state = {'A': 'a'}
    node_names = ['A']

    with pytest.raises(ValueError):
        parse_raw_input_target_state(raw_input_target_state, node_names, Mode.TARGET)


def test_parse_raw_input_target_state_J():
    """
    `parse_raw_input_target_state`
    Feature J: raising exception for missing target state in Target mode.
    """
    with pytest.raises(ValueError):
        parse_raw_input_target_state(None, ['A'], Mode.TARGET)


def test_parse_raw_input_target_state_K():
    """
    `parse_raw_input_target_state`
    Feature K: ignoring target state in non-Target modes.
    """
    for mode in [Mode.SIMULATE, Mode.ATTRACT]:
        target_substate_code, target_node_set = parse_raw_input_target_state({'A': '1'}, ['A'], mode)

        test_description = generate_test_description(locals(), 'mode')
        assert target_substate_code is None, test_description
        assert target_node_set is None, test_description



def test_parse_raw_input_time_steps_A():
    """
    `parse_raw_input_time_steps`
    Feature A: parsing time steps from proper input.
    """
    # Test for {no empty intervals, empty intervals}.
    empty_intervals_1 = []
    empty_intervals_2 = ['']
    # Test for {no simple intervals, simple intervals}.
    simple_intervals_1 = []
    simple_intervals_2 = ['5']
    # Test for {no hyphened intervals, hyphened intervals}.
    hyphened_intervals_1 = []
    hyphened_intervals_2 = ['8-10']

    for empty_intervals, simple_intervals, hyphened_intervals in product(
            [empty_intervals_1, empty_intervals_2], [simple_intervals_1, simple_intervals_2],
            [hyphened_intervals_1, hyphened_intervals_2]):
        raw_input_times = ','.join(empty_intervals + simple_intervals + hyphened_intervals)

        expected_time_steps = []
        if simple_intervals:
            expected_time_steps.append(5)
        if hyphened_intervals:
            expected_time_steps.extend([8, 9, 10])

        time_steps = list(parse_raw_input_time_steps(raw_input_times))

        test_description = generate_test_description(
            locals(), 'empty_intervals', 'simple_intervals', 'hyphened_intervals')
        assert expected_time_steps == time_steps, test_description


def test_parse_raw_input_time_steps_B():
    """
    `parse_raw_input_time_steps`
    Feature B: raising exception for descending interval.
    """
    raw_input_times = '52 -51'

    with pytest.raises(ValueError):
        list(parse_raw_input_time_steps(raw_input_times))


def test_parse_raw_input_time_steps_C():
    """
    `parse_raw_input_time_steps`
    Feature C: raising exception when interval starts with hyphen.
    """
    raw_input_times = '- 51'

    with pytest.raises(ValueError):
        list(parse_raw_input_time_steps(raw_input_times))


def test_parse_raw_input_time_steps_D():
    """
    `parse_raw_input_time_steps`
    Feature D: raising exception when interval ends with hyphen.
    """
    raw_input_times = '52-'

    with pytest.raises(ValueError):
        list(parse_raw_input_time_steps(raw_input_times))


def test_parse_raw_input_time_steps_E():
    """
    `parse_raw_input_time_steps`
    Feature E: raising exception when interval contains more than one
        hyphen.
    """
    raw_input_times = '51 -- 52'

    with pytest.raises(ValueError):
        list(parse_raw_input_time_steps(raw_input_times))


def test_parse_raw_input_time_steps_F():
    """
    `parse_raw_input_time_steps`
    Feature F: raising exception for improper characters.
    """
    raw_input_times = '25a-26'

    with pytest.raises(ValueError):
        list(parse_raw_input_time_steps(raw_input_times))


def test_parse_input_update_rules_A():
    """
    `parse_input_update_rules`
    Feature A: parsing predecessor node lists and truth tables from
        proper input.
    """
    # Test for {no invalid Python node names, invalid Python node names}.
    node_name_1 = 'A'
    node_name_2 = '1A'
    # Test for {lowercase operators, non-lowercase operators}.
    not_txt_1 = 'nOt'
    not_txt_2 = 'not'
    # Test for {no constant '0', constant '0'}.
    zero_expression_txt_1 = ''
    zero_expression_txt_2 = 'or 0'
    # Test for {no constant '1', constant '1'}.
    one_expression_txt_1 = ''
    one_expression_txt_2 = 'and 1'

    expected_predecessor_nodes_lists = [[0]]
    expected_truth_tables = [{(False,): True, (True,): False}]

    for node_name, not_txt, zero_expression_txt, one_expression_txt in product(
            [node_name_1, node_name_2], [not_txt_1, not_txt_2],
            [zero_expression_txt_1, zero_expression_txt_2],
            [one_expression_txt_1, one_expression_txt_2]):
        node_names = [node_name]
        update_rules_dict = {node_name: ' '.join(
            [not_txt, node_name, zero_expression_txt, one_expression_txt])}

        predecessor_nodes_lists, truth_tables = parse_input_update_rules(
            update_rules_dict, node_names, {'section': 'update rules'})

        test_description = generate_test_description(
            locals(), 'node_name', 'not_txt', 'zero_expression_txt', 'one_expression_txt')
        assert expected_predecessor_nodes_lists == predecessor_nodes_lists, test_description
        assert expected_truth_tables == truth_tables, test_description


def test_parse_input_update_rules_B():
    """
    `parse_input_update_rules`
    Feature B: raising exception when update rule is for unrecognized
        node name.
    """
    node_names = ['A']
    update_rules_dict = {'B': 'A'}

    with pytest.raises(ValueError):
        parse_input_update_rules(update_rules_dict, node_names, {'section': 'update rules'})


def test_parse_input_update_rules_C():
    """
    `parse_input_update_rules`
    Feature C: raising exception when update rule is for a constant.
    """
    node_names = ['A']
    # Test for {'0', '1', '0?', '1?', 'any', 'any?'}.
    constant_txt_1 = '0'
    constant_txt_2 = '1'
    constant_txt_3 = '0?'
    constant_txt_4 = '1?'
    constant_txt_5 = 'aNy'
    constant_txt_6 = 'aNy?'

    for constant_txt in [constant_txt_1, constant_txt_2, constant_txt_3, constant_txt_4,
                         constant_txt_5, constant_txt_6]:
        update_rules_dict = {constant_txt: 'A'}

        test_description = generate_test_description(locals(), 'constant_txt')
        with pytest.raises(ValueError, message=test_description):
            parse_input_update_rules(update_rules_dict, node_names, {'section': 'update rules'})


def test_parse_input_update_rules_D():
    """
    `parse_input_update_rules`
    Feature D: raising exception for missing update rule for some node.
    """
    node_names = ['A', 'B']
    update_rules_dict = {'A': 'B'}

    with pytest.raises(ValueError):
        parse_input_update_rules(update_rules_dict, node_names, {'section': 'update rules'})


def test_parse_input_update_rules_E():
    """
    `parse_input_update_rules`
    Feature E: raising exception when update rule is not a string.
    """
    node_names = ['A']
    update_rules_dict = {'A': ['A']}

    with pytest.raises(ValueError):
        parse_input_update_rules(update_rules_dict, node_names, {'section': 'update rules'})


def test_parse_input_update_rules_F():
    """
    `parse_input_update_rules`
    Feature F: raising exception for uncrecognized expression in update
        rule.
    """
    node_names = ['A']
    update_rules_dict = {'A': 'any'}

    with pytest.raises(ValueError):
        parse_input_update_rules(update_rules_dict, node_names, {'section': 'update rules'})


def test_parse_input_update_rules_G():
    """
    `parse_input_update_rules`
    Feature G: raising exception for majority function referenced but
        not called.
    """
    node_names = ['A']
    # Test for {no leading majority call, leading majority call}.
    leading_majority_txt_1 = ''
    leading_majority_txt_2 = 'majority(A, not A, 1) and'
    # Test for {no trailing majority call, trailing majority call}.
    trailing_majority_txt_1 = ''
    trailing_majority_txt_2 = ' and majority(A, not A, 0)'

    update_rules_dict = {'A': 'majority'}

    for leading_majority_txt, trailing_majority_txt in product(
            [leading_majority_txt_1, leading_majority_txt_2],
            [trailing_majority_txt_1, trailing_majority_txt_2]):
        update_rule = leading_majority_txt + ' majority ' + trailing_majority_txt
        update_rules_dict = {'A': update_rule}

        test_description = generate_test_description(
            locals(), 'leading_majority_txt', 'trailing_majority_txt')
        with pytest.raises(ValueError):
            parse_input_update_rules(update_rules_dict, node_names, {'section': 'update rules'})


def test_parse_input_update_rules_H():
    """
    `parse_input_update_rules`
    Feature H: raising exception for comma outside majority call.
    """
    node_names = ['A']
    # Test for {no leading majority call, leading majority call}.
    leading_majority_txt_1 = ''
    leading_majority_txt_2 = 'majority(A, not A, 1)'
    # Test for {no trailing majority call, trailing majority call}.
    trailing_majority_txt_1 = ''
    trailing_majority_txt_2 = 'majority(A, not A, 0)'

    for leading_majority_txt, trailing_majority_txt in product(
            [leading_majority_txt_1, leading_majority_txt_2],
            [trailing_majority_txt_1, trailing_majority_txt_2]):
        update_rule = leading_majority_txt + ',' + trailing_majority_txt
        update_rules_dict = {'A': update_rule}

        test_description = generate_test_description(
            locals(), 'leading_majority_txt', 'trailing_majority_txt')
        with pytest.raises(ValueError):
            parse_input_update_rules(update_rules_dict, node_names, {'section': 'update rules'})


def test_parse_input_update_rules_J():
    """
    `parse_input_update_rules`
    Feature J: raising exception for trailing commas in majority call.
    """
    node_names = ['A']
    # Test for {no leading majority call, leading majority call}.
    leading_majority_txt_1 = ''
    leading_majority_txt_2 = 'majority(A, not A, 1)'
    # Test for {no trailing majority call, trailing majority call}.
    trailing_majority_txt_1 = ''
    trailing_majority_txt_2 = 'majority(A, not A, 0)'

    for leading_majority_txt, trailing_majority_txt in product(
            [leading_majority_txt_1, leading_majority_txt_2],
            [trailing_majority_txt_1, trailing_majority_txt_2]):
        update_rule = leading_majority_txt + ' majority(A,) ' + trailing_majority_txt
        update_rules_dict = {'A': update_rule}

        test_description = generate_test_description(
            locals(), 'leading_majority_txt', 'trailing_majority_txt')
        with pytest.raises(ValueError):
            parse_input_update_rules(update_rules_dict, node_names, {'section': 'update rules'})


def test_parse_incoming_node_name_set_from_update_rule_A():
    """
    `parse_incoming_node_name_set_from_update_rule`
    Feature A: extracting set of predecessor node names from update
        rule.
    """
    # Test for {no constant '0', constant '0'}.
    zero_txt_1 = ''
    zero_txt_2 = '0,'
    # Test for {no constant '1', constant '1'}.
    one_txt_1 = ''
    one_txt_2 = '1 ,'
    # Test for {no 'or',  lowercase 'or', non-lowercase 'or'}.
    not_txt_1 = ''
    not_txt_2 = 'not A not'
    not_txt_3 = 'nOt A NOT'
    # Test for {no 'and',  lowercase 'and', non-lowercase 'and'}.
    and_txt_1 = ''
    and_txt_2 = 'and B and'
    and_txt_3 = 'aNd B AND'
    # Test for {no 'or',  lowercase 'or', non-lowercase 'or'}.
    or_txt_1 = ''
    or_txt_2 = 'or C or'
    or_txt_3 = 'oR C OR'
    # Test for {no majority,  lowercase majority in 1 argument,
    # non-lowercase majority in 1 argument, majority in >1 argument}.
    majority_txt_1 = ''
    majority_txt_2 = 'majority(D)'
    majority_txt_3 = 'maJority(D)'
    majority_txt_4 = 'MAJORITY(D, D,D)'
    # Test for {no successive operands, successive operands}.
    successive_txt_1 = ''
    successive_txt_2 = 'E F 1'
    # Test for {no parentheses outside functions, parentheses outside functions}.
    parentheses_txt_1 = ''
    parentheses_txt_2 = '( G) '

    for zero_txt, one_txt, not_txt, and_txt, or_txt, majority_txt, successive_txt, parentheses_txt in \
            product([zero_txt_1, zero_txt_2], [one_txt_1, one_txt_2], [not_txt_1, not_txt_2, not_txt_3],
                    [and_txt_1, and_txt_2, and_txt_3], [or_txt_1, or_txt_2, or_txt_3],
                    [majority_txt_1, majority_txt_2, majority_txt_3, majority_txt_4],
                    [successive_txt_1, successive_txt_2], [parentheses_txt_1, parentheses_txt_2]):

        update_rule = ' '.join([zero_txt, one_txt, not_txt, and_txt, or_txt, majority_txt,
                                successive_txt, parentheses_txt])

        expected_node_names_set = set()
        if zero_txt:
            expected_node_names_set.add('0')
        if one_txt:
            expected_node_names_set.add('1')
        if not_txt:
            expected_node_names_set.add('A')
        if and_txt:
            expected_node_names_set.add('B')
        if or_txt:
            expected_node_names_set.add('C')
        if majority_txt:
            expected_node_names_set.add('D')
        if successive_txt:
            expected_node_names_set.add('E F 1')
        if parentheses_txt:
            expected_node_names_set.add('G')

        test_description = generate_test_description(
            locals(), 'zero_txt', 'one_txt', 'not_txt', 'and_txt', 'or_txt', 'majority_txt',
            'successive_txt', 'parentheses_txt')
        node_names_set = parse_predecessor_node_names_from_update_rule(update_rule)
        
        assert expected_node_names_set == node_names_set, test_description


def test_build_truth_table_from_safe_update_rule_A():
    """
    `build_truth_table_from_safe_update_rule`
    Feature A: evaluating 'and' operator correctly.
    """
    safe_node_names = ['node1', 'node2']
    predecessor_nodes = [0, 1]
    safe_update_rule = ' and '.join(safe_node_names)

    expected_truth_table = \
        {(False, False): False, (False, True): False, (True, False): False, (True, True): True}

    truth_table = build_truth_table_from_safe_update_rule(
        safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')

    assert expected_truth_table == truth_table


def test_build_truth_table_from_safe_update_rule_B():
    """
    `build_truth_table_from_safe_update_rule`
    Feature B: evaluating 'or' operator correctly.
    """
    safe_node_names = ['node1', 'node2']
    predecessor_nodes = [0, 1]
    safe_update_rule = ' or '.join(safe_node_names)

    expected_truth_table = \
        {(False, False): False, (False, True): True, (True, False): True, (True, True): True}

    truth_table = build_truth_table_from_safe_update_rule(
        safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')

    assert expected_truth_table == truth_table


def test_build_truth_table_from_safe_update_rule_C():
    """
    `build_truth_table_from_safe_update_rule`
    Feature C: evaluating 'not' operator correctly.
    """
    safe_node_names = ['node1']
    predecessor_nodes = [0]
    safe_update_rule = ' not ' + safe_node_names[0]

    expected_truth_table = \
        {(False, ): True, (True, ): False}

    truth_table = build_truth_table_from_safe_update_rule(
        safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')

    assert expected_truth_table == truth_table


def test_build_truth_table_from_safe_update_rule_D():
    """
    `build_truth_table_from_safe_update_rule`
    Feature D: evaluating majority function correctly.
    """
    safe_node_names = ['node1', 'node2']
    predecessor_nodes = [0, 1]
    safe_update_rule = 'majority({})'.format(', '.join(safe_node_names))

    expected_truth_table = \
        {(False, False): False, (False, True): False, (True, False): False, (True, True): True}

    truth_table = build_truth_table_from_safe_update_rule(
        safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')

    assert expected_truth_table == truth_table


def test_build_truth_table_from_safe_update_rule_E():
    """
    `build_truth_table_from_safe_update_rule`
    Feature E: prioritizing parentheses.
    """
    safe_node_names = ['node1', 'node2']
    predecessor_nodes = [0, 1]
    safe_update_rule = 'not ({})'.format(' and '.join(safe_node_names))

    expected_truth_table = \
        {(False, False): True, (False, True): True, (True, False): True, (True, True): False}

    truth_table = build_truth_table_from_safe_update_rule(
        safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')

    assert expected_truth_table == truth_table


def test_build_truth_table_from_safe_update_rule_F():
    """
    `build_truth_table_from_safe_update_rule`
    Feature A: prioritizing 'and' operator over 'or' operator.
    """
    safe_node_names = ['node1', 'node2']
    predecessor_nodes = [0, 1]
    safe_update_rule = ' and '.join(safe_node_names) + ' or ' + safe_node_names[1]

    expected_truth_table = \
        {(False, False): False, (False, True): True, (True, False): False, (True, True): True}

    truth_table = build_truth_table_from_safe_update_rule(
        safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')

    assert expected_truth_table == truth_table

def test_build_truth_table_from_safe_update_rule_G():
    """
    `build_truth_table_from_safe_update_rule`
    Feature G: raising exception for using 'not' as binary operator.
    """
    safe_node_names = ['node1', 'node2']
    predecessor_nodes = [0, 1]
    safe_update_rule = ' not '.join(safe_node_names)

    with pytest.raises(ValueError):
        build_truth_table_from_safe_update_rule(
            safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')


def test_build_truth_table_from_safe_update_rule_H():
    """
    `build_truth_table_from_safe_update_rule`
    Feature G: raising exception for using binary operator as unary one.
    """
    safe_node_names = ['node1']
    predecessor_nodes = [0]
    # Test for {'and' operator, 'or' operator}.
    operator_txt_1 = 'and'
    operator_txt_2 = 'or'

    for operator_txt in [operator_txt_1, operator_txt_2]:
        safe_update_rule = operator_txt + ' ' + safe_node_names[0]

        test_description = generate_test_description(locals(), 'operator_txt')
        with pytest.raises(ValueError, message=test_description):
            build_truth_table_from_safe_update_rule(
                safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')


def test_build_truth_table_from_safe_update_rule_I():
    """
    `build_truth_table_from_safe_update_rule`
    Feature I: raising exception for using operator as value.
    """
    safe_node_names = []
    predecessor_nodes = []
    # Test for {'not' operator, 'and' operator, 'or' operator}.
    operator_txt_1 = 'not'
    operator_txt_2 = 'and'
    operator_txt_3 = 'or'

    for operator_txt in [operator_txt_1, operator_txt_2]:
        safe_update_rule = operator_txt

        test_description = generate_test_description(locals(), 'operator_txt')
        with pytest.raises(ValueError, message=test_description):
            build_truth_table_from_safe_update_rule(
                safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')


def test_build_truth_table_from_safe_update_rule_J():
    """
    `build_truth_table_from_safe_update_rule`
    Feature J: raising exception for missing opening parenthesis.
    """
    safe_node_names = ['node1']
    predecessor_nodes = [0]

    safe_update_rule = 'node1)'

    with pytest.raises(ValueError):
        build_truth_table_from_safe_update_rule(
            safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')


def test_build_truth_table_from_safe_update_rule_K():
    """
    `build_truth_table_from_safe_update_rule`
    Feature J: raising exception for missing closing parenthesis.
    """
    safe_node_names = ['node1']
    predecessor_nodes = [0]

    safe_update_rule = '(node1'

    with pytest.raises(ValueError):
        build_truth_table_from_safe_update_rule(
            safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')


def test_build_truth_table_from_safe_update_rule_L():
    """
    `build_truth_table_from_safe_update_rule`
    Feature J: raising exception if output is not Boolean.
    """
    safe_node_names = []
    predecessor_nodes = []

    safe_update_rule = '()'

    with pytest.raises(ValueError):
        build_truth_table_from_safe_update_rule(
            safe_update_rule, predecessor_nodes, safe_node_names, 'dummy error text')


def test_generate_safe_node_names_A():
    """
    `generate_safe_node_names`
    Feature A: generating safe node names that don't overlap with the
    original node names.
    """
    node_names_1 = ['A']
    node_names_2 = ['A', 'node0', 'node_1']

    expected_safe_node_names_1 = ['node0']
    expected_safe_node_names_2 = ['node__0', 'node__1', 'node__2']

    for node_names, expected_safe_node_names in zip(
            [node_names_1, node_names_2], [expected_safe_node_names_1, expected_safe_node_names_2]):

        safe_node_names = generate_safe_node_names(node_names)

        test_description = generate_test_description(locals(), 'node_names')
        assert expected_safe_node_names == safe_node_names, test_description


def test_count_combinations_A():
    """

    Feature A: counting simulation problems.
    """
    # Test for {no initial variations, initial variations}.
    initial_state_variations_1 = []
    initial_state_variations_2 = [5]
    # Test for {no "maybe 0" fixed nodes' variations, "maybe 0" fixed nodes' variations}.
    maybe_zero_fixed_nodes_variations_1 = []
    maybe_zero_fixed_nodes_variations_2 = [(0, NodeStateRange.MAYBE_FALSE)]
    # Test for {no "maybe 1" fixed nodes' variations, "maybe 1" fixed nodes' variations}.
    maybe_one_fixed_nodes_variations_1 = []
    maybe_one_fixed_nodes_variations_2 = [(1, NodeStateRange.MAYBE_TRUE)]
    # Test for {no "maybe 0 or 1" fixed nodes' variations, "maybe 0 or 1" fixed nodes' variations}.
    maybe_zero_or_one_fixed_nodes_variations_1 = []
    maybe_zero_or_one_fixed_nodes_variations_2 = [(2, NodeStateRange.MAYBE_TRUE_OR_FALSE)]
    # Test for {no "0 or 1" fixed nodes' variations, "0 or 1" fixed nodes' variations}.
    zero_or_one_fixed_nodes_variations_1 = []
    zero_or_one_fixed_nodes_variations_2 = [(3, NodeStateRange.TRUE_OR_FALSE)]
    # Test for {no "maybe 0" perturbations' variations, "maybe 0" perturbations' variations}.
    maybe_zero_perturbations_variations_1 = []
    maybe_zero_perturbations_variations_2 = [(666, 0, NodeStateRange.MAYBE_FALSE)]
    # Test for {no "maybe 1" perturbations' variations, "maybe 1" perturbations' variations}.
    maybe_one_perturbations_variations_1 = []
    maybe_one_perturbations_variations_2 = [(666, 1, NodeStateRange.MAYBE_TRUE)]
    # Test for {no "maybe 0 or 1" perturbations' variations, "maybe 0 or 1" perturbations' variations}.
    maybe_zero_or_one_perturbations_variations_1 = []
    maybe_zero_or_one_perturbations_variations_2 = [(666, 2, NodeStateRange.MAYBE_TRUE_OR_FALSE)]
    # Test for {no "0 or 1" perturbations' variations, "0 or 1" perturbations' variations}.
    zero_or_one_perturbations_variations_1 = []
    zero_or_one_perturbations_variations_2 = [(666, 3, NodeStateRange.TRUE_OR_FALSE)]

    for initial_state_variations, maybe_zero_fixed_nodes_variations, \
        maybe_one_fixed_nodes_variations, maybe_zero_or_one_fixed_nodes_variations, \
        zero_or_one_fixed_nodes_variations, maybe_zero_perturbations_variations, \
        maybe_one_perturbations_variations, maybe_zero_or_one_perturbations_variations, \
        zero_or_one_perturbations_variations in product(
        [initial_state_variations_1, initial_state_variations_2],
        [maybe_zero_fixed_nodes_variations_1, maybe_zero_fixed_nodes_variations_2],
        [maybe_one_fixed_nodes_variations_1, maybe_one_fixed_nodes_variations_2],
        [maybe_zero_or_one_fixed_nodes_variations_1, maybe_zero_or_one_fixed_nodes_variations_2],
        [zero_or_one_fixed_nodes_variations_1, zero_or_one_fixed_nodes_variations_2],
        [maybe_zero_perturbations_variations_1, maybe_zero_perturbations_variations_2],
        [maybe_one_perturbations_variations_1, maybe_one_perturbations_variations_2],
        [maybe_zero_or_one_perturbations_variations_1, maybe_zero_or_one_perturbations_variations_2],
        [zero_or_one_perturbations_variations_1, zero_or_one_perturbations_variations_2]):

        fixed_nodes_variations = \
            maybe_zero_fixed_nodes_variations + maybe_one_fixed_nodes_variations + \
            maybe_zero_or_one_fixed_nodes_variations + zero_or_one_fixed_nodes_variations
        perturbations_variations = \
            maybe_zero_perturbations_variations + maybe_one_perturbations_variations + \
            maybe_zero_or_one_perturbations_variations + zero_or_one_perturbations_variations
        n_bistate_variations = \
            len(initial_state_variations) + len(maybe_zero_fixed_nodes_variations) + \
            len(maybe_one_fixed_nodes_variations) + len(zero_or_one_fixed_nodes_variations) + \
            len(maybe_zero_perturbations_variations) + len(maybe_one_perturbations_variations) + \
            len(zero_or_one_perturbations_variations)
        n_tristate_variations = len(maybe_zero_or_one_fixed_nodes_variations) + \
                                len(maybe_zero_or_one_perturbations_variations)

        expected_n_simulation_problems = 2**n_bistate_variations * 3**n_tristate_variations


        n_simulation_problems = count_simulation_problems(
            initial_state_variations, fixed_nodes_variations, perturbations_variations)

        test_description = generate_test_description(
            locals(), 'initial_state_variations', 'maybe_zero_fixed_nodes_variations',
            'maybe_one_fixed_nodes_variations', 'maybe_zero_or_one_fixed_nodes_variations',
            'zero_or_one_fixed_nodes_variations', 'maybe_zero_perturbations_variations',
            'maybe_one_perturbations_variations', 'maybe_zero_or_one_perturbations_variations',
            'zero_or_one_perturbations_variations')
        assert expected_n_simulation_problems == n_simulation_problems, test_description
