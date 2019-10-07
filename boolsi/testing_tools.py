"""
Variables and functions for the tests. Filth on acid.
"""
import string
import random
from functools import partial

from .constants import inf
from .input import parse_raw_input_update_rules, parse_predecessor_node_names_from_update_rule, count_simulation_problems
from .attract import AggregatedAttractor
from .model import simulate_until_attractor_or_target_substate_or_max_t, encode_state


UPDATE_RULES_A = {
    'A': 'E',
    'B': 'A',
    'C': 'B',
    'D': 'not C',
    'E': 'not D'
}

UPDATE_RULES_B = {
    'A': 'not F',
    'B': 'A',
    'C': 'not A and B',
    'D': 'C and not F',
    'E': 'not B and majority(not A, not B, D)',
    'F': 'not C and E'
}


def build_predecessor_nodes_lists_and_truth_tables(update_rules):
    """
    Build predecessor node lists and truth tables based on dict with update
    rules.

    :param update_rules: dict (key: node name, value: string
        with update rule expression)
    :return: list of predecessor node lists, list of truth tables
    """
    node_name_set = set()
    for node_name, update_rule in update_rules.items():
        node_name_set.add(node_name)
        node_name_set |= parse_predecessor_node_names_from_update_rule(update_rule)
    node_name_set -= {'0', '1'}

    return parse_raw_input_update_rules(update_rules, sorted(node_name_set))


def generate_node_name(min_length=1, max_length=12):
    """
    Generate random node name from capital letters and digits of
    length of given range.

    :param min_length: minimum node name length
    :param max_length: maximum node name length
    :return: node name
    """
    try:
        length = random.randint(min_length, max_length)
    except ValueError:

        return None

    else:

        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))


def construct_aggregated_attractor(states, frequency, trajectory_l_mean, trajectory_l_variation_sum):
    """
    Construct attractor with given attributes.

    :param states: attractor states
    :param frequency: attractor frequency
    :param trajectory_l_mean: attractor trajectory length mean
    :param trajectory_l_variation_sum: attractor trajectory length variation sum
    :return: attractor
    """
    aggregated_attractor = AggregatedAttractor(0, [0] * len(states), states, trajectory_l_mean)
    aggregated_attractor.frequency = frequency
    aggregated_attractor.trajectory_l_variation_sum = trajectory_l_variation_sum

    return aggregated_attractor


def configure_encode_and_simulate(
        max_t=inf, substate_node_set=None, storing_all_states=True, target_substate_code=None):
    """
    Make partial application to encode_state() and simulate_until_attractor_or_target_substate_or_max_t().

    :param max_t: maximum simulation time
    :param substate_node_set: nodes of the substate to reach
    :param storing_all_states: whether to store all simulation states
    :param target_substate_code: code of the substate to reach
    :return: ([function] to encode state and substate, [function] to simulate)
    """
    substate_node_set = substate_node_set or set()
    _encode_state = partial(encode_state, substate_node_set)

    _simulate_until_attractor_or_target_substate_or_max_t = partial(
        simulate_until_attractor_or_target_substate_or_max_t, storing_all_states, max_t,
        _encode_state, target_substate_code)

    return _encode_state, _simulate_until_attractor_or_target_substate_or_max_t


def generate_test_description(local_symbols, *variable_names):
    """
    Generate test description.

    :param local_symbols: local symbol table from where the function was called
    :param variable_names: variable names
    :return: test description
    """
    variables_text = ', '.join('{} = {}'.format(variable_name, eval(variable_name, local_symbols))
                               for variable_name in variable_names)

    return 'when testing \'{}\''.format(variables_text)
