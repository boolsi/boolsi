"""
Reads and validates input files.
"""
import collections
import re
import logging
import itertools
from heapq import heappush

from .constants import Mode, NodeStateRange, mode_descriptions
# For evaluating update rules in this script.
from .model import encode_state, majority # DO NOT REMOVE

from yaml import load, BaseLoader
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from shutil import copy2


section_name_text = 'section'
node_name_text = 'node'
perturbed_node_state_text = 'perturbed node state'
constant_node_state_range_text = "either '0' or '1'"
limited_node_state_range_text = "either '0', '1', or 'any'"
full_node_state_range_text = "either '0', '1', 'any', '0?', '1?', or 'any?'"

nonconstant_node_state_pattern = re.compile('^(0\?|1\?|any\??)$', re.I)
true_or_false_node_state_pattern = re.compile('^any$', re.I)
maybe_true_or_false_node_state_pattern = re.compile('^any\?$', re.I)
update_rule_reserved_symbol_pattern = re.compile(r'\(|\)|,|\b(?:and|or|not|majority)\b', re.I)


class InputValidationException(Exception):
    """
    Happens when something goes wrong during input validation.
    """
    pass


class DuplicateKeyError(ConstructorError):
    """
    Raised when YAML contains duplicate keys. Our keys are always unique.
    """
    pass


class UniqueKeyLoader(BaseLoader):
    """
    Custom yaml Loader that checks for duplicates.
    """

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None,
                    "expected a mapping node, but found %s" % node.id,
                    node.start_mark)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, collections.Hashable):
                raise ConstructorError("while constructing a mapping", node.start_mark,
                        "found unhashable key", key_node.start_mark)

            # Check for duplicates.
            if key in mapping:
                raise DuplicateKeyError(None, None,
                        "Duplicate key '{}' found{}".format(key_node.value, key_node.start_mark))

            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


def process_input(input_file, output_directory, max_t, mode):
    """
    Read input file and performs validation.

    :param input_file: input file
    :param output_directory: output directory
    :param max_t: maximum simulation time
    :param mode: BoolSi mode
    :return: configuration dictionary
    """
    try:
        copy2(input_file, output_directory)
    except IOError as e:
        logging.getLogger().warning('Failed to copy input file into the output directory: {}'.format(e))

    try:
        return parse_input(input_file, max_t, mode)
    except (ValueError, KeyError, DuplicateKeyError) as e:
        logging.getLogger().error('Input validation failed: {}'.format(e))

        raise InputValidationException('Failed to validate input.')


def parse_input(file_name, max_t, mode):
    """
    Parse and validate input configuration file.

    :param file_name: input file name
    :param max_t: maximum simulation time
    :param mode: BoolSi mode
    :return: [dict] parsed configuration
    """
    node_names_section_name = 'nodes'
    update_rules_section_name = 'update rules'
    initial_state_section_name = 'initial state'
    fixed_nodes_section_name = 'fixed nodes'
    perturbations_section_name = 'perturbations'
    target_state_section_name = 'target state'
    valid_section_names = \
        {node_names_section_name, update_rules_section_name, initial_state_section_name,
         fixed_nodes_section_name, perturbations_section_name, target_state_section_name}

    with open(file_name) as stream:
        input_data = load(stream, Loader=UniqueKeyLoader)
        configuration = dict()

        # Every empty input is converted first to proper data type to
        # print standard errors.

        configuration['node names'] = parse_raw_input_node_names(
            input_data.get(node_names_section_name))

        configuration['incoming node lists'], configuration['truth tables'] = \
            parse_raw_input_update_rules(input_data.get(update_rules_section_name),
                                         configuration['node names'])

        initial_state, initial_state_variations = parse_raw_input_initial_state(
            input_data.get(initial_state_section_name), configuration['node names'])

        fixed_nodes, fixed_nodes_variations = parse_raw_input_fixed_nodes(
            input_data.get(fixed_nodes_section_name), configuration['node names'], mode)

        perturbed_nodes_by_t, perturbed_nodes_by_t_variations = parse_raw_input_perturbations(
            input_data.get(perturbations_section_name), configuration['node names'], mode, max_t)

        configuration['target substate code'], configuration['target node set'] = \
            parse_raw_input_target_state(
                input_data.get(target_state_section_name), configuration['node names'], mode)

        for section_name in input_data:
            if section_name not in valid_section_names:
                logging.getLogger().warning("Unknown section '{}'.".format(section_name))

        configuration['origin simulation problem'] = (initial_state,
                                                      fixed_nodes,
                                                      perturbed_nodes_by_t)

        configuration['simulation problem variations'] = (initial_state_variations,
                                                          fixed_nodes_variations,
                                                          perturbed_nodes_by_t_variations)

        configuration['total combination count'] = count_simulation_problems(
            initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations)

    return configuration


def parse_raw_input_node_names(raw_input_node_names):
    """
    Parse and validate node names from the input.

    :param raw_input_node_names: list with raw node names
    :return: node names
    """
    section_name = 'nodes'
    bad_format_text = "Expected sequence of (YAML-compliant) nodes. " \
                     "Example:\n'- node1\n- node2\n- node3'"

    if raw_input_node_names is None:

        raise ValueError("Nodes are missing.")

    # Initialize input as list if it's empty.
    raw_input_node_names = raw_input_node_names or []

    # Validate that input is a list.
    if not isinstance(raw_input_node_names, list):
        err_msg = compile_err_msg(bad_format_text, {section_name_text: section_name})

        raise ValueError(err_msg)

    # Validate that there are nodes are specified.
    if not raw_input_node_names:
        err_msg = compile_err_msg("No nodes specified.", {section_name_text: section_name})

        raise ValueError(err_msg)

    node_names = []

    for raw_node_name in raw_input_node_names:
        # Validate that node name is a YAML-compliant string.
        if not isinstance(raw_node_name, str):
            err_msg = compile_err_msg(bad_format_text, {section_name_text: section_name})

            raise ValueError(err_msg)

        node_names.append(raw_node_name.strip())

    for node_name in node_names:
        # Validate no duplicate node names.
        if node_names.count(node_name) > 1:
            err_msg = compile_err_msg("Duplicate node '{}'.".format(node_name),
                                      {section_name_text: section_name})

            raise ValueError(err_msg)
        # Reserved words are forbidden because they don't get
        # substituted with safe node names before evaluation.
        if re.match('^(:?0|1|and|or|not|majority)$', node_name, re.I):
            err_msg = compile_err_msg(
                "Node cannot be '0', '1', 'and', 'or', 'not', 'majority' (case-insensitive).",
                {section_name_text: section_name, node_name_text: node_name})

            raise ValueError(err_msg)
        # Parentheses and commas are forbidden because they, along
        # with 'and', 'not', 'or', and 'majority', are used as
        # demarcators to parse node names from update rules.
        # Whitespaces are forbidden to prevent space-padded
        # demarcators inside the node names.
        if re.search('\s|\(|\)|,', node_name):
            err_msg = compile_err_msg(
                "Name cannot contain whitespaces, parentheses, or commas.",
                {section_name_text: section_name, node_name_text: node_name})

            raise ValueError(err_msg)

    logging.getLogger().info("Read Boolean network of {} nodes.".format(len(node_names)))

    return node_names


def parse_raw_input_update_rules(raw_input_update_rules, node_names):
    """
    Parse and validate predecessor node lists and truth tables from the
    input.

    :param raw_input_update_rules: dict of update rules by raw node names
    :param node_names: node names
    :return: (predecessor node lists, truth tables)
    """
    section_name = 'update rules'
    bad_format_text = "Expected mappings of nodes to their update rules. " \
                     "Example:\n'node1: node5 and (not node2 or node3)\n" \
                     "node2: majority(node1, node2, not node3, node4, 1)\nnode3: node3'"

    if raw_input_update_rules is None:

        raise ValueError("Update rules are missing.")

    # Initialize input as dict if it's empty.
    raw_input_update_rules = raw_input_update_rules or dict()

    # Validate that input is a dict.
    if not isinstance(raw_input_update_rules, dict):
        err_msg = compile_err_msg(bad_format_text, {section_name_text: section_name})

        raise ValueError(err_msg)

    # Validate no duplicate node name (up to whitespace).
    input_update_rules = validate_raw_dict(raw_input_update_rules, node_name_text,
                                           {section_name_text: section_name})

    return parse_input_update_rules(input_update_rules, node_names,
                                    {section_name_text: section_name})


def parse_raw_input_initial_state(raw_input_initial_state, node_names):
    """
    Parse and validate initial state from the input.

    :param raw_input_initial_state: dict of raw initial states by
        raw node names
    :param node_names: node names
    :return: (initial state, initial state variations)
    """
    section_name = 'initial state'
    bad_format_text = \
        "Expected mappings of nodes to their initial states. Example:" \
        "\n'node1: 0\nnode2: any\nnode3: 1'"
    proper_node_state_text = limited_node_state_range_text

    if raw_input_initial_state is None:

        raise ValueError("Initial state missing.")

    # Initialize input as dict if it's empty.
    raw_input_initial_state = raw_input_initial_state or dict()

    # Validate that input is a dict.
    if not isinstance(raw_input_initial_state, dict):
        err_msg = compile_err_msg(bad_format_text, {section_name_text: section_name})
        raise ValueError(err_msg)

    # Validate no duplicate node name (up to whitespace).
    input_initial_state = validate_raw_dict(raw_input_initial_state, node_name_text,
                                            {section_name_text: section_name})

    for node_name in input_initial_state.keys() - set(node_names):
        err_msg = compile_err_msg("Unknown node '{}'.".format(node_name),
                                  {section_name_text: section_name})
        raise ValueError(err_msg)

    initial_state = [False] * len(node_names)
    initial_state_variations = []

    for node, node_name in enumerate(node_names):
        try:
            raw_node_state = input_initial_state[node_name]
        except KeyError:
            err_msg = compile_err_msg("Missing initial '{}' state.".format(node_name),
                                      {section_name_text: section_name})
            raise ValueError(err_msg)

        # Validate that node state is a YAML-compliant string.
        if not isinstance(raw_node_state, str):
            err_msg = compile_err_msg(
                "Initial node state must be {}.".format(proper_node_state_text),
                {section_name_text: section_name, node_name_text: node_name})
            raise ValueError(err_msg)

        node_state = raw_node_state.strip()
        if node_state == '1':
            initial_state[node] = True
        elif node_state == '0':
            pass
        elif true_or_false_node_state_pattern.match(node_state):
            initial_state_variations.append(node)
        else:
            err_msg = compile_err_msg(
                "Bad initial node state '{}', must be {}.".format(node_state, proper_node_state_text),
                {section_name_text: section_name, node_name_text: node_name})
            raise ValueError(err_msg)

    return initial_state, initial_state_variations


def parse_raw_input_fixed_nodes(raw_input_fixed_nodes, node_names, mode):
    """
    Parse and validate fixed nodes from the input.

    :param raw_input_fixed_nodes: dict of raw node states by
        raw node names
    :param mode: BoolSi mode
    :param node_names: node names
    :return: (fixed nodes, fixed node variations)
    """
    fixed_nodes = dict()
    fixed_nodes_variations = []

    section_name = "fixed nodes"
    proper_fixed_node_state_text = \
        constant_node_state_range_text if mode == Mode.ATTRACT else full_node_state_range_text
    extended_proper_fixed_node_state_msg = \
        proper_fixed_node_state_text + "in {} mode".format(mode_descriptions[mode])

    # Initialize input as dict if it's missing or empty.
    raw_input_fixed_nodes = raw_input_fixed_nodes or dict()

    # Validate that input is a dict.
    if not isinstance(raw_input_fixed_nodes, dict):
        sample_input_fixed_nodes = \
            "'node2: 0\nnode7: {}\nnode8: 1'".format("1" if mode == Mode.ATTRACT else "any")
        err_msg = compile_err_msg(
            "Expected mappings of nodes to their fixed states. Example:\n{}".format(
                sample_input_fixed_nodes),
            {section_name_text: section_name})
        raise ValueError(err_msg)

    # Validate no duplicate node name (up to whitespace).
    input_fixed_nodes = validate_raw_dict(raw_input_fixed_nodes, node_name_text,
                                          {section_name_text: section_name})

    for node_name, raw_node_state in input_fixed_nodes.items():
        try:
            node = node_names.index(node_name)
        except ValueError:
            err_msg = compile_err_msg(
                "Unknown node '{}'.".format(node_name), {section_name_text: section_name})
            raise ValueError(err_msg)

        if not isinstance(raw_node_state, str):
            err_msg = compile_err_msg(
                "Fixed node state must be {}.".format(extended_proper_fixed_node_state_msg),
                {section_name_text: section_name, node_name_text: node_name})
            raise ValueError(err_msg)

        node_state = raw_node_state.strip()

        if mode == Mode.ATTRACT and nonconstant_node_state_pattern.match(node_state):
            err_msg = compile_err_msg(
                "Fixed node state '{}' is forbidden in '{}' mode. Must be {}.".format(
                    node_state.lower(), mode_descriptions[mode], proper_fixed_node_state_text),
                {section_name_text: section_name, node_name_text: node_name})
            raise ValueError(err_msg)
        elif node_state == '0':
            fixed_nodes[node] = False
        elif node_state == '0?':
            heappush(fixed_nodes_variations, (node, NodeStateRange.MAYBE_FALSE))
        elif node_state == '1':
            fixed_nodes[node] = True
        elif node_state == '1?':
            heappush(fixed_nodes_variations, (node, NodeStateRange.MAYBE_TRUE))
        elif true_or_false_node_state_pattern.match(node_state):
            # Initialize to False in fixed nodes.
            fixed_nodes[node] = False
            # But also add to fixed node variations to variate this node.
            heappush(fixed_nodes_variations, (node, NodeStateRange.TRUE_OR_FALSE))
        elif maybe_true_or_false_node_state_pattern.match(node_state):
            heappush(fixed_nodes_variations, (node, NodeStateRange.MAYBE_TRUE_OR_FALSE))
        else:
            err_msg = compile_err_msg(
                "Bad fixed node state '{}'. Must be {}.".format(
                    node_state, extended_proper_fixed_node_state_msg),
                {section_name_text: section_name, node_name_text: node_name})
            raise ValueError(err_msg)

    return fixed_nodes, fixed_nodes_variations


def parse_raw_input_perturbations(raw_input_perturbations, node_names, mode, max_t):
    """
    Parse and validate perturbations from the input.

    :param raw_input_perturbations: dict of raw
        perturbation dicts by raw node names
    :param node_names: node names
    :param mode: BoolSi mode
    :param max_t: maximum simulation time
    :return: (perturbations, perturbation variations)
    """
    sample_input_times_1 = "1-3,5,7"
    sample_input_times_2 = "4,6"
    sample_input_times_3 = "10"
    times_bad_format_text = \
        "Bad format of perturbation times. Expected sequence of times or time intervals, " \
        "separated with semicolons. Examples: {}, {}, {}.".format(
            sample_input_times_1, sample_input_times_2, sample_input_times_3)
    node_perturbations_sample_input_1 = "'{{0: {}, {}: {}}}'".format(
        sample_input_times_1, '1' if mode == Mode.ATTRACT else 'any?', sample_input_times_2)
    node_perturbations_sample_input_2 = "'{{{}: {}}}'".format(
        '1' if mode == Mode.ATTRACT else 'any', sample_input_times_3)

    perturbations_sample_input = "'node2: {{{}}}\nnode7: {{{}}}'".format(
        node_perturbations_sample_input_1, node_perturbations_sample_input_2)

    perturbed_nodes_by_t = dict()
    perturbations_variations = []

    section_name = "perturbations"
    proper_perturbed_node_state_text = \
        constant_node_state_range_text if mode == Mode.ATTRACT else full_node_state_range_text
    extended_perturbed_node_state_text = \
        proper_perturbed_node_state_text + "in {} mode".format(mode_descriptions[mode])
    node_perturbations_bad_format_text = \
        "Bad format of node perturbations. Expected mappings of perturbed node states to" \
        " the times at which they must occur. Examples: {}, {}.".format(
            node_perturbations_sample_input_1, node_perturbations_sample_input_2)
    bad_format_text = "Expected mappings of nodes to their perturbations. Example:\n{}".format(
        perturbations_sample_input)

    # Initialize input as dict if it's missing or empty.
    raw_input_perturbations = raw_input_perturbations or dict()
    
    # Validate that input is a dict.
    if not isinstance(raw_input_perturbations, dict):
        err_msg = compile_err_msg(bad_format_text, {section_name_text: section_name})

        raise ValueError(err_msg)

    # Validate no duplicate node name (up to whitespace).
    input_perturbations = validate_raw_dict(
        raw_input_perturbations, node_name_text, {section_name_text: section_name})

    for node_name, raw_input_node_perturbations in input_perturbations.items():
        try:
            node = node_names.index(node_name)
        except ValueError:
            err_msg = compile_err_msg(
                "Unknown node '{}'.".format(node_name), {section_name_text: section_name})

            raise ValueError(err_msg)

        # Validate that node perturbations are a dict.
        if not isinstance(raw_input_node_perturbations, dict):
            err_msg = compile_err_msg(
                node_perturbations_bad_format_text,
                {section_name_text: section_name, node_name_text: node_name})

            raise ValueError(err_msg)

        # Validate no duplicate perturbed node state (up to whitespace).
        input_node_perturbations = validate_raw_dict(
            raw_input_node_perturbations, perturbed_node_state_text,
            {section_name_text: section_name, node_name_text: node_name})

        for node_state, raw_times in input_node_perturbations.items():
            # Validate that times are a string.
            if not isinstance(raw_times, str):
                err_msg = compile_err_msg(
                    times_bad_format_text,
                    {section_name_text: section_name, node_name_text: node_name,
                     perturbed_node_state_text: node_state})

                raise ValueError(err_msg)

            for t in parse_raw_input_time_steps(raw_times):
                if t <= 0:
                    err_msg = compile_err_msg(
                        "Bad perturbation time {}. Must be positive.",
                        {section_name_text: section_name, node_name_text: node_name,
                         perturbed_node_state_text: node_state})

                    raise ValueError(err_msg)

                if t > max_t:
                    err_msg = compile_err_msg(
                        "Bad perturbation time {}. Must not exceed simulation "
                        "length {}.".format(t, max_t),
                        {section_name_text: section_name, node_name_text: node_name,
                         perturbed_node_state_text: node_state})

                    raise ValueError(err_msg)

                # Validate no perturbations overlap.
                if node in perturbed_nodes_by_t.get(t, dict()) or \
                        sum((t, node, node_state_range) in perturbations_variations for
                            node_state_range in NodeStateRange) > 0:
                    err_msg = compile_err_msg(
                        "Perturbations at time {} overlap.".format(t),
                        {section_name_text: section_name, node_name_text: node_name})

                    raise ValueError(err_msg)

                if mode == Mode.ATTRACT and nonconstant_node_state_pattern.match(node_state):
                    err_msg = compile_err_msg(
                        "Perturbed node state '{}' is forbidden in '{}' mode. Must be {}.".format(
                            node_state.lower(), mode_descriptions[mode], proper_perturbed_node_state_text),
                        {section_name_text: section_name, node_name_text: node_name})

                    raise ValueError(err_msg)

                elif node_state == '0':
                    try:
                        perturbed_nodes_by_t[t][node] = False
                    except KeyError:
                        perturbed_nodes_by_t[t] = {node: False}
                elif node_state == '0?':
                    heappush(perturbations_variations, (t, node, NodeStateRange.MAYBE_FALSE))
                elif node_state == '1':
                    try:
                        perturbed_nodes_by_t[t][node] = True
                    except KeyError:
                        perturbed_nodes_by_t[t] = {node: True}
                elif node_state == '1?':
                    heappush(perturbations_variations, (t, node, NodeStateRange.MAYBE_TRUE))
                elif true_or_false_node_state_pattern.match(node_state):
                    # Initialize as False.
                    try:
                        perturbed_nodes_by_t[t][node] = False
                    except KeyError:
                        perturbed_nodes_by_t[t] = {node: False}
                    # But also add into perturbation variations to variate this perturbation.
                    heappush(perturbations_variations, (t, node, NodeStateRange.TRUE_OR_FALSE))
                elif maybe_true_or_false_node_state_pattern.match(node_state):
                    heappush(perturbations_variations, (t, node, NodeStateRange.MAYBE_TRUE_OR_FALSE))
                else:
                    err_msg = compile_err_msg(
                        "Bad perturbed node state '{}'. Must be {}.".format(
                            node_state, extended_perturbed_node_state_text),
                        {section_name_text: section_name, node_name_text: node_name})

                    raise ValueError(err_msg)

    return perturbed_nodes_by_t, perturbations_variations


def parse_raw_input_target_state(raw_input_target_state, node_names, mode):
    """
    Parse and validate target state from the input.

    :param raw_input_target_state: dict of raw node states by
        raw node names
    :param node_names: node names
    :param mode: BoolSi mode
    :return: (code of target state, set of nodes from target state)
    """
    section_name = 'target state'
    bad_format_text = \
        "Expected mappings of nodes to their target states. " \
        "Example:\n'node1: 0\nnode2: 1\nnode3: any'"
    proper_node_state_text = limited_node_state_range_text

    if mode == Mode.TARGET:

        if raw_input_target_state is None:

            raise ValueError("Target state missing.")

        # Initialize input as dict if it's empty.
        raw_input_target_state = raw_input_target_state or dict()

        # Validate that input is a dict.
        if not isinstance(raw_input_target_state, dict):
            err_msg = compile_err_msg(bad_format_text, {section_name_text: section_name})
            raise ValueError(err_msg)

        # Validate no duplicate node name (up to whitespace).
        input_target_state = validate_raw_dict(raw_input_target_state, node_name_text,
                                               {section_name_text: section_name})

        for node_name in input_target_state.keys() - set(node_names):
            err_msg = compile_err_msg("Unknown node '{}'.".format(node_name),
                                      {section_name_text: section_name})
            raise ValueError(err_msg)

        dummy_target_state = [False] * len(node_names)
        target_node_set = set(range(len(node_names)))
        for node, node_name in enumerate(node_names):
            try:
                raw_node_state = input_target_state[node_name]
            except KeyError:
                err_msg = compile_err_msg("Missing target '{}' state.".format(node_name),
                                          {section_name_text: section_name})

                raise ValueError(err_msg)

            # Validate that node state is a YAML-compliant string.
            if not isinstance(raw_node_state, str):
                err_msg = compile_err_msg(
                    "Target node state must be {}.".format(proper_node_state_text),
                    {section_name_text: section_name, node_name_text: node_name})

                raise ValueError(err_msg)

            node_state = raw_node_state.strip()
            if node_state == '0':
                pass
            elif node_state == '1':
                dummy_target_state[node] = True
            elif true_or_false_node_state_pattern.match(node_state):
                target_node_set.remove(node)
            else:
                err_msg = compile_err_msg(
                    "Bad target node state '{}'. Must be {}.".format(
                        node_state, proper_node_state_text),
                    {section_name_text: section_name, node_name_text: node_name})

                raise ValueError(err_msg)

        _, target_substate_code = encode_state(target_node_set, dummy_target_state)

        return target_substate_code, target_node_set

    elif raw_input_target_state is not None:
        logging.getLogger().warning(
            "Target state is only used in '{}' mode.".format(mode_descriptions[Mode.TARGET]))

    return None, None


def parse_raw_input_time_steps(raw_input_time_steps):
    """
    Parse and validate string with time step intervals from the
    input. Generates next time step with each call.

    :param raw_input_time_steps: string of raw time intervals
    :return: (generated) time steps
    """
    for raw_input_interval in raw_input_time_steps.split(','):
        input_interval = raw_input_interval.strip()
        if not input_interval:
            continue

        time_step_match = re.match("^(\d+)(?:\s*-\s*(\d+))?$", input_interval)

        if time_step_match is None:

            raise ValueError("'{}' is not a valid time step or interval.".format(input_interval))

        time_step_from_txt = time_step_match.group(1)
        time_step_to_txt = time_step_match.group(2)

        if time_step_to_txt is None:

            yield int(time_step_from_txt)

        else:
            time_step_from = int(time_step_from_txt)
            time_step_to = int(time_step_to_txt)

            if time_step_from > time_step_to:

                raise ValueError("'{}-{}' is not a valid time interval.".format(
                    time_step_from, time_step_to))

            else:
                for i in range(time_step_from, time_step_to + 1):

                    yield i


def parse_input_update_rules(input_update_rules, node_names, section_location_dict):
    """
    Parse dict with update rules into predecessor node lists and truth tables.

    :param input_update_rules: update rules from the input
    :param node_names: node names
    :param section_location_dict: dict for identifying 'update rules'
        section
    :return: (predecessor node lists, truth tables)
    """
    bad_format_text = \
        "Bad format of update rule. Expected valid logical expression, consisting of nodes, " \
        "constants '0' and '1', operators 'and', 'or', 'not', and function 'majority(...)' " \
        "for any number of arguments. Examples: 'node5 and (not node2 or node3)', " \
        "'majority(node1, node2, not node3, node4, 1)'."

    # Validate names of the nodes whose update rules are given.
    for node_name in input_update_rules.keys() - set(node_names):
        err_msg = compile_err_msg("Unknown node '{}'.".format(node_name), section_location_dict)

        raise ValueError(err_msg)

    predecessor_nodes_lists = []
    update_rules = []
    location_dict = section_location_dict.copy()
    for node_name in node_names:
        try:
            update_rule = input_update_rules[node_name]
        except KeyError:
            err_msg = compile_err_msg("Missing '{}' update rule.".format(node_name),
                                      section_location_dict)

            raise ValueError(err_msg)

        location_dict[node_name_text] = node_name

        if not isinstance(update_rule, str):
            err_msg = compile_err_msg(bad_format_text, location_dict)

            raise ValueError(err_msg)

        # Parse incoming node names from the update rule.
        predecessor_node_names_set = parse_predecessor_node_names_from_update_rule(update_rule)

        # Validate incoming node names.
        predecessor_nodes_set = set()
        for predecessor_node_name in predecessor_node_names_set:
            try:
                predecessor_nodes_set.add(node_names.index(predecessor_node_name))
            except ValueError:
                if predecessor_node_name not in {'0', '1'}:
                    err_msg = compile_err_msg(
                        "Unknown expression '{}'.".format(predecessor_node_name), location_dict)

                    raise ValueError(err_msg)

        # Validate majority function is only used for a call.
        if re.search('majority(?!\()', update_rule, re.I):
            err_msg = compile_err_msg(
                'Majority function not followed by parentheses.', section_location_dict)

            raise ValueError(err_msg)

        # Validate no commas outside majority call.
        majority_args_positions = [m.start() + len('majority')
                                   for m in re.finditer('majority', update_rule, re.I)]
        parentheses_level = 0
        is_in_majority_arguments = False
        for position, c in enumerate(update_rule):
            if c == '(':
                parentheses_level += 1
                if position in majority_args_positions:
                    is_in_majority_arguments = True
            elif c == ')' and parentheses_level > 0:
                parentheses_level -= 1
                if parentheses_level == 0:
                    is_in_majority_arguments = False
            elif c == ',' and (not is_in_majority_arguments or parentheses_level != 1):
                err_msg = compile_err_msg(
                    'Unexpected comma.', section_location_dict)

                raise ValueError(err_msg)

        # Validate no trailing commas in majority call.
        if re.search(',\s*\)', update_rule):
            err_msg = compile_err_msg(
                'Last argument missing in majority function call.', section_location_dict)

            raise ValueError(err_msg)

        # Store incoming node list and update rule.
        predecessor_nodes_lists.append(sorted(predecessor_nodes_set))
        update_rules.append(update_rule)

    # Generate safe-to-eval versions of node names.
    safe_node_names = generate_safe_node_names(node_names)

    # Build truth tables from update rules, converting them to comply with Python syntax first.
    truth_tables = []
    for node_name, predecessor_nodes, update_rule in zip(
            node_names, predecessor_nodes_lists, update_rules):
        safe_update_rule = update_rule
        # Substitute incoming node names by their safe-to-eval
        # counterparts, from longest to shortest, as node names can
        # possibly be contained in one another.
        for predecessor_node in sorted(predecessor_nodes, key=lambda i: len(node_names[i]),
                                       reverse=True):
            safe_update_rule = re.sub(re.escape(node_names[predecessor_node]),
                                      safe_node_names[predecessor_node], safe_update_rule)
        # Ensure that all operators comply with Python syntax.
        safe_update_rule = safe_update_rule.lower()
        safe_update_rule = re.sub(r'\b0\b', 'False', safe_update_rule)
        safe_update_rule = re.sub(r'\b1\b', 'True', safe_update_rule)

        # Ensure that all operators comply with Python syntax.

        # Evaluate truth table for the node.
        location_dict[node_name_text] = node_name
        err_msg = compile_err_msg(bad_format_text, location_dict)
        truth_table = build_truth_table_from_safe_update_rule(
            safe_update_rule, predecessor_nodes, safe_node_names, err_msg)

        truth_tables.append(truth_table)

    return predecessor_nodes_lists, truth_tables


def parse_predecessor_node_names_from_update_rule(update_rule):
    """
    Parse set of predecessor node names from an update rule.

    :param update_rule: update rule
    :return: set of predecessor node names
    """
    incoming_node_name_set = {incoming_node_name.strip() for incoming_node_name in
                              update_rule_reserved_symbol_pattern.split(update_rule)}

    return incoming_node_name_set - {''}


def generate_safe_node_names(node_names):
    """
    Generate safe-to-eval counterparts of node names, guaranteed
    distinct from the original ones.

    :param node_names: list of node names
    :return: list of safe node names
    """
    node_name_set = set(node_names)
    safe_node_name_base = "node"
    safe_node_names_are_safe = False
    while not safe_node_names_are_safe:
        safe_node_names = \
            ['{}{}'.format(safe_node_name_base, i) for i in range(len(node_names))]
        if set(safe_node_names).intersection(node_name_set):
            safe_node_name_base += "_"
        else:
            safe_node_names_are_safe = True

    return safe_node_names


def build_truth_table_from_safe_update_rule(safe_update_rule, predecessor_nodes, safe_node_names,
                                            err_msg):
    """
    Create truth table dict from converted update rule.

    :param safe_update_rule: string with save-to-eval update rule
    :param predecessor_nodes: list of predecessor nodes in this update rule
    :param safe_node_names: list of safe-to-eval node names, used
        across safe-to-eval update rules
    :param err_msg: bad format error message for this update rule
    :return: truth table
    """
    truth_table = dict()

    for predecessor_node_states in itertools.product((False, True), repeat=len(predecessor_nodes)):
        for i, predecessor_node in enumerate(predecessor_nodes):
            exec("{} = {}".format(safe_node_names[predecessor_node], predecessor_node_states[i]))
        try:
            truth_table[predecessor_node_states] = eval(safe_update_rule)
        # It seems that only SyntaxError exceptions are possible after
        # the validations, but just to be on the safe side.
        except:

            raise ValueError(err_msg)

        if truth_table[predecessor_node_states] not in {False, True}:

            raise ValueError(err_msg)

    return truth_table


def count_simulation_problems(initial_state_variations,
                              fixed_node_variations,
                              perturbation_variations):
    """
    Count combinations to simulate.

    :param initial_state_variations: list with indices of the nodes whose initial states are variated
    :param fixed_node_variations: list of tuples with fixed nodes variations (node, node state range), sorted by node
    :param perturbation_variations: list of perturbations variations, ordered by time steps and then by nodes
    :return: total combination count
    """

    tristate_fixed_node_variations_count = \
        sum(1 for _, node_state_range in fixed_node_variations if
            node_state_range == NodeStateRange.MAYBE_TRUE_OR_FALSE)
    bistate_fixed_node_variations_count = \
        len(fixed_node_variations) - tristate_fixed_node_variations_count
    tristate_perturbation_variations_count = \
        sum(1 for _, _, node_state_range in perturbation_variations if
            node_state_range == NodeStateRange.MAYBE_TRUE_OR_FALSE)
    bistate_perturbation_variations_count = \
        len(perturbation_variations) - tristate_perturbation_variations_count

    # Every variation of initial node state and bistate variation of fixed node or perturbation increases combinations
    # count times 2, while every tristate variation of fixed node or perturbation increases combinations count times 3.
    power_of_two = len(initial_state_variations) + bistate_fixed_node_variations_count + \
        bistate_perturbation_variations_count
    power_of_three = tristate_fixed_node_variations_count + tristate_perturbation_variations_count

    return 2 ** power_of_two * 3 ** power_of_three


def validate_raw_dict(raw_dict, key_text, location_dict):
    """
    Validate dict from the input by raising exceptions for duplicate
    keys.

    :param raw_dict: dict from the input with raw keys
    :param key_text: description of raw_dict keys
    :param location_dict: dict of nested input keys, identifying
        location being parsed
    :return: dict from the input with validated and formatted keys
    """
    dict_with_formatted_keys = dict()
    for raw_key in raw_dict:
        key = raw_key.strip()
        if key in dict_with_formatted_keys:
            err_msg = compile_err_msg("Duplicate {} '{}'.".format(key_text, key), location_dict)

            raise ValueError(err_msg)

        dict_with_formatted_keys[key] = raw_dict[raw_key]

    return dict_with_formatted_keys


def compile_err_msg(problem_text, location_dict):
    """
    Compile error message for a user, exposing part of the input it
    came from.

    :param problem_text: problem description
    :param location_dict: dict of nested input keys, identifying
        problematic location
    :return: error message
    """
    location_str = ', '.join(
        '{} "{}"'.format(key_text, location_dict[key_text]) for key_text in
        (section_name_text, node_name_text, perturbed_node_state_text) if key_text in location_dict)

    return location_str + ":\n" + problem_text if location_str else problem_text


