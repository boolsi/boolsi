"""
Contains model-specific parts of the logic.
"""


def majority(*args):
    """
    Evaluate majority function on arbitrary number of inputs.

    :param args: tuple of Boolean inputs
    :return: [bool] value of majority function
    """
    return sum(args) > len(args)/2


def apply_update_rules(state, predecessor_node_lists, truth_tables):
    """
    Update network state according to the update rule.

    :param state: current network state
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node
        states, value: resulting node state)
    :return: next network state
    """
    return [truth_tables[node][tuple(state[predecessor_node]
                                     for predecessor_node in predecessor_node_lists[node])]
            for node in range(len(state))]


def adjust_update_rules_for_fixed_nodes(predecessor_node_lists, truth_tables, fixed_nodes):
    """
    Adjust "update rules" matrix and its free element vector so that the fixed nodes will end up in their fixed
    states on each time step automatically, with no manual interventions required.

    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states, value: resulting node state)
    :param fixed_nodes: dict with fixed nodes (key: node, value: node state)
    :return: (predecessor node lists and truth tables, adjusted with respect to fixed nodes)
    """

    adjusted_predecessor_node_lists = \
        [predecessor_nodes.copy() for predecessor_nodes in predecessor_node_lists]
    adjusted_truth_tables = [truth_table.copy() for truth_table in truth_tables]
    for node, node_state in fixed_nodes.items():
        adjusted_predecessor_node_lists[node] = []
        adjusted_truth_tables[node] = {(): node_state}

    return adjusted_predecessor_node_lists, adjusted_truth_tables


def simulate_step(current_state,
                  predecessor_node_lists,
                  truth_tables,
                  next_perturbed_nodes=dict()):
    """
    Update network state by one time step (execute update rule and
    apply perturbations).

    :param current_state: [list] state at current time t
    :param predecessor_node_lists: [list] predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :param next_perturbed_nodes: [dict] perturbations at t+1
    :return: state at t+1
    """

    next_state = apply_update_rules(current_state, predecessor_node_lists, truth_tables)

    for node, node_state in next_perturbed_nodes.items():
        next_state[node] = node_state

    return next_state


def simulate_n_steps(initial_state,
                     perturbed_nodes_by_t,
                     predecessor_node_lists,
                     truth_tables,
                     n_steps):
    """
    Simulate network states for given number of time steps.

    :param initial_state: initial network state
    :param perturbed_nodes_by_t: perturbations
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :param n_steps: number of steps to simulate
    :return: list of all simulated states
    """

    current_state = initial_state
    states = [current_state]

    for t in range(n_steps):
        try:
            next_perturbed_nodes = perturbed_nodes_by_t[t + 1]
        except KeyError:
            next_state = simulate_step(current_state, predecessor_node_lists, truth_tables)
        else:
            next_state = simulate_step(current_state, predecessor_node_lists, truth_tables,
                                       next_perturbed_nodes=next_perturbed_nodes)
        states.append(next_state)
        current_state = next_state

    return states


def simulate_until_no_perturbations(initial_state,
                                    perturbed_nodes_by_t,
                                    predecessor_node_lists,
                                    truth_tables):
    """
    Perform simulation of the minimum number of time steps needed to carry out all the perturbations.

    :param initial_state: state to start simulation with
    :param perturbed_nodes_by_t: [dict] perturbations
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :return: simulated network states
    """
    # Find out for how many time steps to simulate.
    perturbation_max_t = max(perturbed_nodes_by_t) if perturbed_nodes_by_t else 0

    return simulate_n_steps(initial_state, perturbed_nodes_by_t, predecessor_node_lists,
                            truth_tables, perturbation_max_t)


def encode_state(substate_node_set, state):
    """
    Injectively encode state and its substate (with respect to given
    subset of nodes) by (big) numbers.

    :param state: state to encode
    :param substate_node_set: nodes of the substate
    :return: (state code, substate code)
    """
    state_code = 0
    substate_code = 0
    for node, node_state in enumerate(state):
        if node_state:
            code_increase = 1 << node
            state_code += code_increase
            if node in substate_node_set:
                substate_code += code_increase

    return state_code, substate_code


def simulate_until_attractor_or_target_substate_or_max_t(
        storing_all_states, max_t, _encode_state, target_substate_code, initial_state,
        perturbed_nodes_by_t, predecessor_node_lists, truth_tables):
    """
    Perform simulations until attractor is found, or until time cap or
    target substate is reached.

    If storing all states, attractor is found immediately upon repetition
    of any of its states (given all perturbations are carried out).
    Otherwise, the repetition is unlikely to be detected immediately.

    :param storing_all_states: whether to store all states of the simulation
    :param max_t: maximum simulation time
    :param _encode_state: [function] to encode state and substate
    :param target_substate_code: code of the substate to reach
    :param initial_state: initial state
    :param perturbed_nodes_by_t: dict (by time step) of dicts (by node) of
        node states
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :return: simualted states, state codes since last perturbation, time when stopped, \
           attractor found flag, target substate reached flag, attractor reference points
    """
    # Carry out all the perturbations first without considering
    # intermediate states.
    states = simulate_until_no_perturbations(
        initial_state, perturbed_nodes_by_t, predecessor_node_lists, truth_tables)
    last_perturbation_t = len(states) - 1
    state = states[last_perturbation_t]
    state_code, substate_code = _encode_state(state)
    state_codes_since_last_perturbation = [state_code]

    if storing_all_states:
        attractor_reference_points = None
        state_code_set_from_last_perturbation = {state_code}
    else:
        # Initialize reference point at time r, and maximum length L of
        # detectable attractor. Compare next L states to the reference
        # state s(r). If s(r+l) == s(r) for some l <= L, then network has reached
        # attractor of length l at time r.
        attractor_reference_points = [(last_perturbation_t, state, state_code)]
        current_attractor_reference_state_code = state_code
        max_detectable_attractor_l = (len(initial_state) + 1) // 2
        next_attractor_reference_t = last_perturbation_t + max_detectable_attractor_l

    t = last_perturbation_t
    attractor_is_found = False
    target_substate_is_reached = (substate_code == target_substate_code)
    while t < max_t and not attractor_is_found and not target_substate_is_reached:
        t += 1
        # Evaluate next state.
        state = simulate_step(state, predecessor_node_lists, truth_tables)
        state_code, substate_code = _encode_state(state)
        # Check if the state matches target substate.
        target_substate_is_reached = (substate_code == target_substate_code)
        if storing_all_states:
            # Check if the state has already occurred, which means that attractor
            # is found.
            attractor_is_found = (state_code in state_code_set_from_last_perturbation)
            # Store the state.
            states.append(state)
            state_codes_since_last_perturbation.append(state_code)
            state_code_set_from_last_perturbation.add(state_code)
        else:
            # Check if the state repeats reference state, which means
            # that attractor is found.
            attractor_is_found = (state_code == current_attractor_reference_state_code)
            # If reached next reference point and didn't find attractor,
            # update reference points and increase maximum length of
            # discoverable attractor.
            if t == next_attractor_reference_t and not attractor_is_found:
                # Store reached reference point.
                attractor_reference_points.append((t, state, state_code))
                current_attractor_reference_state_code = state_code
                max_detectable_attractor_l *= 2
                next_attractor_reference_t = last_perturbation_t + max_detectable_attractor_l

    # None to represent '...' in states and state codes.
    if not storing_all_states and t > last_perturbation_t:
        states.extend([None, state])
        state_codes_since_last_perturbation.extend([None, state_code])

    return states, state_codes_since_last_perturbation, t, \
           attractor_is_found, target_substate_is_reached, attractor_reference_points


def count_perturbations(perturbed_nodes_by_t):
    """
    Count number of perturbations.

    :param perturbed_nodes_by_t: dict (by time step) of dicts (by node) of node states
    :return: number of perturbations
    """
    return sum(len(current_perturbed_nodes)
               for current_perturbed_nodes in perturbed_nodes_by_t.values())
