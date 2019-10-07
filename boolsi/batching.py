"""
Functions for generating batches of simulation problems, to comprise
computational tasks.
"""
import itertools

from .constants import NodeStateRange


def create_numeral_system_from_variations(simulation_problem_variations):
    """
    Return radices and unit place values corresponding to the
    simulation problem variations, so that each simulation problem
    can be represented as a mixed-radix number in this numeral system.

    :param simulation_problem_variations: variations from origin simulation problem
    :return: (simulation_problem_radices, simulation_problem_unit_place_values)
    """
    simulation_problem_radices = []
    simulation_problem_unit_place_values = []
    unit_place_value = 1
    initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations = \
        simulation_problem_variations

    # Fill radices and unit place values from initial state variations (less significant ones come first).
    for _ in initial_state_variations:
        radix = 2
        simulation_problem_radices.append(radix)
        simulation_problem_unit_place_values.append(unit_place_value)
        unit_place_value *= radix

    # Fill radices and unit place values from fixed nodes variations (less significant ones come first).
    for _, node_state_range in fixed_nodes_variations:
        radix = 3 if node_state_range == NodeStateRange.MAYBE_TRUE_OR_FALSE else 2
        simulation_problem_radices.append(radix)
        simulation_problem_unit_place_values.append(unit_place_value)
        unit_place_value *= radix

    # Fill radices and unit place values from perturbations variations (less significant ones come first).
    for _, _, node_state_range in perturbed_nodes_by_t_variations:
        radix = 3 if node_state_range == NodeStateRange.MAYBE_TRUE_OR_FALSE else 2
        simulation_problem_radices.append(radix)
        simulation_problem_unit_place_values.append(unit_place_value)
        unit_place_value *= radix

    return simulation_problem_radices, simulation_problem_unit_place_values


def calculate_increment_for_chunking_simulation_problems(n_simulation_problems, n_chunks):
    """
    Calculate increment for partitioning simulation problems into chunks.
    increment is aimed to be as close to a multiple of n_chunks as
    possible, while not being a multiple of 2 or 3, the only radices
    simulation problem variations can have.

    Starting from the first (0-th) simulation problem and incrementing
    by the increment (modulo n_sims) n_sims times will attain each
    simulation problem exactly once.

    :param n_simulation_problems: number of simulation problems
    :param n_chunks: number of chunks
    :return: increment
    """
    if n_chunks % 2 > 0 and n_chunks % 3 > 0:
        increment = n_chunks
    elif n_chunks % 2 == 1:
        # n_chunks is odd and 0 modulo 3.
        increment = n_chunks - 2
    elif n_chunks % 3 == 2:
        # n_chunks is even and 2 modulo 3.
        increment = n_chunks - 1
    else:
        # n_chunks is even and either 0 or 1 modulo 3.
        increment = n_chunks + 1

    return increment % n_simulation_problems


def generate_simulation_problem_batch_seeds(simulation_problem_variations, n_chunks, n_simulation_problems,
                                            n_simulation_problem_batches_per_chunk):
    """
    Generate seeds from which batches can be constructed. Each batch
    seed is (decomposition of first simulation problem in the batch, #
    of simulation problems in the batch, decomposition of sampling
    increment, decomposition bases).

    Simulation problems are evenly subdivided into chunks, each corresponding
    to a process. Every chunk is then subdivided into the same # of batches
    regulated by the input parameter.

    :param simulation_problem_variations: variations from origin simulation problem
    :param n_chunks: number of chunks to evenly split all simulation problems into
    :param n_simulation_problems: number of simulations to perform
    :param n_simulation_problem_batches_per_chunk: number of batches to evenly split each chunk into
    :return: batch seeds
    """
    # Set up multibase decomposition of a simulation (less significant positions come first).

    simulation_problem_radices, simulation_problem_unit_place_values = \
        create_numeral_system_from_variations(simulation_problem_variations)

    # Calculate the # of bigger chunks, and the size of both bigger and
    # smaller chunks.
    smaller_simulation_problem_chunk_size, n_bigger_simulation_problem_chunks = divmod(
        n_simulation_problems, n_chunks)
    bigger_simulation_problem_chunk_size = smaller_simulation_problem_chunk_size + 1
    # Calculate the # and size of bigger and smaller batches in the chunks.
    smaller_simulation_problem_batch_size, n_bigger_simulation_problem_batches_in_smaller_chunk = divmod(
        smaller_simulation_problem_chunk_size, n_simulation_problem_batches_per_chunk)
    bigger_simulation_problem_batch_size = smaller_simulation_problem_batch_size + 1
    n_bigger_simulation_problem_batches_in_bigger_chunk = \
        n_bigger_simulation_problem_batches_in_smaller_chunk + 1

    # Calculate increment for subdividing simulation problems into chunks.
    increment = calculate_increment_for_chunking_simulation_problems(
        n_simulation_problems, n_chunks)
    # Decompose increment by variations.
    increment_variational_representation = convert_number_to_variational_representation(
        increment, simulation_problem_radices, simulation_problem_unit_place_values)

    for simulation_problem_batch_index, simulation_problem_chunk_index in \
            itertools.product(range(n_simulation_problem_batches_per_chunk), range(n_chunks)):
        # Figure what kind of chunk we are in, and how many simulation
        # problems comprise the previous chunks.
        if simulation_problem_chunk_index < n_bigger_simulation_problem_chunks:
            n_bigger_simulation_problem_batches = n_bigger_simulation_problem_batches_in_bigger_chunk
            n_simulation_problems_in_previous_chunks = \
                simulation_problem_chunk_index * bigger_simulation_problem_chunk_size
        else:
            n_bigger_simulation_problem_batches = n_bigger_simulation_problem_batches_in_smaller_chunk
            n_simulation_problems_in_previous_chunks = \
                n_bigger_simulation_problem_chunks * bigger_simulation_problem_chunk_size + \
                (simulation_problem_chunk_index - n_bigger_simulation_problem_chunks) * smaller_simulation_problem_chunk_size
        # Figure what kind of batch we are in, and how many simulation
        # problems comprise the previous batches in the chunk.
        if simulation_problem_batch_index < n_bigger_simulation_problem_batches:
            n_batched_simulation_problems_in_chunk = simulation_problem_batch_index * bigger_simulation_problem_batch_size
            simulation_problem_batch_size = bigger_simulation_problem_batch_size
        else:
            n_batched_simulation_problems_in_chunk = \
                n_bigger_simulation_problem_batches * bigger_simulation_problem_batch_size + \
                (simulation_problem_batch_index - n_bigger_simulation_problem_batches) * smaller_simulation_problem_batch_size
            simulation_problem_batch_size = smaller_simulation_problem_batch_size
            # Terminate once reached zero-sized batches.
            if simulation_problem_batch_size == 0:
                break

        # Find index of the first simulation problem in the batch.
        first_simulation_problem_index = \
            increment * (n_simulation_problems_in_previous_chunks + n_batched_simulation_problems_in_chunk)
        first_simulation_problem_index %= n_simulation_problems
        # Decompose the first simulation problem by variations.
        first_simulation_problem_variational_representation = convert_number_to_variational_representation(
            first_simulation_problem_index, simulation_problem_radices, simulation_problem_unit_place_values)
        # Generate seed of the batch.
        yield first_simulation_problem_variational_representation, simulation_problem_batch_size, \
              increment_variational_representation, simulation_problem_radices


def convert_variational_representation_to_simulation_problem(variational_representation,
                                                             origin_simulation_problem,
                                                             simulation_problem_variations):
    """
    Create simulation problem based on its decomposition by variations.

    :param variational_representation: simulation problem decomposed by variations
    :param origin_simulation_problem: origin simulation problem
    :param simulation_problem_variations: variations from origin simulation problem
    :return: simulation problem
    """
    node_state_by_digit_by_node_state_range = \
        {NodeStateRange.MAYBE_FALSE: {0: None, 1: False},
         NodeStateRange.MAYBE_TRUE: {0: None, 1: True},
         NodeStateRange.TRUE_OR_FALSE: {0: False, 1: True},
         NodeStateRange.MAYBE_TRUE_OR_FALSE: {0: None, 1: False, 2: True}}

    origin_initial_state, origin_fixed_nodes, origin_perturbed_nodes_by_t = origin_simulation_problem
    initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations = \
        simulation_problem_variations

    initial_state_variational_representation = variational_representation[:len(initial_state_variations)]
    fixed_nodes_variational_representation = \
        variational_representation[len(initial_state_variations):len(initial_state_variations) +
                                                                 len(fixed_nodes_variations)]
    perturbed_nodes_by_t_variational_representation = \
        variational_representation[len(initial_state_variations) + len(fixed_nodes_variations):]

    initial_state = origin_initial_state.copy()
    for node, digit in zip(initial_state_variations, initial_state_variational_representation):
        initial_state[node] = bool(digit)

    fixed_nodes = origin_fixed_nodes.copy()
    for (node, node_state_range), digit in zip(fixed_nodes_variations, fixed_nodes_variational_representation):
        node_state = node_state_by_digit_by_node_state_range[node_state_range][digit]
        if node_state is not None:
            fixed_nodes[node] = node_state

    perturbed_nodes_by_t = {t: perturbed_nodes.copy()
                            for t, perturbed_nodes in origin_perturbed_nodes_by_t.items()}
    for (t, node, node_state_range), digit in zip(perturbed_nodes_by_t_variations,
                                                  perturbed_nodes_by_t_variational_representation):
        node_state = node_state_by_digit_by_node_state_range[node_state_range][digit]
        if node_state is not None:
            try:
                perturbed_nodes_by_t[t][node] = node_state
            except KeyError:
                perturbed_nodes_by_t[t] = {node: node_state}

    return initial_state, fixed_nodes, perturbed_nodes_by_t


def convert_number_to_variational_representation(number,
                                                 radices,
                                                 unit_place_values):
    """
    Decompose a number by mixed-radix decomposition.

    :param number: number to decompose
    :param radices: radices of decomposition
    :param unit_place_values: place values of decomposition
    :return: decomposition of the number
    """
    variational_representation = [0] * len(radices)
    # Determine the digit in each position, corresponding to some variating element.
    for i, unit_place_value in enumerate(reversed(unit_place_values)):
        digit, number = divmod(number, unit_place_value)
        variational_representation[len(radices) - i - 1] = digit

    return variational_representation


def add_variational_representations(variational_representation_1, variational_representation_2, radices):
    """
    Add two decompositions (by variations) as multiradix numbers.

    :param variational_representation_1: first decomposition
    :param variational_representation_2: second decomposition
    :param radices: radices of decomposition
    :return: summation result (decomposition)
    """
    overflow = 0
    result = []

    for digit_1, digit_2, radix in zip(variational_representation_1, variational_representation_2, radices):
        result_digit = digit_1 + digit_2 + overflow

        if result_digit < radix:
            overflow = 0
        else:
            result_digit %= radix
            overflow = 1

        result.append(result_digit)

    return result


def generate_simulation_problems(first_simulation_problem_variational_representation,
                                 simulation_problem_batch_size,
                                 increment_variational_representation,
                                 simulation_problem_radices,
                                 origin_simulation_problem,
                                 simulation_problem_variations):
    """
    Generate all simulation problems of a batch, based on batch seed and
    variation space.

    :param first_simulation_problem_variational_representation: first combination of the batch decomposed by variations
    :param simulation_problem_batch_size: number of combinations in the batch
    :param increment_variational_representation: sampling step decomposed by variations
    :param simulation_problem_radices: bases of multibase decomposition by variations
    :param origin_simulation_problem: origin simulation problem
    :param simulation_problem_variations: variations from origin simulation problem
    :return: generated combinations of the batch
    """
    simulation_problem_variational_representation = first_simulation_problem_variational_representation.copy()

    for _ in range(simulation_problem_batch_size):
        yield convert_variational_representation_to_simulation_problem(
            simulation_problem_variational_representation, origin_simulation_problem, simulation_problem_variations)
        simulation_problem_variational_representation = add_variational_representations(
            simulation_problem_variational_representation, increment_variational_representation, simulation_problem_radices)


def generate_tasks(origin_simulation_problem,
                   simulation_problem_variations,
                   predecessor_node_lists,
                   truth_tables,
                   n_simulation_problem_chunks,
                   n_simulation_problems,
                   n_simulation_problem_batches_per_chunk):
    """
    Generate batch seeds and passes seeds and other required arguments into the batch processors,
    including mode-specific arguments.

    :param origin_simulation_problem: origin simulation problem
    :param simulation_problem_variations: variations from origin simulation problem
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :param n_simulation_problem_chunks: number of chunks
    :param n_simulation_problems: number of simulations to perform
    :param n_simulation_problem_batches_per_chunk: number of batches to evenly split each chunk into
    :return: (batch seed, batch_index, predecessor node lists, truth tables)
    """
    for simulation_problem_batch_index, \
        (first_simulation_problem_variational_representation, simulation_problem_batch_size,
         increment_variational_representation, simulation_problem_radices) in enumerate(
        generate_simulation_problem_batch_seeds(
            simulation_problem_variations, n_simulation_problem_chunks, n_simulation_problems,
            n_simulation_problem_batches_per_chunk)):

        yield (first_simulation_problem_variational_representation, simulation_problem_batch_size,
               increment_variational_representation, simulation_problem_radices,
               origin_simulation_problem, simulation_problem_variations), \
              simulation_problem_batch_index, predecessor_node_lists, truth_tables


def count_simulation_problem_batches(n_simulation_problem_chunks, n_simulation_problems,
                                     n_simulation_problem_batches_per_chunk):
    """
    Count batches to be generated.

    :param n_simulation_problem_chunks: number of chunks of simulation problems
    :param n_simulation_problems: number of simulations to perform
    :param n_simulation_problem_batches_per_chunk: number of batches to evenly split each chunk into
    :return: number of batches
    """
    return min(n_simulation_problem_chunks * n_simulation_problem_batches_per_chunk, n_simulation_problems)
