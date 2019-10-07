from itertools import product
import math

from boolsi.constants import NodeStateRange
from boolsi.testing_tools import count_simulation_problems, generate_test_description
from boolsi.batching import calculate_increment_for_chunking_simulation_problems, \
    generate_simulation_problem_batch_seeds, convert_variational_representation_to_simulation_problem, \
    convert_number_to_variational_representation, add_variational_representations, \
    generate_simulation_problems, count_simulation_problem_batches


def test_calculate_increment_for_chunking_simulation_problems_A():
    """
    `calculate_increment_for_chunking_simulation_problems`
    Feature A: sampling step is coprime to # of simulation problems.
    """
    # Test for the # of chunks in {1, 2, 3, > 3 & even & 0 mod 3,
    # > 3 & even & 1 mod 3, > 3 & even & 2 mod 3, > 3 & odd & 0 mod 3,
    # > 3 & odd & 1 mod 3, > 3 & odd & 2 mod 3}.
    n_chunks_1 = 1
    n_chunks_2 = 2
    n_chunks_3 = 3
    n_chunks_4 = 6
    n_chunks_5 = 10
    n_chunks_6 = 8
    n_chunks_7 = 9
    n_chunks_8 = 7
    n_chunks_9 = 5
    # Test for the # of simulation problems in {less than the # of
    # chunks (if possible), no less than the # of chunks}.
    n_simulation_problems_1 = 1
    n_simulation_problems_2 = 24

    for n_chunks, n_simulation_problems in product(
            [n_chunks_1, n_chunks_2, n_chunks_3, n_chunks_4, n_chunks_5, n_chunks_6,
             n_chunks_7, n_chunks_8, n_chunks_9],
            [n_simulation_problems_1, n_simulation_problems_2]):
        sampling_step = calculate_increment_for_chunking_simulation_problems(
            n_simulation_problems, n_chunks)
        gcd = math.gcd(sampling_step, n_simulation_problems)

        test_description = generate_test_description(
            locals(), 'n_chunks', 'n_simulation_problems')
        assert gcd == 1, test_description

def test_calculate_increment_for_chunking_simulation_problems_B():
    """
    `calculate_increment_for_chunking_simulation_problems`
    Feature B: sampling step is > 1 whenever
    the # of simulation problems > the # of chunks.
    """
    n_simulation_problems = 12
    # Test for the # of chunks in {even & 0 mod 3, even & 1 mod 3,
    # even & 2 mod 3, odd & 0 mod 3, odd & 1 mod 3, odd & 2 mod 3}.
    n_chunks_1 = 6
    n_chunks_2 = 10
    n_chunks_3 = 8
    n_chunks_4 = 9
    n_chunks_5 = 7
    n_chunks_6 = 5

    for n_chunks in [n_chunks_1, n_chunks_2, n_chunks_3, n_chunks_4, n_chunks_5, n_chunks_6]:
        sampling_step = calculate_increment_for_chunking_simulation_problems(
            n_simulation_problems, n_chunks)

        test_description = generate_test_description(
            locals(), 'n_chunks', 'n_simulation_problems', 'sampling_step')
        assert sampling_step > 1, test_description

def test_generate_simulation_problem_batch_seeds_A():
    """
    `generate_simulation_problem_batch_seeds`
    Feature A: generating batch seeds.
    """
    initial_state_variations = [3, 5]
    fixed_nodes_variations = [(1, NodeStateRange.MAYBE_FALSE),
                              (2, NodeStateRange.TRUE_OR_FALSE),
                              (4, NodeStateRange.MAYBE_TRUE_OR_FALSE)]
    perturbed_nodes_by_t_variations = [(2, 0, NodeStateRange.TRUE_OR_FALSE)]
    n_simulation_problems = count_simulation_problems(
        initial_state_variations, fixed_nodes_variations,
        perturbed_nodes_by_t_variations)

    # Testing for {no bigger chunks & no bigger batches in smaller chunks,
    # no smaller batches in bigger chunks & no bigger batches in smaller chunks,
    # no bigger batches in smaller chunks, no bigger chunks, no smaller batches
    # in bigger chunks, everything}.
    n_chunks_1 = 4
    batches_per_chunk_1 = 1
    n_chunks_2 = 1
    batches_per_chunk_2 = 5
    n_chunks_3 = 5
    batches_per_chunk_3 = 1
    n_chunks_4 = 5
    batches_per_chunk_4 = 2
    n_chunks_5 = 9
    batches_per_chunk_5 = 2
    n_chunks_6 = 5
    batches_per_chunk_6 = 3

    expected_decomposition_bases = [2, 2, 2, 2, 3, 2]
    expected_batch_sizes_1 = [24] * 4
    expected_initial_states_decompositions_1 = \
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1]]
    expected_sampling_step_decomposition_1 = [1, 0, 1, 0, 0, 0]
    expected_batch_sizes_2 = [20] + [19] * 4
    expected_initial_states_decompositions_2 = \
        [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0], [1, 1, 1, 0, 2, 0], [0, 1, 0, 1, 0, 1],
         [1, 0, 1, 1, 1, 1]]
    expected_sampling_step_decomposition_2 = [1, 0, 0, 0, 0, 0]
    expected_batch_sizes_3 = [20] + [19] * 4
    expected_initial_states_decompositions_3 = \
        [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0]]
    expected_sampling_step_decomposition_3 = [1, 0, 1, 0, 0, 0]
    expected_batch_sizes_4 = [10] * 6 + [9] * 4
    expected_initial_states_decompositions_4 = \
        [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1],
         [0, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1]]
    expected_sampling_step_decomposition_4 = [1, 0, 1, 0, 0, 0]
    expected_batch_sizes_5 = [6] * 6 + [5] * 12
    expected_initial_states_decompositions_5 = \
        [[0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 2, 0],
         [0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 1],
         [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 2, 0], [1, 1, 1, 0, 1, 0], [0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 2, 1], [0, 1, 1, 1, 0, 1], [1, 1, 0, 1, 2, 0], [1, 0, 0, 0, 1, 0],
         [1, 1, 1, 0, 2, 1], [1, 0, 1, 1, 0, 1]]
    expected_sampling_step_decomposition_5 = [1, 1, 1, 0, 0, 0]
    expected_batch_sizes_6 = [7] * 6 + [6] * 9
    expected_initial_states_decompositions_6 = \
        [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 2, 0], [1, 1, 1, 0, 2, 0], [0, 1, 1, 0, 2, 0],
         [1, 0, 1, 0, 2, 0], [0, 0, 1, 0, 2, 0], [0, 1, 1, 0, 1, 1], [1, 0, 1, 0, 1, 1],
         [0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1]]
    expected_sampling_step_decomposition_6 = [1, 0, 1, 0, 0, 0]

    for (n_chunks, batches_per_chunk), (expected_batch_sizes, expected_batch_seed_initial_states,
                                        expected_sampling_step_decomposition) in zip(
        zip([n_chunks_1, n_chunks_2, n_chunks_3, n_chunks_4, n_chunks_5, n_chunks_6],
            [batches_per_chunk_1, batches_per_chunk_2, batches_per_chunk_3,
             batches_per_chunk_4, batches_per_chunk_5, batches_per_chunk_6]),
        zip([expected_batch_sizes_1, expected_batch_sizes_2, expected_batch_sizes_3,
             expected_batch_sizes_4, expected_batch_sizes_5, expected_batch_sizes_6],
            [expected_initial_states_decompositions_1, expected_initial_states_decompositions_2,
             expected_initial_states_decompositions_3, expected_initial_states_decompositions_4,
             expected_initial_states_decompositions_5, expected_initial_states_decompositions_6],
            [expected_sampling_step_decomposition_1, expected_sampling_step_decomposition_2,
             expected_sampling_step_decomposition_3, expected_sampling_step_decomposition_4,
             expected_sampling_step_decomposition_5, expected_sampling_step_decomposition_6])):

        expected_batch_seeds = \
            [(expected_batch_seed_initial_state, expected_batch_size,
              expected_sampling_step_decomposition, expected_decomposition_bases)
             for expected_batch_size, expected_batch_seed_initial_state in zip(
                expected_batch_sizes, expected_batch_seed_initial_states)]

        batch_seeds = list(generate_simulation_problem_batch_seeds(
            (initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations),
            n_chunks, n_simulation_problems, batches_per_chunk))

        test_description = generate_test_description(
            locals(), 'n_chunks', 'batches_per_chunk')
        assert expected_batch_seeds == batch_seeds, test_description

def test_generate_simulation_problem_batch_seeds_B():
    '''
    `generate_simulation_problem_batch_seeds`
    Feature B: not generating seeds for zero-sized batches.
    '''
    initial_state_variations = [3, 5]
    fixed_nodes_variations = []
    perturbed_nodes_by_t_variations = [(2, 0, NodeStateRange.TRUE_OR_FALSE)]
    n_simulation_problems = count_simulation_problems(
        initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations)
    # Testing for {no bigger chunks, no zero-sized batches in
    # bigger chunks, zero-sized batches in bigger chunks}.
    n_chunks_1 = 4
    batches_per_chunk_1 = 3
    n_chunks_2 = 3
    batches_per_chunk_2 = 3
    n_chunks_3 = n_chunks_2
    batches_per_chunk_3 = 4

    expected_batch_size = 1
    expected_decomposition_bases = [2, 2, 2]
    expected_initial_states_decompositions_1 = \
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]]
    expected_sampling_step_decomposition_1 = [1, 0, 1]
    expected_initial_states_decompositions_2 = \
        [[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
    expected_sampling_step_decomposition_2 = [1, 0, 0]
    expected_initial_states_decompositions_3 = \
        expected_initial_states_decompositions_2
    expected_sampling_step_decomposition_3 = expected_sampling_step_decomposition_2

    for (n_chunks, batches_per_chunk), (expected_initial_states_decompositions,
                                        expected_sampling_step_decomposition) in zip(
        zip([n_chunks_1, n_chunks_2, n_chunks_3],
            [batches_per_chunk_1, batches_per_chunk_2, batches_per_chunk_3]),
        zip([expected_initial_states_decompositions_1, expected_initial_states_decompositions_2,
             expected_initial_states_decompositions_3],
            [expected_sampling_step_decomposition_1, expected_sampling_step_decomposition_2,
             expected_sampling_step_decomposition_3])):

        expected_batch_seeds = \
            [(expected_initial_state_decomposition, expected_batch_size,
              expected_sampling_step_decomposition, expected_decomposition_bases)
             for expected_initial_state_decomposition in expected_initial_states_decompositions]

        batch_seeds = list(generate_simulation_problem_batch_seeds(
            (initial_state_variations, fixed_nodes_variations, perturbed_nodes_by_t_variations),
            n_chunks, n_simulation_problems, batches_per_chunk))

        test_description = generate_test_description(
            locals(), 'n_chunks', 'batches_per_chunk')
        assert expected_batch_seeds == batch_seeds, test_description


def test_convert_variational_representation_to_simulation_problem_A():
    """
    `convert_variational_representation_to_simulation_problem`
    Feature A: creating simulation problem from its variational
        representation.
    """
    basic_initial_state = [False] * 6
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

        expected_initial_state = [False] * 5 + [bool(initial_state_variations)]
        expected_fixed_nodes = dict()
        if maybe_zero_fixed_nodes_variations:
            expected_fixed_nodes[0] = False
        if maybe_one_fixed_nodes_variations:
            expected_fixed_nodes[1] = True
        if maybe_zero_or_one_fixed_nodes_variations:
            expected_fixed_nodes[2] = False
        if zero_or_one_fixed_nodes_variations:
            expected_fixed_nodes[3] = True
        expected_perturbed_nodes_by_t = dict()
        if maybe_zero_perturbations_variations:
            try:
                expected_perturbed_nodes_by_t[666][0] = False
            except KeyError:
                expected_perturbed_nodes_by_t[666] = {0: False}
        if maybe_one_perturbations_variations:
            try:
                expected_perturbed_nodes_by_t[666][1] = True
            except KeyError:
                expected_perturbed_nodes_by_t[666] = {1: True}
        if maybe_zero_or_one_perturbations_variations:
            try:
                expected_perturbed_nodes_by_t[666][2] = False
            except KeyError:
                expected_perturbed_nodes_by_t[666] = {2: False}
        if zero_or_one_perturbations_variations:
            try:
                expected_perturbed_nodes_by_t[666][3] = True
            except KeyError:
                expected_perturbed_nodes_by_t[666] = {3: True}

        basic_fixed_nodes = {3: False} if zero_or_one_fixed_nodes_variations else dict()
        fixed_nodes_variations = \
            maybe_zero_fixed_nodes_variations + maybe_one_fixed_nodes_variations + \
            maybe_zero_or_one_fixed_nodes_variations + zero_or_one_fixed_nodes_variations
        basic_perturbed_nodes_by_t = \
            {666: {3: False}} if zero_or_one_perturbations_variations else dict()
        perturbations_variations = \
            maybe_zero_perturbations_variations + maybe_one_perturbations_variations + \
            maybe_zero_or_one_perturbations_variations + zero_or_one_perturbations_variations
        decomposition = [1] * len(initial_state_variations + fixed_nodes_variations + perturbations_variations)

        initial_state, fixed_nodes, perturbed_nodes_by_t = \
            convert_variational_representation_to_simulation_problem(
                decomposition, (basic_initial_state, basic_fixed_nodes, basic_perturbed_nodes_by_t),
                (initial_state_variations, fixed_nodes_variations, perturbations_variations))

        test_description = generate_test_description(
            locals(), 'initial_state_variations', 'maybe_zero_fixed_nodes_variations',
            'maybe_one_fixed_nodes_variations', 'maybe_zero_or_one_fixed_nodes_variations',
            'zero_or_one_fixed_nodes_variations', 'maybe_zero_perturbations_variations',
            'maybe_one_perturbations_variations', 'maybe_zero_or_one_perturbations_variations',
            'zero_or_one_perturbations_variations')
        assert expected_initial_state == initial_state, test_description
        assert expected_fixed_nodes == fixed_nodes, test_description
        assert expected_perturbed_nodes_by_t == perturbed_nodes_by_t, test_description


def test_convert_number_to_variational_representation_A():
    """
    `convert_number_to_variational_representation()`
    Feature A: decomposing number into multibase representation.
    COMPLETE.
    """
    # Test for {bases of 2 only, bases of 3 only, mixed bases}.
    number_1 = 10
    decomposition_bases_1 = [2, 2, 2, 2]
    decomposition_place_values_1 = [1, 2, 4, 8]
    number_2 = 41
    decomposition_bases_2 = [3, 3, 3, 3]
    decomposition_place_values_2 = [1, 3, 9, 27]
    number_3 = 30
    decomposition_bases_3 = [2, 3, 2, 3]
    decomposition_place_values_3 = [1, 2, 6, 12]

    expected_decomposition_1 = [0, 1, 0, 1]
    expected_decomposition_2 = [2, 1, 1, 1]
    expected_decomposition_3 = [0, 0, 1, 2]

    for number, decomposition_bases, decomposition_place_values, expected_decomposition in zip(
            [number_1, number_2, number_3],
            [decomposition_bases_1, decomposition_bases_2, decomposition_bases_3],
            [decomposition_place_values_1, decomposition_place_values_2, decomposition_place_values_3],
            [expected_decomposition_1, expected_decomposition_2, expected_decomposition_3]):

        decomposition = convert_number_to_variational_representation(
            number, decomposition_bases, decomposition_place_values)

        test_description = generate_test_description(
            locals(), 'number', 'decomposition_bases', 'decomposition_place_values')
        assert expected_decomposition == decomposition, test_description


def test_add_variational_representations_A():
    """
    `add_variational_representations`
    Feature A: adding two multibase representations.
    """
    first_decomposition = [1, 0, 1]
    # Test for {bases of 2 only, bases of 3 only, mixed bases}.
    bases_1 = [2] * 3
    bases_2 = [3] * 3
    bases_3 = [3] * 2 + [2] * 1
    # Test for {no overflows, overflows but not in the most
    # significant digit, overflow in the most significant digit}.
    digit_to_increase_1 = 1
    digit_to_increase_2 = 2
    digit_to_increase_3 = 0

    for bases, digit_to_increase in product(
            [bases_1, bases_2, bases_3],
            [digit_to_increase_1, digit_to_increase_2, digit_to_increase_3]):
        digit_increase = bases[digit_to_increase] - 1
        second_decomposition = \
            [digit_increase if digit == digit_to_increase else 0 for digit in range(3)]

        expected_decomposition = first_decomposition.copy()
        overflow, expected_decomposition[digit_to_increase] = divmod(
            digit_increase + first_decomposition[digit_to_increase], bases[digit_to_increase])
        if overflow > 0 and digit_to_increase < 2:
            expected_decomposition[digit_to_increase + 1] = overflow

        decomposition = add_variational_representations(
            first_decomposition, second_decomposition, bases)

        test_description = generate_test_description(locals(), 'bases', 'second_decomposition')
        assert expected_decomposition == decomposition, test_description


def test_generate_simulation_problems_A():
    """
    `generate_simulation_problems`
    Feature A: generating all simulation problems in a batch from seed.
    """
    batch_size = 3
    first_simulation_problem_decomposition = [1, 2, 0, 1]
    sampling_step_decomposition = [1, 2, 0, 0]
    decomposition_bases = [2, 3, 2, 2]
    basic_initial_state = [False] * 6
    initial_state_variations = []
    basic_fixed_nodes = {0: False}
    fixed_nodes_variations = [(1, NodeStateRange.MAYBE_TRUE),
                              (4, NodeStateRange.MAYBE_TRUE_OR_FALSE)]
    basic_perturbed_nodes_by_t = {20: {3: False}}
    perturbations_variations = [(5, 1, NodeStateRange.MAYBE_FALSE),
                                (20, 3, NodeStateRange.TRUE_OR_FALSE)]

    expected_simulation_problems = \
        [(basic_initial_state, {0: False, 1: True, 4: True}, {20: {3: True}}),
         (basic_initial_state, {0: False, 4: True}, {5: {1: False}, 20: {3: True}}),
         (basic_initial_state, {0: False, 1: True, 4: False}, {20: {3: False}})]

    simulation_problems = list(generate_simulation_problems(
        first_simulation_problem_decomposition, batch_size, sampling_step_decomposition,
        decomposition_bases, (basic_initial_state, basic_fixed_nodes, basic_perturbed_nodes_by_t),
        (initial_state_variations, fixed_nodes_variations, perturbations_variations)))

    assert expected_simulation_problems == simulation_problems


def test_count_simulation_problem_batches_A():
    """
    `count_simulation_problem_batches`
    Feature A: counting simulation problem batches to be generated.
    """
    n_simulation_problems = 100
    n_chunks = 10
    # Test for {no zero-sized batches, zero-sized batches}.
    batches_per_chunk_1 = 10
    batches_per_chunk_2 = 11

    expected_n_simulation_problem_batches = 100

    for batches_per_chunk in [batches_per_chunk_1, batches_per_chunk_2]:
        n_simulation_problem_batches = count_simulation_problem_batches(
            n_chunks, n_simulation_problems, batches_per_chunk)

        test_description = generate_test_description(locals(), 'batches_per_chunk')
        assert expected_n_simulation_problem_batches == n_simulation_problem_batches
