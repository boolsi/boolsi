import numpy as np
from scipy import stats
from itertools import product
import ZODB

from boolsi.attractor_analysis import find_node_correlations, compute_frequency_pearsonr, \
    assign_frequency_ranks, compute_frequency_spearmanrho
from boolsi.testing_tools import construct_aggregated_attractor, generate_test_description
from boolsi.attract import init_attractor_db_structure

attractor_states_1 = [[True, True, False, False],
                      [True, False, True, False],
                      [True, False, False, True],
                      [True, False, True, True]]
attractor_states_2 = [[False, True, False, True],
                      [False, False, True, True]]
attractor_states_3 = [[False, True, False, False],
                      [False, False, True, False],
                      [True, True, False, True],
                      [False, True, True, True]]


def test_find_node_correlations_A():
    """
    `find_node_correlations`
    Feature A: correctly terminating if insufficient attractors found.
    """
    # Test for {no attractors found, one attractor found, two attractors found}.
    aggregated_attractors_1 = dict()
    aggregated_attractors_2 = {1: construct_aggregated_attractor(attractor_states_1, 10, 1, 0)}
    aggregated_attractors_3 = {1: construct_aggregated_attractor(attractor_states_1, 1, 1, 0),
                               8: construct_aggregated_attractor(attractor_states_2, 1, 1, 0)}
    for aggregated_attractors in [aggregated_attractors_1, aggregated_attractors_2,
                                  aggregated_attractors_3]:
        db_conn = ZODB.connection(None)
        init_attractor_db_structure(db_conn)
        for key, aggregated_attractor in aggregated_attractors.items():
            db_conn.root.aggregated_attractors[key] = aggregated_attractor
            db_conn.root.n_aggregated_attractors.change(1)
            db_conn.root.total_frequency.change(aggregated_attractor.frequency)

        node_correlations = find_node_correlations(db_conn, None, True)

        test_description = generate_test_description(locals(), 'aggregated_attractors')
        assert node_correlations is None, test_description


def test_find_node_correlations_B():
    """
    `find_node_correlations`
    Feature B: calculating node correlations if sufficient attractors found.
    """
    # Test for {two attractors found, more than two attractors found}.
    aggregated_attractors_1 = {2: construct_aggregated_attractor(attractor_states_1, 1, 1, 0),
                               3: construct_aggregated_attractor(attractor_states_2, 2, 2, 1)}
    aggregated_attractors_2 = {1: construct_aggregated_attractor(attractor_states_1, 10, 1, 0),
                               8: construct_aggregated_attractor(attractor_states_2, 2, 2, 1),
                               18: construct_aggregated_attractor(attractor_states_3, 20, 5, 4)}

    expected_Rhos = [
        np.array([[1, -1, np.nan, -1],
                  [-1, 1, np.nan, 1],
                  [np.nan, np.nan, np.nan, np.nan],
                  [-1, 1, np.nan, 1]]),
        np.array([[1, -0.77777778, np.nan, -0.49236596],
                  [-0.77777778, 1, np.nan, -0.16412199],
                  [np.nan, np.nan, np.nan, np.nan],
                  [-0.49236596, -0.16412199, np.nan, 1]])]
    expected_Ps = [
        np.array([[0, 0, np.nan, 0],
                  [0, 0, np.nan, 0],
                  [np.nan, np.nan, np.nan, np.nan],
                  [0, 0, np.nan, 0]]),
        np.array([[0, 1.62331867e-07, np.nan, 4.20171535e-03],
                  [1.62331867e-07, 0, np.nan, 3.69407243e-01],
                  [np.nan, np.nan, np.nan, np.nan],
                  [4.20171535e-03, 3.69407243e-01, np.nan, 0]])]
    for aggregated_attractors, (expected_Rho, expected_P) in zip(
            [aggregated_attractors_1, aggregated_attractors_2], zip(expected_Rhos, expected_Ps)):
        db_conn = ZODB.connection(None)
        init_attractor_db_structure(db_conn)
        for key, aggregated_attractor in aggregated_attractors.items():
            db_conn.root.aggregated_attractors[key] = aggregated_attractor
            db_conn.root.n_aggregated_attractors.change(1)
            db_conn.root.total_frequency.change(aggregated_attractor.frequency)

        Rho, P = find_node_correlations(db_conn, None, True)

        test_description = generate_test_description(locals(), 'aggregated_attractors')
        assert np.allclose(expected_Rho, Rho, equal_nan=True), test_description
        assert np.allclose(expected_P, P, equal_nan=True), test_description


def test_assign_frequency_ranks_A():
    """
    `assign_frequency_ranks`
    Feature A: properly calculating "averaged" ranks.
    """
    # Test for {no repeated values, repeated values}.
    data_1 = np.array([.67, .25, .75, .2, .6])
    data_2 = np.array([1, 0, 1, 0, 1])
    # Test for {no frequencies > 1, frequencies > 1}.
    frequencies_1 = np.array([1, 1, 1, 1, 1])
    frequencies_2 = np.array([2, 4, 1, 3, 2])

    expected_ranks_list = [np.array([4, 2, 5, 1, 3]),
                           np.array([10.5, 5.5, 12, 2, 8.5]),
                           np.array([4, 1.5, 4, 1.5, 4]),
                           np.array([10, 4, 10, 4, 10])]

    for (data, frequencies), expected_ranks in zip(
            product([data_1, data_2], [frequencies_1, frequencies_2]), expected_ranks_list):
        ranks = assign_frequency_ranks(data, frequencies)

        test_description = generate_test_description(locals(), 'data', 'frequencies')
        assert np.array_equal(expected_ranks, ranks), test_description

def test_frequency_spearmanrho_A():
    """
    `compute_frequency_spearmanrho`
    Feature A: properly calculating correlations and p-values.
    """
    # Columns are variables and rows are observations, whose frequencies
    # are specified.
    data = np.array([[1, 0, 1, 0, 1],
                     [.67, .25, .75, .2, .6],
                     [.1, .3, .8, .3, .2]]).T
    frequencies = np.array([2, 4, 1, 3, 2])
    # Same data, but with observations (rows) actually repeated instead of
    # their frequencies being specified.
    expanded_data = np.repeat(data.T, frequencies, axis=1).T

    expected_Rho, expected_P = stats.spearmanr(expanded_data)

    Rho, P = compute_frequency_spearmanrho(data, frequencies)

    assert np.allclose(expected_Rho, Rho)
    assert np.allclose(expected_P, P)


def test_frequency_pearsonr_A():
    """
    `compute_frequency_pearsonr`
    Feature A: properly calculating correlations and p-values.
    """
    frequencies = np.array([2, 4, 1, 3, 2])
    # Test for {p-value < 0.05, p-value >= 0.05}.
    raw_data_1 = [[1, 0, 1, 0, 1], [.67, .25, .75, .2, .6]]
    raw_data_2 = [[.67, .25, .75, .2, .6], [.1, .3, .8, .3, .2]]
    for raw_data in [raw_data_1, raw_data_2]:
        # Columns are variables and rows are observations, whose
        # frequencies are specified.
        data = np.array(raw_data).T
        # Same data, but with observations (rows) actually repeated
        # instead of their frequencies being specified.
        expanded_data = np.repeat(data.T, frequencies, axis=1)

        expected_r, expected_p = stats.pearsonr(*expanded_data)
        expected_R = np.array([[1, expected_r], [expected_r, 1]])
        expected_P = np.array([[0, expected_p], [expected_p, 0]])

        R, P = compute_frequency_pearsonr(data, frequencies)

        test_description = generate_test_description(locals(), 'raw_data')
        assert np.allclose(expected_R, R, equal_nan=True), test_description
        assert np.allclose(expected_P, P, equal_nan=True), test_description
