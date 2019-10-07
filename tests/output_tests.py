import numpy as np
import os
import shutil
from itertools import product
from math import inf
import ZODB
import pytest

from boolsi.output import output_attractors, output_simulations, output_node_correlations
from boolsi.testing_tools import construct_aggregated_attractor, generate_test_description
from boolsi.attract import init_attractor_db_structure, write_aggregated_attractors_to_db
from boolsi.simulate import Simulation, init_simulation_db_structure, write_simulations_to_db


def test_output_simulations_A():
    """
    `output_simulations`
    Feature A: successful output.
    """
    output_dirpath = "Test output of simulations"
    pdf_page_limit = 1
    is_single_process = True
    # Test for {horizontal layout, stacked layout}.
    n_nodes_1 = 2
    n_nodes_2 = 40
    # Test for {no fixed nodes, fixed nodes}.
    fixed_nodes_1 = dict()
    fixed_nodes_2 = {0: False}
    # Test for {no perturbations, perturbations}.
    perturbed_nodes_by_t_1 = dict()
    perturbed_nodes_by_t_2 = {1: {0: True}, 2: {0: False}}
    # Test for {single simulation batch, multiple simulation batches}.
    simulation_cutoff_index_for_batches_1 = 1
    simulation_cutoff_index_for_batches_2 = 2
    # Test for {no PDF format, PDF format}.
    to_pdf_1 = False
    to_pdf_2 = True
    # Test for {no image formats, all image formats}.
    image_formats_and_dpis_1 = []
    image_formats_and_dpis_2 = [('svg', None), ('png', 300), ('tiff', 150)]
    # Test for {no CSV format, CSV format}.
    to_csv_1 = False
    to_csv_2 = True

    for n_nodes, fixed_nodes, perturbed_nodes_by_t, simulation_cutoff_index_for_batches in product(
            [n_nodes_1, n_nodes_2], [fixed_nodes_1, fixed_nodes_2],
            [perturbed_nodes_by_t_1, perturbed_nodes_by_t_2],
            [simulation_cutoff_index_for_batches_1, simulation_cutoff_index_for_batches_2]):

        node_names = [('${}$' if i % 2 else '{}').format('node{}'.format(i))
                      for i in range(n_nodes)]
        # Fill in simulation database.
        db_conn = ZODB.connection(None)
        init_simulation_db_structure(db_conn)
        simulation_1 = Simulation([[False] * n_nodes], fixed_nodes, perturbed_nodes_by_t)
        simulation_2 = Simulation([[True] * n_nodes] * 2, fixed_nodes, perturbed_nodes_by_t)
        simulations = [simulation_1, simulation_2]
        simulation_batch_1 = simulations[:simulation_cutoff_index_for_batches]
        simulation_batch_2 = simulations[simulation_cutoff_index_for_batches:]

        for simulation_batch_index, simulation_batch in enumerate(
                [simulation_batch_1, simulation_batch_2]):
            write_simulations_to_db(db_conn, simulation_batch, simulation_batch_index)

        for to_pdf, image_formats_and_dpis, to_csv in product(
                [to_pdf_1, to_pdf_2], [image_formats_and_dpis_1, image_formats_and_dpis_2],
                [to_csv_1, to_csv_2]):

            if not to_pdf and not image_formats_and_dpis and not to_csv:
                continue

            os.makedirs(output_dirpath, exist_ok=True)
            test_description = generate_test_description(
                locals(), 'n_nodes', 'fixed_nodes', 'perturbed_nodes_by_t',
                'simulation_cutoff_index_for_batches', 'to_pdf', 'image_formats_and_dpis', 'to_csv')
            try:
                output_simulations(db_conn, node_names, output_dirpath, is_single_process, to_pdf,
                                   pdf_page_limit, image_formats_and_dpis, to_csv)
            except:
                pytest.fail(test_description)
            finally:
                shutil.rmtree(output_dirpath)


def test_output_attractors_A():
    """
    `output_attractors`
    Feature A: successful output.
    """
    output_dirpath = "Test output of attractors"
    pdf_page_limit = 1
    is_single_process = True
    packing_db = False
    # Test for {horizontal layout, stacked layout}.
    n_nodes_1 = 2
    n_nodes_2 = 40
    # Test for {no fixed nodes, fixed nodes}.
    fixed_nodes_1 = dict()
    fixed_nodes_2 = {0: False}
    # Test for {attractor found for every simulation problem,
    # attractor not found for some simulation problems}.
    n_simulation_problems_1 = 3
    n_simulation_problems_2 = 4
    # Test for {no time cap, time cap}.
    max_t_1 = inf
    max_t_2 = 10
    # Test for {no attractor length cap, attractor length cap}.
    max_attractor_l_1 = inf
    max_attractor_l_2 = 10
    # Test for {single aggregated attractor batch, multiple
    # aggregated attractor batches}.
    aggregated_attractor_batch_indices_1 = [1, 1]
    aggregated_attractor_batch_indices_2 = [1, 2]
    # Test for {no PDF format, PDF format}.
    to_pdf_1 = False
    to_pdf_2 = True
    # Test for {no image formats, all image formats}.
    image_formats_and_dpis_1 = []
    image_formats_and_dpis_2 = [('svg', None), ('png', 300), ('tiff', 150)]
    # Test for {no CSV format, CSV format}.
    to_csv_1 = False
    to_csv_2 = True

    for n_nodes, fixed_nodes, n_simulation_problems, max_t, max_attractor_l, \
        aggregated_attractor_batch_indices in product(
        [n_nodes_1, n_nodes_2], [fixed_nodes_1, fixed_nodes_2],
        [n_simulation_problems_1, n_simulation_problems_2], [max_t_1, max_t_2],
        [max_attractor_l_1, max_attractor_l_2],
        [aggregated_attractor_batch_indices_1, aggregated_attractor_batch_indices_2]):

        if n_simulation_problems > 3 and max_t == inf and max_attractor_l == inf:
            continue

        node_names = [('${}$' if i % 2 else '{}').format('node{}'.format(i))
                      for i in range(n_nodes)]
        # Fill in attractor database.
        db_conn = ZODB.connection(None)
        init_attractor_db_structure(db_conn)
        aggregated_attractor_1 = construct_aggregated_attractor([[False] * n_nodes], 1, 1, 0)
        aggregated_attractor_2 = construct_aggregated_attractor([[True] * n_nodes] * 2, 2, 1.5, .5)

        for i, (aggregated_attractor_batch_index, aggregated_attractor) in enumerate(
                zip(aggregated_attractor_batch_indices,
                    [aggregated_attractor_1, aggregated_attractor_2])):
            aggregated_attractor_key = i + 1
            aggregated_attractors = {aggregated_attractor_key: aggregated_attractor}
            write_aggregated_attractors_to_db(
                packing_db, db_conn, aggregated_attractors, aggregated_attractor_batch_index)

        for to_pdf, image_formats_and_dpis, to_csv in product(
                [to_pdf_1, to_pdf_2], [image_formats_and_dpis_1, image_formats_and_dpis_2],
                [to_csv_1, to_csv_2]):

            if not to_pdf and not image_formats_and_dpis and not to_csv:
                continue

            os.makedirs(output_dirpath, exist_ok=True)
            test_description = generate_test_description(
                locals(), 'n_nodes', 'fixed_nodes', 'n_simulation_problems', 'max_t',
                'max_attractor_l', 'aggregated_attractor_batch_indices', 'to_pdf',
                'image_formats_and_dpis', 'to_csv')
            try:
                output_attractors(
                    db_conn, fixed_nodes, node_names, n_simulation_problems, max_attractor_l, max_t,
                    output_dirpath, is_single_process, to_pdf, pdf_page_limit, image_formats_and_dpis,
                    to_csv)
            except:
                pytest.fail(test_description)
            finally:
                shutil.rmtree(output_dirpath)


def test_output_node_correlations_A():
    """
    `output_node_correlations`
    Feature A: successful output.
    """
    output_dirpath = "Test output of node correlations"
    n_nodes = 10
    node_names = [('${}$' if i % 2 else '{}').format('node{}'.format(i)) for i in range(n_nodes)]
    p_value = 0.05
    # Test for {no single repeated averaged node states, single
    # repeated averaged node states}.
    repeated_state_nodes_1 = []
    repeated_state_nodes_2 = [5]
    # Test for {valid p-values, insufficient frequencies for valid
    # p-values}.
    invalid_p_value_nodes_1 = []
    invalid_p_value_nodes_2 = [7]
    # Test for {no PDF format, PDF format}.
    to_pdf_1 = False
    to_pdf_2 = True
    # Test for {no image formats, all image formats}.
    image_formats_and_dpis_1 = []
    image_formats_and_dpis_2 = [('svg', None), ('png', 300), ('tiff', 150)]
    # Test for {no CSV format, CSV format}.
    to_csv_1 = False
    to_csv_2 = True

    for repeated_state_nodes, invalid_p_value_nodes in product(
            [repeated_state_nodes_1, repeated_state_nodes_2],
            [invalid_p_value_nodes_1, invalid_p_value_nodes_2]):
        Rho = np.random.uniform(low=-1, high=1, size=(n_nodes, n_nodes))
        P = np.random.uniform(low=0, high=1, size=(n_nodes, n_nodes))
        Rho[np.diag_indices_from(Rho)] = 1
        P[np.diag_indices_from(P)] = 0
        Rho = np.tril(Rho) + np.tril(Rho, -1).T
        P = np.tril(P) + np.tril(P, -1).T
        for i in repeated_state_nodes:
            Rho[i, :].fill(np.nan)
            Rho[:, i].fill(np.nan)
        for i in invalid_p_value_nodes:
            P[i, :].fill(np.nan)
            P[:, i].fill(np.nan)
            Rho[i, :].fill(np.nan)
            Rho[:, i].fill(np.nan)

        for to_pdf, image_formats_and_dpis, to_csv in product(
                [to_pdf_1, to_pdf_2], [image_formats_and_dpis_1, image_formats_and_dpis_2],
                [to_csv_1, to_csv_2]):

            if not to_pdf and not image_formats_and_dpis and not to_csv:
                continue

            os.makedirs(output_dirpath, exist_ok=True)
            test_description = generate_test_description(
                locals(), 'repeated_state_nodes', 'invalid_p_value_nodes', 'to_pdf',
                'image_formats_and_dpis', 'to_csv')
            try:
                output_node_correlations(
                    Rho, P, p_value, node_names, output_dirpath, to_pdf, image_formats_and_dpis, to_csv)
            except Exception as e:
                pytest.fail(test_description)
            finally:
                shutil.rmtree(output_dirpath)
