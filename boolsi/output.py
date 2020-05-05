import logging
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Circle, Patch
from matplotlib.lines import Line2D
from matplotlib import colors, colorbar
import seaborn as sns
from contextlib import ExitStack
from os import path, makedirs
from io import BytesIO
from itertools import product
import gc
from functools import partial

from .constants import aggregated_attractor_name, simulation_name
from .utils import create_progressbar


class PlottingFailedException(Exception):
    """
    Happens when failed to produce a plot.
    """
    pass

# 'failed to plot' text template.
failed_to_plot_template = "Longest {} cannot be plotted (image is too large or bad MathText in " \
                           "node names). Printing to CSV {}."
# Points per inch constant.
ppi = 72
# Default font size (in points).
font_size = 10
# Page number font size (in points).
page_number_font_size = font_size - 2
# Parameters to use for plot annotations.
plot_annotation_kwargs = dict(ha='left', va='bottom', linespacing=1.75)
# Padding between axis label and axis ticks (in points).
axis_labelpad_pts = 8
# Length of axis ticks (in points).
axis_ticklength_pts = 0
# Padding between axis and axis ticks (in points).
axis_tickpad_pts = 6
# Size of space from either axis to its label, excluding
# the size of ticklabels themselves (in inches).
labeling_size_without_ticklabels = \
    (axis_ticklength_pts + axis_tickpad_pts + axis_labelpad_pts + font_size) / ppi
# Size of square node state cell (in inches).
cell_size = 0.25
# Size of gap between node state cells (as fraction of cell size).
gap_size_fraction = 1/3
# Size of gap between heatmap cells (in points).
gap_size_pts = gap_size_fraction * cell_size * ppi
# Border width for legend and colorbar (in points).
border_size_pts = 0.5
# Size of square legend key of node state cell (in points).
legend_cell_size_pts = cell_size * (1 - gap_size_fraction) * ppi
# Width of patch representing fixed nodes in the legend (in font-size
# units).
legend_fixed_node_patch_width = cell_size * ppi / font_size
# Radius of perturbation marker (as fraction of node state cell size).
perturbation_marker_radius_fraction = 0.3 * (1 - gap_size_fraction)
# Radius of circle legend key of perturbation marker (in points).
legend_perturbation_marker_diameter_pts = \
    2 * perturbation_marker_radius_fraction * cell_size * ppi
# Horizontal padding between page annotation and legend (in inches).
legend_h_pad = 0.25
# Vertical padding below the legend (in inches). The object below is
# the heatmap (in network states) or the heatmap (in node correlations).
legend_v_pad = 0.15
# Padding between legend items and legend frame (in inches).
legend_frame_pad = 0.01
# Vertical padding below page annotation (in inches). The object below
# is the legend (if stacked) or the heatmap (otherwise).
annotation_v_pad = 0.15
# Vertical padding between PDF page number and heatmap x-labels (in
# inches).
page_number_v_pad = 0.3
# Linewidth for outlining the main diagonal of node correlations.
diagonal_border_linewidth = 2
# Padding between colorbar edges and its extreme labels (in inches).
cbar_label_edge_pad = 0.05
# Vertical padding between colorbar bottom and correlation heatmap top
# (in inches).
cbar_v_pad = 0.15
# Template for PDF page number text.
page_number_template = 'page {}/{}'
# Color for "off" node state cells.
node_state_0_color = "#d1d1d1"
# Color for "on" node state cells.
node_state_1_color = "#353531"
# Color for gaps between node state cells.
gap_color = "#ffffff"
# Color for marking fixed nodes on state heatmap.
fixed_node_color = "#00ced1"
# Color for marking perturbations on state heatmap.
perturbation_color = "#ff4500"
# Color for the node correlations of -1.
negative_rho_color = "#00ced1"
# Color for zero node correlations.
zero_rho_color = "#ffffff"
# Color for the node correlations of 1.
positive_rho_color = "#ff4500"
# Color for nonsignificant node correlations.
nonsignificant_rho_color = "#777777"
# Color for absent node correlations.
na_rho_color = "#CCCCCC"
# Color of the labels on the node correlations of -1.
negative_rho_label_color = "k"
# Color of the labels on zero node correlations.
zero_rho_label_color = "k"
# Color of the labels on the node correlations of 1.
positive_rho_label_color = "w"
# Default color for design elements.
service_color = "#000000"
# Caption for absent node correlations.
na_correlation_caption = "n/a"
# Caption for nonsignificant node correlations.
nonsignificant_correlation_caption = "n/s"
# Decimal places of attractor trajectory length mean and standard deviation to plot.
trajectory_l_precision = 2
# Decimal places of attractor relative frequency to plot.
frequency_precision = 5
# Decimal places of node correlations to plot.
rho_precision = 3


def init_plotting():
    """
    Set Matplotlib parameters, create figure for plotting, and find
    renderer used by the system.

    :return: (figure, renderer)
    """
    rcParams['text.usetex'] = False
    rcParams['xtick.labelsize'] = rcParams['xtick.labelsize'] = rcParams['legend.fontsize'] = \
        rcParams['font.size'] = font_size
    rcParams['xtick.major.size'] = rcParams['ytick.major.size'] = axis_ticklength_pts
    rcParams['xtick.major.pad'] = rcParams['ytick.major.pad'] = axis_tickpad_pts
    rcParams['axes.labelpad'] = 8
    rcParams['legend.numpoints'] = 1

    fig = plt.figure()

    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.print_figure(BytesIO())
        renderer = fig._cachedRenderer

    return fig, renderer


def configure_plotting_functions(
        db_conn, read_bundled_simulation_results_from_db, extract_simulation_result,
        n_simulation_results, simulation_result_name, extract_display_info,
        compile_plot_annotation_text, node_names, output_dirpath, is_single_process,
        pdf_page_limit, image_formats_and_dpis, no_attractor_plot_annotation_text):
    """
    Make partial application to functions plot_annotation_only(),
    plot_states(), and plot_pdf_page_details().

    :param db_conn: DB connection
    :param read_bundled_simulation_results_from_db: [fuction] to read wrapped
        simulation results from DB
    :param extract_simulation_result: [function] to unwrap simulation result
    :param n_simulation_results: number of simulation results in DB
    :param simulation_result_name: description of simulation result for user
    :param extract_display_info: [function] to extract display
        information from simulation result
    :param compile_plot_annotation_text: [function] to compile plot
        annotation for simulation result
    :param node_names: list of node names
    :param output_dirpath: output directory path
    :param is_single_process: [bool] whether BoolSi run has no workers
    :param pdf_page_limit: maximum number of simulation results to
        print to PDF
    :param image_formats_and_dpis: iterable of tuples (format, dpi)
        corresponding to output image formats
    :param no_attractor_plot_annotation_text: annotation text for the number
        of not found attractors
    :return: ([function] to plot annotation with no states, [function] to
        plot states, [function] to plot PDF-specific elements)
    """
    fig, renderer = init_plotting()

    # Find legend dimensions (in inches) for each legend type.
    legend_widths = dict()
    legend_heights = dict()
    for legend_type in product((False, True), repeat=2):
        legend_widths[legend_type], legend_heights[legend_type] = \
            get_states_legend_dimensions(fig, renderer, *legend_type)

    # Configure function to generate simulation results to be plotted.
    n_simulation_results_to_plot = n_simulation_results if \
        image_formats_and_dpis else min(n_simulation_results, pdf_page_limit)
    def simulation_result_generator():
        for simulation_resut_index, wrapped_simulation_result in enumerate(
                read_bundled_simulation_results_from_db(db_conn)):
            if simulation_resut_index == n_simulation_results_to_plot:
                break

            yield extract_simulation_result(wrapped_simulation_result)

    # Compile annotation texts for the simulation results. Determine
    # whether they can be spread out horizontally with the legends
    # to fit within heatmap width, or they should be stacked instead.
    max_annotation_width = get_text_width(fig, renderer, no_attractor_plot_annotation_text) if \
        no_attractor_plot_annotation_text else 0
    max_horizontal_layout_width = max_annotation_width
    max_stacked_layout_width = max_annotation_width
    longest_states = []
    longest_time_labels = []
    longest_time_labels_for_pdf = []
    stage_label = 'Preprocessing {}s for graphic output...'.format(simulation_result_name)

    with create_progressbar(n_simulation_results_to_plot, output_dirpath, is_single_process,
                            iterable=simulation_result_generator(), show_pos=True, show_eta=False,
                            stage_label=stage_label) as progressbar:
        for simulation_result_index, simulation_result in enumerate(progressbar):
            states, fixed_nodes, perturbed_nodes_by_t, time_labels, _ = \
                extract_display_info(simulation_result)
            annotation_text = compile_plot_annotation_text(
                simulation_result, simulation_result_index)
            annotation_width = get_text_width(fig, renderer, annotation_text)

            if annotation_width > max_annotation_width:
                max_annotation_width = annotation_width
            if len(states) > len(longest_states):
                longest_states = states
                longest_time_labels = time_labels
                if simulation_result_index < pdf_page_limit:
                    longest_time_labels_for_pdf = time_labels

            # Determine legend type for the simulation result.
            legend_type = bool(fixed_nodes), bool(perturbed_nodes_by_t)
            # Calculate width (in inches) of annotation and legend
            # when laid out horizontally.
            legend_width = legend_widths[legend_type]
            horizontal_layout_width = annotation_width + legend_h_pad + legend_width
            if horizontal_layout_width > max_horizontal_layout_width:
                max_horizontal_layout_width = horizontal_layout_width
            stacked_layout_width = max(annotation_width, legend_width)
            if stacked_layout_width > max_stacked_layout_width:
                max_stacked_layout_width = stacked_layout_width

    # Calculate heatmap width (in inches).
    plot_width = len(node_names) * cell_size
    # Determine whether the widest annotation and legend
    # horizontal layout fits heatmap width or they should
    # be stacked.
    if max_horizontal_layout_width > plot_width:
        layout_is_stacked = True
        page_width = max(plot_width, max_stacked_layout_width)
    else:
        layout_is_stacked = False
        page_width = plot_width
    # Compensate for tiny undocumented padding produced by
    # bbox_inches='tight' & pad_inches=0 around text and states' plot.
    page_width *= 1.01

    # Check if longest states can be plotted and measure x-axis
    # labeling height.
    try:
        _, plot = plot_states(fig, False, dict(), 0, longest_states, dict(), dict(), node_names,
                              longest_time_labels, '')
    except ValueError:
        # Image failed to plot, no plotting will occur.
        raise PlottingFailedException
    else:
        plt.draw()
        # Calculate height of space from x-axis to bottom edge of its
        # label (in inches).
        xaxis_labeling_height = labeling_size_without_ticklabels + max(
            label.get_window_extent().height / fig.dpi for label in plot.get_xticklabels())

        # Measure y-axis labeling width to ensure equal PDF page
        # width.
        _, plot = plot_states(fig, False, dict(), 0, longest_states, dict(), dict(),
                              node_names, longest_time_labels_for_pdf, '')
        plt.draw()
        # Calculate width of space from y-axis to leftmost edge of its
        # label (in inches).
        yaxis_labeling_width = labeling_size_without_ticklabels + max(
            label.get_window_extent().width / fig.dpi for label in plot.get_yticklabels())

        fig.clear()

        # Configure function for plotting annotation of simulation
        # problems with no attractor found.
        _plot_annotation_only = partial(plot_annotation_only, fig=fig, n_nodes=len(node_names),
                                        annotation_text=no_attractor_plot_annotation_text)
        # Configure function for plotting states.
        _plot_states = partial(
            plot_states, fig, layout_is_stacked, legend_heights, xaxis_labeling_height)
        # Configure plotting PDF page details.
        _plot_page_details = partial(
            plot_page_details, fig, page_width, xaxis_labeling_height, yaxis_labeling_width)

        return _plot_annotation_only, _plot_states, _plot_page_details


def output_simulation_results(
        _plot_states, _plot_page_details, db_conn, read_bundled_simulation_results_from_db,
        extract_simulation_result, n_simulation_results, simulation_result_name, summaries_csv_header,
        extract_display_info, compile_plot_annotation_text, compile_csv_summaries_row, node_names,
        output_dirpath, is_single_process, to_pdf, pdf_page_limit, image_formats_and_dpis, to_csv,
        output_preamble):
    """
    Output simulation results (simulations or attractors) to specified
    formats.

    :param: _plot_states: [function] to plot states
    :param: _plot_page_details: [function] to add PDF page details to
        plotted states
    :param db_conn: DB connection
    :param read_bundled_simulation_results_from_db: [fuction] to read wrapped
        simulation results from DB
    :param extract_simulation_result: [function] to unwrap simulation result
    :param n_simulation_results: number of simulation results to output
    :param simulation_result_name: description of simulation result for
        user
    :param summaries_csv_header: header to print in CSV summary file
    :param extract_display_info: function to extract display
        information from simulation result
    :param compile_plot_annotation_text: function to compile plot
        annotation for simulation result
    :param compile_csv_summaries_row: function to compile CSV summary
        row for simulation result
    :param node_names: list of node names
    :param output_dirpath: output directory path
    :param is_single_process: [bool] whether BoolSi run has no workers
    :param to_pdf: whether to print simulations as a PDF
    :param pdf_page_limit: maximum number of simulation results to
        print to PDF
    :param image_formats_and_dpis: iterable of tuples (format, dpi)
        corresponding to output image formats
    :param to_csv: whether to print simulations as two CSVs
    :param output_preamble: function to output preamble information
    :return: None
    """
    # Configure function to generate simulation results to be output.
    n_simulation_results_to_output = \
        n_simulation_results if to_csv or image_formats_and_dpis else \
            min(n_simulation_results, pdf_page_limit)
    def simulation_result_generator():
        for simulation_resut_index, wrapped_simulation_result in enumerate(
                read_bundled_simulation_results_from_db(db_conn)):
            if simulation_resut_index == n_simulation_results_to_output:
                break

            yield extract_simulation_result(wrapped_simulation_result)

    output_locations = []
    # Ensure that output directory exists.
    makedirs(output_dirpath, exist_ok=True)
    with ExitStack() as stack:
        if image_formats_and_dpis:
            # Find how many digits are in the number of simulations.
            max_simulation_result_index_length = \
                int(np.log10(n_simulation_results_to_output)) + 1
            # Set up directory paths for the images.
            image_dirpaths = []
            for fmt, _ in image_formats_and_dpis:
                image_dirname = "{}s_{}".format(simulation_result_name, fmt.upper())
                output_locations.append('"{}"'.format(path.join(image_dirname, '')))
                dirpath = path.join(output_dirpath, image_dirname)
                makedirs(dirpath, exist_ok=True)
                image_dirpaths.append(dirpath)

        if to_pdf:
            pdf_filename = "{}s.pdf".format(simulation_result_name)
            output_locations.append('"{}"'.format(pdf_filename))
            pdf_file = stack.enter_context(
                PdfPages(path.join(output_dirpath, pdf_filename)))

        if to_csv:
            csv_summaries_filename = "{}_summaries.csv".format(simulation_result_name)
            csv_filename = "{}s.csv".format(simulation_result_name)
            output_locations.extend(
                ['"{}"'.format(path.join(csv_summaries_filename, '')),
                 '"{}"'.format(path.join(csv_filename, ''))])
            csv_summaries_file = stack.enter_context(
                open(path.join(output_dirpath, csv_summaries_filename),
                     'w', newline='', encoding='utf-8'))
            csv_file = stack.enter_context(
                open(path.join(output_dirpath, csv_filename), 'w', newline='', encoding='utf-8'))

            csv_summaries_writer = csv.writer(csv_summaries_file)
            csv_writer = csv.writer(csv_file)
            csv_summaries_writer.writerow(summaries_csv_header)
            csv_writer.writerow(['{}_id'.format(simulation_result_name), 'time'] + node_names)

        # Output preamble, if needed.
        if output_preamble:

            output_preamble(
                image_dirpaths if image_formats_and_dpis else [], pdf_file if to_pdf else None,
                csv_summaries_writer if to_csv else None)

            if to_pdf:
                n_preamble_pages = 1

        elif to_pdf:
            n_preamble_pages = 0

        if to_pdf:
            n_pages = n_simulation_results_to_output + n_preamble_pages

        stage_label = 'Printing {}s to {}...'.format(simulation_result_name, list_texts(output_locations))

        # Print simulation results.
        with create_progressbar(n_simulation_results_to_output, output_dirpath, is_single_process,
                                iterable=simulation_result_generator(), show_pos=True, show_eta=False,
                                stage_label=stage_label) as progressbar:
            for simulation_result_index, simulation_result in enumerate(progressbar):
                states, fixed_nodes, perturbed_nodes_by_t, time_labels, n_perturbations = \
                    extract_display_info(simulation_result)

                if (to_pdf and simulation_result_index < pdf_page_limit) or image_formats_and_dpis:
                    simulation_result_plot_annotation_text = compile_plot_annotation_text(
                        simulation_result, simulation_result_index)
                    fig, _ = _plot_states(states, fixed_nodes, perturbed_nodes_by_t, node_names,
                                          time_labels, simulation_result_plot_annotation_text)

                    if image_formats_and_dpis:
                        img_filename_base = "{0}{1:0{2}}".format(
                            simulation_result_name, simulation_result_index + 1,
                            max_simulation_result_index_length)
                        for (fmt, dpi), dirpath in zip(image_formats_and_dpis, image_dirpaths):
                            filepath = path.join(dirpath, img_filename_base + "." + fmt)
                            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

                    if to_pdf and simulation_result_index < pdf_page_limit:
                        page_number_text = page_number_template.format(
                            simulation_result_index + n_preamble_pages + 1, n_pages)
                        _plot_page_details(page_number_text)
                        pdf_file.savefig(fig, bbox_inches='tight')

                    fig.clear()
                    gc.collect()

                if to_csv:
                    csv_summaries_row = compile_csv_summaries_row(
                        simulation_result, simulation_result_index)
                    simulation_result_id = csv_summaries_row[0]
                    csv_summaries_writer.writerow(csv_summaries_row)
                    csv_rows = write_states(states, fixed_nodes, perturbed_nodes_by_t,
                                            simulation_result_id, time_labels)
                    csv_writer.writerows(csv_rows)

    if image_formats_and_dpis or to_pdf:
        plt.close()


def output_simulations(db_conn, node_names, output_dirpath, is_single_process, to_pdf,
                       pdf_page_limit, image_formats_and_dpis, to_csv):
    """
    Output simulations to specified formats.

    :param db_conn: connection to database with simulations
    :param node_names: list of node names
    :param output_dirpath: output directory path
    :param is_single_process: [bool] whether BoolSi run has no workers
    :param to_pdf: whether to print simulations as a PDF
    :param pdf_page_limit: maximum number of simulation results to
        print to PDF
    :param image_formats_and_dpis: list of tuples (format, dpi)
        corresponding to output image formats
    :param to_csv: whether to print simulations as two CSVs
    :return: None
    """
    from .simulate import read_bundled_simulations_from_db

    # Define function for extracting display information from simulation.
    def extract_display_info(simulation):
        return simulation.states, simulation.fixed_nodes, simulation.perturbed_nodes_by_t, \
               list(range(len(simulation.states))), simulation.n_perturbations

    # Configure functions for plotting simulations, if needed.
    if to_pdf or image_formats_and_dpis:
        # Define function for annotating simulation plot.
        def compile_plot_annotation_text(simulation, simulation_index):
            id = r'{}\ {}'.format(simulation_name, simulation_index + 1)
            id_text = r'$\mathrm{\bf{' + id + '}}$'
            l_text = "of length ${}$".format(len(simulation.states) - 1)
            fixed_nodes_text = 'with ${}$ fixed nodes'.format(len(simulation.fixed_nodes))
            perturbations_text = 'and ${}$ perturbations'.format(simulation.n_perturbations)

            return '\n'.join([id_text, l_text, fixed_nodes_text, perturbations_text])

        try:
            _, _plot_states, _plot_page_details = configure_plotting_functions(
                db_conn, read_bundled_simulations_from_db, lambda x: x[1],
                db_conn.root.n_simulations(), simulation_name, extract_display_info,
                compile_plot_annotation_text, node_names, output_dirpath, is_single_process,
                pdf_page_limit, image_formats_and_dpis, None)
        except PlottingFailedException:
            logging.getLogger().warning(
                failed_to_plot_template.format(simulation_name, 'only' if to_csv else 'instead'))
            to_csv = True
            to_pdf = False
            image_formats_and_dpis = []
            _plot_states = _plot_page_details = None
    else:
        _plot_states = _plot_page_details = compile_plot_annotation_text = None

    # Define function for compiling simulation CSV summary, if needed.
    if to_csv:
        def compile_csv_summaries_row(simulation, simulation_index):
            n_perturbations_by_node = [0] * len(node_names)
            for perturbed_nodes in simulation.perturbed_nodes_by_t.values():
                for node in perturbed_nodes:
                    n_perturbations_by_node[node] += 1

            return ['{}{}'.format(simulation_name, simulation_index + 1),
                    len(simulation.states) - 1] + [int(node in simulation.fixed_nodes)
                                                   for node in range(len(node_names))] + \
                   n_perturbations_by_node

        csv_summaries_header = \
            [simulation_name + '_id', 'length'] + \
            ['{}_is_fixed'.format(node_name) for node_name in node_names] + \
            ['n_{}_perturbations'.format(node_name) for node_name in node_names]

    else:
        compile_csv_summaries_row = csv_summaries_header = None

    output_simulation_results(
        _plot_states, _plot_page_details, db_conn, read_bundled_simulations_from_db, lambda x: x[1],
        db_conn.root.n_simulations(), simulation_name, csv_summaries_header, extract_display_info,
        compile_plot_annotation_text, compile_csv_summaries_row, node_names, output_dirpath,
        is_single_process, to_pdf, pdf_page_limit, image_formats_and_dpis, to_csv, None)


def output_attractors(
        db_conn, fixed_nodes, node_names, n_simulation_problems, max_attractor_l, max_t,
        output_dirpath, is_single_process, to_pdf, pdf_page_limit, image_formats_and_dpis, to_csv):
    """
    Output aggregated attractors to specified formats.

    :param db_conn: connection to database with attractors
    :param fixed_nodes: dict (by node) of fixed node states
    :param node_names: list of node names
    :param n_simulation_problems: total number of simulation problems
    :param max_attractor_l: maximum length of attractors searched for
    :param max_t: maximum number of time steps simulated
    :param output_dirpath: output directory path
    :param is_single_process: [bool] whether BoolSi run has no workers
    :param to_pdf: whether to output attractors as a PDF
    :param pdf_page_limit: maximum number of simulation results to
        print to PDF
    :param image_formats_and_dpis: list of tuples (format, dpi)
        corresponding to output image formats
    :param to_csv: whether to output attractors as two CSVs
    :return: None
    """
    from .attract import read_bundled_aggregated_attractors_from_db

    # Define function for extracting display information from
    # aggregated attractor.
    def extract_display_info(aggregated_attractor):
        time_labels = \
            ['t'] + ['t+{}'.format(t) for t in range(1, len(aggregated_attractor.states))]

        return aggregated_attractor.states, fixed_nodes, dict(), time_labels, 0

    to_plot = to_pdf or bool(image_formats_and_dpis)

    # Compile annotation text for simulation problems with no attractor
    # found.
    if db_conn.root.total_frequency() < n_simulation_problems:
        no_attractor_relative_frequency = 1 - db_conn.root.total_frequency() / \
                                          n_simulation_problems
        if to_plot:
            no_attractor_text = r'$\mathrm{\bf{no\ ' + aggregated_attractor_name + '}}$'
            if max_attractor_l:
                no_attractor_text += r' of length $\leq {}$'.format(max_attractor_l)
            max_t_text = 'can be reached' if max_t is None else \
                r'can be detected in $\leq {}$ time steps'.format(max_t)
            no_attractor_relative_frequency_text = \
                r'from $\bf{' + '{:.2f}'.format(100 * no_attractor_relative_frequency) + \
                r'\%}$ initial conditions'
            no_attractor_plot_annotation_text = '\n'.join(
                [no_attractor_text, max_t_text, no_attractor_relative_frequency_text])

    elif to_plot:
        no_attractor_plot_annotation_text = None

    # Define function for annotating aggregated attractor plot,
    # if needed.
    if to_plot:
        def compile_plot_annotation_text(aggregated_attractor, aggregated_attractor_index):
            id = r'{}\ {}'.format(aggregated_attractor_name, aggregated_attractor_index + 1)
            id_text = r'$\mathrm{\bf{' + id + '}}$'
            l_text = 'of length ${}$'.format(len(aggregated_attractor.states))
            if aggregated_attractor.frequency > 1:
                trajectory_l_std_text = '±{:.{}f} (mean±SD)'.format(
                    np.sqrt(aggregated_attractor.trajectory_l_variation_sum /
                            (aggregated_attractor.frequency - 1)), trajectory_l_precision)
            else:
                trajectory_l_std_text = ''
            trajectory_l_text = 'reached in {:.{}f}{} time steps'.format(
                aggregated_attractor.trajectory_l_mean, trajectory_l_precision, trajectory_l_std_text)
            relative_frequency_text = \
                r'from $\bf{' + '{:.{}f}'.format(100 * aggregated_attractor.frequency /
                                                 n_simulation_problems, frequency_precision) + \
                r'\%}$ initial conditions'

            return '\n'.join([id_text, l_text, trajectory_l_text, relative_frequency_text])

        try:
            _plot_annotation_only , _plot_states, _plot_page_details = configure_plotting_functions(
                db_conn, read_bundled_aggregated_attractors_from_db, lambda x: x[1][1],
                db_conn.root.n_aggregated_attractors(), aggregated_attractor_name,
                extract_display_info, compile_plot_annotation_text, node_names, output_dirpath,
                is_single_process, pdf_page_limit, image_formats_and_dpis,
                no_attractor_plot_annotation_text)
        except PlottingFailedException:
            logging.getLogger().warning(failed_to_plot_template.format(
                aggregated_attractor_name, 'only' if to_csv else 'instead'))
            to_csv = True
            to_pdf = False
            image_formats_and_dpis = []
            _plot_annotation_only = _plot_states = _plot_page_details = None

    else:
        _plot_states = _plot_page_details  = compile_plot_annotation_text = None

    # Configure plotting function and compile CSV summary row with
    # percentage of simulation problems where no attractor was found.
    if db_conn.root.total_frequency() < n_simulation_problems:
        def output_no_attractor(image_dirpaths, pdf_file, aggregated_attractor_summaries_writer):
            if to_plot:
                fig = _plot_annotation_only()

                for (fmt, dpi), dirpath in zip(image_formats_and_dpis, image_dirpaths):
                    filepath = path.join(dirpath, '_no_{}.{}'.format(
                        aggregated_attractor_name, fmt))
                    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

                if to_pdf:
                    page_number_text = page_number_template.format(
                        1, min(db_conn.root.n_aggregated_attractors(), pdf_page_limit) + 1)
                    _plot_page_details(page_number_text, xaxis_labeling_is_visible=False)
                    pdf_file.savefig(fig, bbox_inches='tight')

                fig.clear()

            if to_csv:
                no_attractor_csv_summary_row = \
                    ['no_' + aggregated_attractor_name,
                     '' if max_attractor_l is None else '<= {}'.format(max_attractor_l),
                     '' if max_t is None else '<= {}'.format(max_t), '',
                     no_attractor_relative_frequency]
                aggregated_attractor_summaries_writer.writerow(no_attractor_csv_summary_row)
    else:
        output_no_attractor = None

    # Define function for compiling aggregated attractor CSV summary,
    # if needed.
    if to_csv:
        def compile_csv_summaries_row(aggregated_attractor, aggregated_attractor_index):
            if aggregated_attractor.frequency > 1:
                trajectory_l_std = np.sqrt(aggregated_attractor.trajectory_l_variation_sum /
                                           (aggregated_attractor.frequency - 1))
            else:
                trajectory_l_std = np.nan

            return ['{}{}'.format(aggregated_attractor_name, aggregated_attractor_index + 1),
                    len(aggregated_attractor.states), aggregated_attractor.trajectory_l_mean,
                    trajectory_l_std, aggregated_attractor.frequency / n_simulation_problems]

        csv_summaries_header = \
            [aggregated_attractor_name + '_id', 'length', 'trajectory_length_mean',
             'trajectory_length_SD', 'relative_frequency']
    else:
        compile_csv_summaries_row = csv_summaries_header = None

    output_simulation_results(
        _plot_states, _plot_page_details, db_conn, read_bundled_aggregated_attractors_from_db,
        lambda x: x[1][1], db_conn.root.n_aggregated_attractors(), aggregated_attractor_name,
        csv_summaries_header, extract_display_info, compile_plot_annotation_text,
        compile_csv_summaries_row, node_names, output_dirpath, is_single_process, to_pdf,
        pdf_page_limit, image_formats_and_dpis, to_csv, output_no_attractor)


def plot_annotation_only(fig, n_nodes, annotation_text):
    """
     Plot invisible dummy network states and annotate them.

     :param fig: figure to plot to
     :param n_nodes: number of nodes in dummy states
     :param annotation_text: annotation of dummy states
     :return: figure
     """
    dummy_states = [[False] * n_nodes]
    # Set plot dimension for dummy states.
    fig.set_size_inches(n_nodes * cell_size, len(dummy_states) * cell_size)
    # Plot dummy states to be invisible.
    sns.heatmap(
        dummy_states, square=True, xticklabels=[''], yticklabels=[''],
        cmap=[gap_color], cbar=None, linecolor=gap_color)
    # Annotate dummy states.
    plt.text(0, 1, annotation_text, **plot_annotation_kwargs)

    return fig


def plot_states(fig, layout_is_stacked, legend_heights, xaxis_labeling_height, states,
                fixed_nodes, perturbed_nodes_by_t, node_names, time_labels, annotation_text):
    """
    Plot states to a heatmap, annotate it, and mark fixed nodes and
    perturbations on it, adding them to the legend.

    :param fig: figure to plot to
    :param layout_is_stacked: whether to stack legend below annotation
    :param legend_heights: dict with legend heights (in inches) for all
        possible legend types
    :param xaxis_labeling_height: height of space from x-axis to bottom
        edge of its label (in inches)
    :param states: states to plot
    :param fixed_nodes: dict (by node) of fixed node states
    :param perturbed_nodes_by_t: dict (by time step) of dicts (by node) of
        perturbed node states
    :param node_names: list of node names
    :param time_labels: labels of states' time steps
    :param annotation_text: annotation of the states
    :return: (figure, plot)
    """
    # Calculate plot dimensions (in inches).
    plot_width, plot_height = len(node_names) * cell_size, len(states) * cell_size

    # Plot states.
    fig.set_size_inches(plot_width, plot_height)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0)
    plot = sns.heatmap(
        states, square=True, xticklabels=node_names, yticklabels=time_labels,
        cmap=[node_state_0_color, node_state_1_color], cbar=None, vmin=False, vmax=True,
        linecolor=gap_color, linewidths=gap_size_pts)
    plot.xaxis.set_label_text("node")
    plot.yaxis.set_label_text("time", rotation=90)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Plot fixed nodes.
    for fixed_node in fixed_nodes:
        plot.add_patch(
            Rectangle((fixed_node + 0.25 * gap_size_fraction, 0.25 * gap_size_fraction),
                      1 - 0.5 * gap_size_fraction, len(states) - 0.5 * gap_size_fraction,
                      fill=False, edgecolor=fixed_node_color, linewidth=0.5 * gap_size_pts))
        for t in range(len(states) - 1):
            plot.add_patch(
                Rectangle((fixed_node + 0.25 * gap_size_fraction, t + 1 - 0.5 * gap_size_fraction),
                          1 - 0.25 * gap_size_fraction, gap_size_fraction,
                          fill=True, facecolor=fixed_node_color, linestyle='None'))

    # Plot perturbations.
    for t in perturbed_nodes_by_t:
        for node in perturbed_nodes_by_t[t]:
            plot.add_patch(
                Circle((node + 0.5, t + 0.5), perturbation_marker_radius_fraction,
                       fill=True, facecolor=perturbation_color, linestyle='None'))

    # Initialize vertical offset between page annotation and plot top
    # (in inches).
    annotation_v_offset = 0
    # Provided legend heights means it's actual plotting and not
    # a test run for checking label dimensions.
    if legend_heights:
        # Determine legend type.
        legend_type = (bool(fixed_nodes), bool(perturbed_nodes_by_t))
        # Calculate y-coordinate of legend bottom (in axes coordinates).
        legend_bottom_y = 1 + (legend_frame_pad + legend_v_pad) / plot_height
        # Plot another set of x-labels on the top if the plot is
        # long enough, and account for their height when plotting
        # legend and annotation.
        if len(states) > 20:
            plot.tick_params(labeltop=True)
            annotation_v_offset += xaxis_labeling_height
            legend_bottom_y += xaxis_labeling_height / plot_height
        # Set legend position and alignment.
        if layout_is_stacked:
            legend_corner = (0, legend_bottom_y)
            legend_location = "lower left"
            # Account for legend height (with frame) when offsetting annotation.
            annotation_v_offset += \
                legend_heights[legend_type] + legend_frame_pad + legend_v_pad
        else:
            legend_corner = (1, legend_bottom_y)
            legend_location = "lower right"
        # Plot legend.
        plot_states_legend(legend_corner, legend_location, *legend_type)

    # Plot annotation.
    plt.text(0, -(annotation_v_pad + annotation_v_offset) / cell_size, annotation_text,
             **plot_annotation_kwargs)

    return fig, plot


def plot_page_details(fig, page_width, xaxis_labeling_height, yaxis_labeling_width,
                      page_number_text, xaxis_labeling_is_visible=True):
    """
    Enforce equal PDF page width and add page number to the plot.

    :param fig: figure to plot to
    :param page_width: PDF page width to enforce (in inches)
    :param xaxis_labeling_height: height of space from x-axis to bottom
        edge of its label (in inches)
    :param yaxis_labeling_width: width of space from y-axis to leftmost
        edge of its label (in inches)
    :param page_number_text: string with PDF page number
    :param xaxis_labeling_is_visible: whether to account for
        xaxis_labeling_height
    :return: None
    """
    plot_width, plot_height = fig.get_size_inches()

    # Plot invisible horizontal line to ensure equal width of all pages.
    plt.axhline(xmin=-yaxis_labeling_width / plot_width, xmax=page_width / plot_width,
                linestyle='None', clip_on=False)

    # Plot page number.
    page_number_v_offset = page_number_v_pad + \
                           (xaxis_labeling_height if xaxis_labeling_is_visible else 0)
    plt.annotate(
        page_number_text, xy=(0.5 * (page_width - yaxis_labeling_width) / plot_width,
                              -page_number_v_offset / plot_height),
        xycoords='axes fraction', ha='center', va='top', weight='light', size=page_number_font_size)


def write_states(states, fixed_nodes, perturbed_nodes_by_t, states_id, time_labels):
    """
    Write states to a CSV-compatible list of rows, marking fixed nodes
    and perturbations.

    :param states: states to write
    :param fixed_nodes: dict (by node) of fixed node states
    :param perturbed_nodes_by_t: dict (by time step) of dicts (by node) of node states
    :param states_id: identifier of the states
    :param time_labels: labels of states' time steps
    :return: list of rows (each is a list)
    """
    states_id_subrow = [states_id] if states_id else []

    rows = []
    for t, state in enumerate(states):
        time_label = time_labels[t] if time_labels else str(t)
        rows.append(states_id_subrow + [time_label] +
                    format_state(state, t, fixed_nodes, perturbed_nodes_by_t))

    return rows


def output_node_correlations(Rho,
                             P,
                             p_value,
                             node_names,
                             output_dirpath,
                             to_pdf,
                             image_formats_and_dpis,
                             to_csv):

    """
    Output pairwise correlations between node activity in attractors.

    :param Rho: [2-D] pairwise Spearman's rho node correlations
    :param P: [2-D] p-values for the pairwise node correlations
    :param p_value: significance level (cutoff for two-sided p-value)
    :param node_names: list of names of the nodes in the network
    :param output_dirpath: output directory path
    :param to_pdf: whether to print node correlations as a PDF
    :param image_formats_and_dpis: iterable of tuples (format, dpi)
        corresponding to output image formats
    :param to_csv: whether to print node correlations as a CSV
    :return: None
    """
    if to_pdf:
        plot_formats_and_dpis = image_formats_and_dpis + [('pdf', None)]
    else:
        plot_formats_and_dpis = image_formats_and_dpis

    output_locations = []
    # Ensure that output directory exists.
    makedirs(output_dirpath, exist_ok=True)
    with ExitStack() as stack:
        output_locations.extend(['"node_correlations.{}"'.format(fmt)
                                 for fmt, _ in plot_formats_and_dpis])
        if to_csv:
            csv_filename = 'node_correlations.csv'
            output_locations.append('"{}"'.format(csv_filename))
            csv_file = stack.enter_context(
                open(path.join(output_dirpath, csv_filename), 'w', newline='', encoding='utf-8'))
            writer = csv.writer(csv_file)
            writer.writerow(['node_1', 'node_2', 'rho', 'p_value'])

        logging.getLogger().info('Printing node correlations to {}...'.format(
            list_texts(output_locations)))

        na_mask = np.isnan(Rho)
        with np.errstate(invalid='ignore'):
            significant_mask = P < p_value
        nonsignificant_mask = np.logical_and(~na_mask, ~significant_mask)
        rho_caption_mask = np.logical_or(significant_mask, np.tril(nonsignificant_mask))
        triu_nonsignificant_mask = np.triu(nonsignificant_mask)
        if plot_formats_and_dpis:
            fig, renderer = init_plotting()
            # Compile annotation text.
            node_correlations_id_text = r'$\mathrm{\bf{node\ correlations}}$'
            significance_text = r'significant for $\alpha=${}'.format(p_value)
            correlation_method_text = "(two-tailed Spearman's rho)"
            annotation_text = '\n'.join(
                [node_correlations_id_text, significance_text, correlation_method_text])

            Rho_for_plotting = Rho.copy()
            # Assign to nonsignificant and absent correlations the
            # values outside [-1, 1] for special colors.
            Rho_for_plotting[triu_nonsignificant_mask] = 666
            Rho_for_plotting[na_mask] = -666
            # Hide values on main diagonal to use it as visual boundary.
            np.fill_diagonal(Rho_for_plotting, np.nan)
            # Determine if there are nonsignificant and absent
            # correlations.
            nonsignificant_present = nonsignificant_mask.any()
            na_present = na_mask.any()
            # Maximum correlation value length is the length of
            # "-.X", where the length of X is precision.
            max_caption_text_length = max(
                len(na_correlation_caption), len(nonsignificant_correlation_caption),
                rho_precision + 2)
            # Fill captions for correlation cells.
            captions = np.empty_like(Rho, dtype=np.dtype('U{}'.format(max_caption_text_length)))
            format_rho = lambda rho: "{0:.{1}f}".format(rho, rho_precision).replace(
                '.' + '0' * rho_precision, '').replace('0.', '.')
            captions[rho_caption_mask] = np.vectorize(format_rho)(
                Rho_for_plotting[rho_caption_mask])
            captions[triu_nonsignificant_mask] = nonsignificant_correlation_caption
            captions[np.triu(na_mask)] = na_correlation_caption
            # Flip vertically the order of nodes for graphic output.
            Rho_for_plotting = np.flipud(Rho_for_plotting)
            captions = np.flipud(captions)
            flipped_node_names = node_names[::-1]
            flipped_rho_caption_mask = np.flipud(rho_caption_mask)
            # Calculate cell dimensions (in inches) so that each cell has
            # sufficient space for its caption.
            max_caption_width = max(get_text_width(fig, renderer, caption) for
                                    caption in np.ravel(captions))
            cell_width = 1.7 * max_caption_width
            cell_height = 2.3 * font_size / ppi
            # Calculate plot dimensions (in inches).
            plot_width = cell_width * len(node_names)
            plot_height = cell_height * len(node_names)
            # Calculate colorbar height.
            cbar_height = cell_height
            # Init plotting.
            fig.set_size_inches(plot_width, plot_height)
            fig.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0)
            sns.set_style("white")
            # Set colormaps for the plot.
            cmap_list = [(0, negative_rho_color), (0.5, zero_rho_color), (1, positive_rho_color)]
            plot_cmap = colors.LinearSegmentedColormap.from_list('plot_cmap', cmap_list)
            plot_cmap.set_over(nonsignificant_rho_color)
            plot_cmap.set_under(na_rho_color)

            # Plot correlations in regular fontweight.
            sns.heatmap(
                Rho_for_plotting, xticklabels=node_names, yticklabels=flipped_node_names,
                cmap=plot_cmap, cbar=False, mask=~flipped_rho_caption_mask, annot=captions,
                fmt='', vmin=-1, vmax=1)
            # Plot labels for nonsignificant and absent correlations in
            # thin fontweight.
            plot = sns.heatmap(
                Rho_for_plotting, xticklabels=node_names, yticklabels=flipped_node_names,
                cmap=plot_cmap, cbar=False, mask=flipped_rho_caption_mask, annot=captions,
                annot_kws={'weight': 'light', 'alpha': 0.9}, fmt='', vmin=-1, vmax=1)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            # Outline and cross out cells on the main diagonal for it
            # to serve as a visual boundary.
            border_width_fraction = diagonal_border_linewidth / cell_width / ppi
            border_height_fraction = diagonal_border_linewidth / cell_height / ppi
            for i in range(len(node_names)):
                left_x = i + border_width_fraction / 2
                right_x = i + 1 - border_width_fraction / 2
                lower_y = len(node_names) - i - border_height_fraction / 2
                upper_y = len(node_names) - i - 1 + border_height_fraction / 2
                plot.plot([left_x, right_x], [upper_y, lower_y], color=service_color,
                          linewidth=diagonal_border_linewidth)
                plot.add_patch(Rectangle([left_x, upper_y], 1 - border_width_fraction,
                                         1 - border_height_fraction, fill=None,
                                         color=service_color, linewidth=diagonal_border_linewidth))

            # Set baseline annotation vertical offset.
            annotation_v_offset = annotation_v_pad + cbar_v_pad + cbar_height
            # Determine if legend is required and what keys should it have.
            if na_present or nonsignificant_present:
                # Calculate width (in inches) of annotation and legend
                # when laid out horizontally.
                annotation_width = get_text_width(fig, renderer, annotation_text)
                artist_list = []
                if na_present:
                    na_artist = Patch(label="not available from\nfound attractors", color=na_rho_color)
                    artist_list.append(na_artist)
                if nonsignificant_present:
                    nonsignificant_artist = Patch(label="not significant\n(p-value ≥ {})".format(p_value),
                                                  color=nonsignificant_rho_color)
                    artist_list.append(nonsignificant_artist)
                # Calculate y-coordinate of legend bottom (in axes coordinates).
                legend_bottom_y = \
                    1 + (legend_frame_pad + legend_v_pad + cbar_v_pad + cbar_height) / plot_height
                # Plot legend on the right by default.
                legend = plt.legend(
                    handles=artist_list, bbox_to_anchor=(1, legend_bottom_y), loc="lower right",
                    borderpad=legend_frame_pad * ppi, labelspacing=1, borderaxespad=0,
                    markerscale=1, ncol=2, fancybox=False, frameon=True, framealpha=1, shadow=False)
                legend_frame = legend.get_frame()
                legend_frame.set_edgecolor(service_color)
                legend_frame.set_linewidth(border_size_pts)
                legend.draw(renderer)
                legend_width, legend_height = legend.get_window_extent().size / fig.dpi
                # If annotation and legend don't fit when aligned
                # horizontally, plot legend on the left and adjust
                # annotation vertical offset so that it's above the legend.
                if annotation_width + legend_h_pad + legend_width > plot_width:
                    legend.remove()
                    n_columns = 1 if legend_width > plot_width else 2
                    legend = plt.legend(
                        handles=artist_list, bbox_to_anchor=(0, legend_bottom_y), loc="lower left",
                        borderpad=legend_frame_pad * ppi, labelspacing=1, borderaxespad=0,
                        markerscale=1, ncol=n_columns, fancybox=False, frameon=True, framealpha=1,
                        shadow=False)
                    legend_frame = legend.get_frame()
                    legend_frame.set_edgecolor(service_color)
                    legend_frame.set_linewidth(border_size_pts)
                    legend.draw(renderer)
                    annotation_v_offset += \
                        legend.get_window_extent().height / fig.dpi + legend_v_pad

            # Plot annotation.
            plot.annotate(
                annotation_text, xy=(0, 1), xytext=(0, annotation_v_offset * ppi),
                xycoords=('axes fraction', 'axes fraction'), textcoords='offset points',
                **plot_annotation_kwargs)

            # Plot colorbar.
            cbar_axes = fig.add_axes(
                [0, 1 + cbar_v_pad / plot_height, 1, cbar_height / plot_height])
            cbar = colorbar.ColorbarBase(
                cbar_axes, cmap=plot_cmap, orientation='horizontal', ticks=[])
            cbar.outline.set_visible(False)
            cbar_label_offset = cbar_label_edge_pad / plot_width
            cbar.ax.text(cbar_label_offset, .5, '-1', ha='left', va='center',
                         color=negative_rho_label_color)
            cbar.ax.text(.5, .5, '0', ha='center', va='center', color=zero_rho_label_color)
            cbar.ax.text(1 - cbar_label_offset, .5, '1', ha='right', va='center',
                         color=positive_rho_label_color)

            # Output to file(s).
            for fmt, dpi in plot_formats_and_dpis:
                plot_file_path = path.join(output_dirpath, 'node_correlations.' + fmt)
                fig.savefig(plot_file_path, dpi=dpi, bbox_inches='tight')

            plt.close()
        if to_csv:
            # Write significant, nonsignificant, and absent correlations,
            # in this exact order.
            for mask, sorting_key in \
                    [(significant_mask, lambda corr: (1 - abs(corr[1]), corr[0][0], corr[0][1])),
                     (nonsignificant_mask, lambda corr: (corr[2], 1 - abs(corr[1]), corr[0][0],
                                                         corr[0][1])),
                     (na_mask, lambda corr: (corr[0][0], corr[0][1]))]:
                triu_mask = np.triu(mask, 1)
                for (node1, node2), rho, p_value in \
                        sorted(zip(np.argwhere(triu_mask), Rho[triu_mask], P[triu_mask]),
                               key=sorting_key):
                    writer.writerow([node_names[node1], node_names[node2], rho, p_value])


def format_state(state, t, fixed_nodes, perturbations):
    """
    Format network state for output by marking perturbations and fixed nodes.

    :param state: state to format
    :param t: time step at which perturbations and fixed nodes are to be marked
    :param fixed_nodes: dict (by node) of fixed node states
    :param perturbations: dict (by time step) of dicts (by node) of node states
    :return: formatted state with perturbations and fixed nodes marked
    """
    formatted_state = [str(int(node_state)) for node_state in state]

    for node in fixed_nodes:
        formatted_state[node] += '_'

    for node in perturbations.get(t, []):
        formatted_state[node] += "*"

    return formatted_state


def get_text_width(fig, renderer, text):
    """
    Approximate (from above) the width of plotted text.

    :param fig: figure where the text is to be plotted
    :param renderer: renderer used by the system
    :param text: text to be plotted
    :return: width of plotted text (in inches)
    """
    plotted_text = plt.text(0, 0, text)
    plotted_text_width = plotted_text.get_window_extent(renderer).width / fig.dpi
    plotted_text.remove()

    return plotted_text_width


def get_states_legend_dimensions(fig, renderer, fixed_nodes_are_present, perturbations_are_present):
    """
    Calculate legend dimensions.

    :param fig: figure where the legend is to be plotted
    :param renderer: renderer used by the system
    :param fixed_nodes_are_present: whether fixed nodes are present in
        the legend
    :param perturbations_are_present: whether perturbations are present in
        the legend
    :return: legend width (in inches), legend height (in inches)
    """
    legend = plot_states_legend(
        (0, 0), "lower left", fixed_nodes_are_present, perturbations_are_present)
    legend.draw(renderer)
    legend_width, legend_height = legend.get_window_extent(renderer).size / fig.dpi
    legend.remove()

    return legend_width, legend_height


def plot_states_legend(legend_corner, legend_location, fixed_nodes_are_present,
                       perturbations_are_present):
    """
    Plot heatmap legend in the specified location.

    :param legend_corner: coordinates of a corner of legend's span box
        (in axes coordinates)
    :param legend_location: location that specifies the coordinates of
        which corner are provided
    :param fixed_nodes_are_present: whether fixed nodes are present in
        heatmap legend
    :param perturbations_are_present: whether perturbations are present in
        heatmap legend
    :return: plotted legend
    """
    node_state_0_proxy_artist = Line2D(
        [], [], marker="s", markersize=legend_cell_size_pts, linestyle='None',
        markeredgewidth=0, color=node_state_0_color, label="node state 0")
    node_state_1_proxy_artist = Line2D(
        [], [], marker="s", markersize=legend_cell_size_pts, linestyle='None',
        markeredgewidth=0, color=node_state_1_color, label="node state 1")
    artist_list = [node_state_0_proxy_artist, node_state_1_proxy_artist]

    if fixed_nodes_are_present:
        fixed_node_artist = Patch(color=fixed_node_color, label="fixed node", linestyle='None')
        artist_list.append(fixed_node_artist)

    if perturbations_are_present:
        perturbation_artist = Line2D(
            [], [], marker="o", markersize=legend_perturbation_marker_diameter_pts,
            linestyle='None', markeredgewidth=0, color=perturbation_color, label="perturbation")
        artist_list.append(perturbation_artist)
        
    # Fixed nodes and perturbations go to another column.
    n_columns = 2 if len(artist_list) > 2 else 1
    # Plot the legend.
    legend = plt.legend(
        handles=artist_list, bbox_to_anchor=legend_corner, loc=legend_location,
        borderpad=legend_frame_pad * ppi, labelspacing=1, borderaxespad=0, markerscale=1,
        ncol=n_columns, fancybox=False, frameon=True, framealpha=1, shadow=False,
        handlelength=legend_fixed_node_patch_width, handleheight=0.7)
    legend_frame = legend.get_frame()
    legend_frame.set_edgecolor(service_color)
    legend_frame.set_linewidth(border_size_pts)

    return legend


def list_texts(texts):
    """
    Compile (Oxford) comma-delimited list of texts.

    :param texts: [list] non-empty sequence of texts
    :return: compiled text
    """
    all_texts_but_last = texts[:-1]
    if all_texts_but_last:
        last_delimiter = (',' if len(all_texts_but_last) > 1 else '') + ' and '

        return last_delimiter.join([', '.join(all_texts_but_last), texts[-1]])
    else:

        return texts[-1]


def format_elapsed_time(seconds):
    """
    Format time difference as "#d ##:##:##".

    :param seconds: [float] time difference in seconds
    :return: [str] formatted time difference
    """
    n_seconds = round(seconds)
    n_days, n_seconds = divmod(n_seconds, 3600 * 24)
    n_hours, n_seconds = divmod(n_seconds, 3600)
    n_minutes, n_seconds = divmod(n_seconds, 60)

    return "{}d {:02}:{:02}:{:02}".format(n_days, n_hours, n_minutes, n_seconds)

