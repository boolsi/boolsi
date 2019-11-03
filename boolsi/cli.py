import os
import sys
import click
import logging
from datetime import datetime
from functools import update_wrapper

from .log import configure_logging
from .constants import Mode, inf
from .input import process_input, InputValidationException
from .simulate import simulate_master, simulate_worker, init_simulation_db_structure
from .attract import attract_master, attract_worker, init_attractor_db_structure
from .attractor_analysis import find_node_correlations
from .target import target_master, target_worker
from .output import output_simulations, output_attractors, output_node_correlations
from .mpi import MPICommWrapper, MPIProcessingException, release_workers_early
from .db import init_db, close_and_cleanup_db, DatabaseInitializationException


timestamp = datetime.now().strftime('%Y%m%dT%H%M%S.%f')[:-3]


@click.group()
@click.pass_context
def cli(ctx):
    """
    BoolSi is an open-source command line tool for distributed simulations of deterministic
    Boolean networks with synchronous update. It uses MPI standard to allow execution on computational
    clusters, as well as parallel processing on a single computer.
    """
    if not ctx or not ctx.obj:
        ctx.obj = {}


_common_opts = [
    click.argument('input_file', type=click.Path(exists=True)),
    click.option('-o', '--output-directory', type=click.Path(file_okay=False, dir_okay=True, exists=False),
                 default='output_' + timestamp,
                 help='Directory to print output to. Defaults to "<current directory>/output_<timestamp>".'),
    click.option('-b', '--batches-per-process', type=click.IntRange(min=1, max=2**31), default=100,
                 help='Number of batches to split the expected simulations of one process into. '
                      'Defaults to 100. Increasing it reduces memory usage and makes the '
                      'distribution of the simulations more even, but decreases the performance.'),
    click.option('-d', '--tmp-database-directory', 'db_dir', type=click.Path(file_okay=False, dir_okay=True),
                 default='tmp_db',
                 help='Directory to store intermediate results in. Defaults to "<current directory>/tmp_db".'),
    click.option('--no-pdf', is_flag=True,
                 help='Disable PDF output. PDF output is enabled by default.'),
    click.option('--pdf-page-limit', type=click.IntRange(min=1), default=500,
                 help='Maximum number of PDF pages to print. Defaults to 500. Only works when PDF output is enabled.'),
    click.option('--no-csv', is_flag=True,
                 help='Disable CSV output. CSV output is enabled by default.'),
    click.option('--print-png', is_flag=True,
                 help='Enable PNG output. PNG output is disabled by default.'),
    click.option('--png-dpi', type=click.IntRange(min=1), default=300,
                 help='PNG dpi. Defaults to 300. Only works when PNG output is enabled.'),
    click.option('--print-tiff', is_flag=True,
                 help='Enable TIFF output. TIFF output is disabled by default.'),
    click.option('--tiff-dpi', type=click.IntRange(min=1), default=150,
                 help='TIFF dpi. Defaults to 150. Only works when TIFF output is enabled.'),
    click.option('--print-svg', is_flag=True,
                 help='Enable SVG output. SVG output is disabled by default.')
]


def add_opts(f):
    """
    Decorate with common CLI arguments and options.

    :param f: function
    :return: function
    """

    # Add common arguments and options.
    for param in reversed(_common_opts):
        f = param(f)

    @click.pass_context
    def f_with_opts(ctx, *args, **kwargs):
        ctx.invoke(f, *args, **kwargs)

    return update_wrapper(f_with_opts, f)


def init(f):
    """
    Initialize MPI, DB, handle errors, etc.

    :param f: function
    :return: function
    """

    @click.pass_context
    def init_f(ctx, *args, **kwargs):

        # Initialize MPI.
        mpi_comm = MPICommWrapper()

        output_directory = kwargs['output_directory']
        db_dir = kwargs['db_dir']

        # Print hello message and run params if master. Silence workers.
        if mpi_comm.rank == 0:
            configure_logging(output_directory)

            logging.getLogger().info('Hi! All BoolSi output (including this log) will appear in "{}".'.format(
                os.path.join(os.path.abspath(output_directory), '')))

            logging.getLogger().info('Run parameters: "{}".'.format(
                ' '.join(sys.argv[1:])))

        else:
            silence_output()

        # Prepare output formats
        image_formats_and_dpis = [format_and_dpi for i, format_and_dpi in
                                  enumerate(
                                      [('svg', None), ('png', kwargs['png_dpi']), ('tiff', kwargs['tiff_dpi'])])
                                  if [kwargs['print_svg'], kwargs['print_png'], kwargs['print_tiff']][i]]

        db_conn = None

        terminating_forcefully = False
        terminating_early = False

        keep_stale_db_items = kwargs.get('keep_stale_db_items', True)

        # Check if outputs are enabled before running.
        if kwargs['no_pdf'] and kwargs['no_csv'] and not image_formats_and_dpis:
            logging.getLogger().warning('Cannot proceed, all output formats are disabled.')
        else:
            try:
                # Get DB connection.
                db_conn = init_db(db_dir, mpi_comm.run_id, mpi_comm.rank, keep_stale_db_items)

                # Call command.
                ctx.invoke(f, mpi_comm, db_conn, image_formats_and_dpis, *args, **kwargs)
            except (DatabaseInitializationException, InputValidationException):
                terminating_early = True
            except MPIProcessingException:
                pass
            except KeyboardInterrupt:
                logging.getLogger().error('Interrupted by user.')

                terminating_forcefully = True
            except Exception as e:
                logging.getLogger().exception('Exception caught: {}. See stacktrace below.'.format(e))

                terminating_early = True

        terminate(output_directory, db_dir, db_conn, mpi_comm, terminating_early, terminating_forcefully)

    return update_wrapper(init_f, f)


def terminate(output_directory, db_dir, db_conn, mpi_comm, terminating_early, terminating_forcefully):
    """
    Gracefully(-ish) terminate BoolSi.

    :param output_directory: local path to BoolSi output directory
    :param db_conn: DB connection (or None if not opened yet)
    :param db_dir: DB directory path
    :param mpi_comm: MPI Communicator
    :param terminating_forcefully: whether to forcefully kill workers (MPI.Abort())
    :return None
    """
    logging.getLogger().info('Terminating...')

    close_and_cleanup_db(db_conn, db_dir)

    logging.getLogger().info('All BoolSi output is located in "{}". Bye!'.format(
        os.path.join(os.path.abspath(output_directory), '')))

    if mpi_comm.n_workers > 0:
        if mpi_comm.rank == 0 and terminating_early:
            release_workers_early(mpi_comm)

        if terminating_forcefully:
            mpi_comm.Abort()

    sys.exit()


@cli.command(help='Simulate for a number of time steps.')
@add_opts
@init
@click.option('-t', '--simulation-time', type=click.IntRange(min=1), required=True,
              help='(required) Number of time steps to simulate for.')
def simulate(mpi_comm, db_conn, image_formats_and_dpis, input_file, output_directory, batches_per_process,
             db_dir, no_pdf, pdf_page_limit, no_csv, print_png, png_dpi,
             print_tiff, tiff_dpi, print_svg, simulation_time):
    init_simulation_db_structure(db_conn)

    if mpi_comm.rank == 0:
        # Current process is master.

        input_cfg = process_input(input_file, output_directory, simulation_time, Mode.SIMULATE)

        # Perform simulations.
        simulate_master(
            mpi_comm, batches_per_process, input_cfg['origin simulation problem'],
            input_cfg['simulation problem variations'], input_cfg['incoming node lists'],
            input_cfg['truth tables'], simulation_time, input_cfg['total combination count'],
            db_conn, output_directory)

        # Print simulations.
        output_simulations(
            db_conn, input_cfg['node names'], output_directory, mpi_comm.n_workers == 0,
            not no_pdf, pdf_page_limit, image_formats_and_dpis, not no_csv)
    else:
        # Current process is worker.
        simulate_worker(mpi_comm, simulation_time, db_conn)


@cli.command(help='Find and analyze attractors for correlations between the nodes.')
@add_opts
@init
@click.option('-t', '--max-simulation-time', type=click.IntRange(min=1),
              help='Maximum simulation time. '
                   'If set, simulation stops after this time step even if attractor was not found.')
@click.option('-a', '--max-attractor-length', type=click.IntRange(min=1),
              help='Maximum length of attractor to look for. If set, attractors longer than this value are discarded.')
@click.option('-r', '--reduce-memory-usage', 'reduce_memory_usage', is_flag=True,
              help='Turn off storing all simulation states. '
                   'Slows down the search for attractors.')
@click.option('-k', '--keep-stale-db-items', 'keep_stale_db_items', is_flag=True,
              help='Turn off cleaning stale attractors from database on each write. Greatly increases'
                   ' disk space usage, speeds processing up, and makes ETA more accurate.')
@click.option('-c', '--no-node-correlations', 'no_node_correlations', is_flag=True,
              help='Turn off computing Spearman\'s correlations between node states in attractors.')
@click.option('-x', '--no-attractor-output', 'no_attractor_output', is_flag=True,
              help='Turn off outputting attractors.')
@click.option('-p', '--p-value', 'p_value',
              type=click.FLOAT, default=0.05,
              help='p-value threshold for statistical significance of node correlations. Defaults to 0.05.')
def attract(mpi_comm, db_conn, image_formats_and_dpis, input_file, output_directory, batches_per_process,
            db_dir, no_pdf, pdf_page_limit, no_csv, print_png, png_dpi,
            print_tiff, tiff_dpi, print_svg, max_simulation_time, max_attractor_length,
            reduce_memory_usage, keep_stale_db_items, no_node_correlations, no_attractor_output,
            p_value):
    max_simulation_time = max_simulation_time or inf
    max_attractor_length = max_attractor_length or inf

    init_attractor_db_structure(db_conn)

    if mpi_comm.rank == 0:
        # Current process is master.

        input_cfg = process_input(input_file, output_directory, max_simulation_time, Mode.ATTRACT)

        if no_node_correlations and no_attractor_output:
            logging.getLogger().info(
                'Cannot proceed, both attractors\' and node correlations\' output is disabled.')
        else:
            # Find aggregated attractors.
            n_aggregated_attractors = attract_master(
                mpi_comm, batches_per_process,
                input_cfg['origin simulation problem'], input_cfg['simulation problem variations'],
                input_cfg['incoming node lists'], input_cfg['truth tables'],
                max_simulation_time, max_attractor_length, input_cfg['total combination count'],
                not reduce_memory_usage, db_conn, not keep_stale_db_items, output_directory)

            if n_aggregated_attractors > 0:

                if no_node_correlations:
                    node_correlations = None
                else:
                    # Find and output node correlations from aggregated attractors.
                    node_correlations = find_node_correlations(db_conn, output_directory, mpi_comm.n_workers == 0)

                if node_correlations:
                    Rho, P = node_correlations
                    output_node_correlations(
                        Rho, P, p_value, input_cfg['node names'], output_directory,
                        not no_pdf, image_formats_and_dpis, not no_csv)

                if not no_attractor_output:
                    # Output attractors.
                    _, fixed_nodes, _ = input_cfg['origin simulation problem']
                    output_attractors(
                        db_conn, fixed_nodes, input_cfg['node names'], input_cfg['total combination count'],
                        max_attractor_length, max_simulation_time, output_directory,
                        mpi_comm.n_workers == 0, not no_pdf, pdf_page_limit,
                        image_formats_and_dpis, not no_csv)
    else:
        # Current process is worker.
        attract_worker(mpi_comm, max_simulation_time, max_attractor_length,
                       not reduce_memory_usage, db_conn, not keep_stale_db_items)


@cli.command(help='Find conditions leading to specific states of the network.')
@add_opts
@init
@click.option('-t', '--max-simulation-time', type=click.IntRange(min=1), required=False,
              help='Maximum simulation time. If set, simulation stops after this time step even '
                   'if a target state was not reached.')
@click.option('-n', '--n-simulations-reaching-target', 'n_simulations_reaching_target',
              type=click.IntRange(min=1), required=False,
              help='Stop after this many simulations have reached target state.')
def target(mpi_comm, db_conn, image_formats_and_dpis, input_file, output_directory,
           batches_per_process, db_dir, no_pdf, pdf_page_limit, no_csv, print_png, png_dpi,
           print_tiff, tiff_dpi, print_svg, max_simulation_time, n_simulations_reaching_target):
    max_simulation_time = max_simulation_time or inf
    n_simulations_reaching_target = n_simulations_reaching_target or inf

    init_simulation_db_structure(db_conn)

    if mpi_comm.rank == 0:
        # Current process is master.

        input_cfg = process_input(input_file, output_directory, max_simulation_time, Mode.TARGET)

        # Find simulations.
        n_simulations = target_master(
            mpi_comm, batches_per_process, input_cfg['origin simulation problem'],
            input_cfg['simulation problem variations'], input_cfg['target substate code'],
            input_cfg['target node set'], input_cfg['incoming node lists'],
            input_cfg['truth tables'], n_simulations_reaching_target, max_simulation_time,
            input_cfg['total combination count'], db_conn, output_directory)

        if n_simulations > 0:
            # Print simulations.
            output_simulations(
                db_conn, input_cfg['node names'], output_directory, mpi_comm.n_workers == 0,
                not no_pdf, pdf_page_limit, image_formats_and_dpis, not no_csv)
    else:
        # Current process is worker.
        target_worker(mpi_comm, max_simulation_time, n_simulations_reaching_target, db_conn)


def silence_output():
    """
    Direct all output to dev/null. Useful if we want workers to shut up.
    """
    try:
        devnull = open(os.devnull, 'w')

        sys.stdout = devnull
        sys.stderr = devnull
    except OSError:
        pass


if __name__ == '__main__':
    cli(obj={})
