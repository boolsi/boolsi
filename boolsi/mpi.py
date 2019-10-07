import sys
import transaction
import logging
from datetime import datetime
from functools import partial

try:
    from mpi4py import MPI
except ImportError:
    pass

from .constants import MpiTags
from .batching import generate_simulation_problems, count_simulation_problem_batches, \
    generate_tasks
from .model import adjust_update_rules_for_fixed_nodes, encode_state, \
    simulate_until_attractor_or_target_substate_or_max_t
from .utils import TickingProgressBar


class MPICommWrapper:
    """
    Wrapper around MPICommunicator that also includes Status object (as a property).
    """
    def __init__(self):
        self._run_id = datetime.now().strftime('%Y%m%d%H%M%S%f')

        if 'mpi4py' in sys.modules:
            self.mpi_comm = MPI.COMM_WORLD
            self._mpi_status = MPI.Status()
            if self.mpi_comm.rank == 0:
                self.mpi_comm.bcast(self._run_id, root=0)
            else:
                self._run_id = self.mpi_comm.bcast(None, root=0)
        else:
            self.mpi_comm = None
            self.size = 1
            self.rank = 0

        self._n_workers = self.size - 1

    @property
    def run_id(self):
        return self._run_id

    @property
    def status(self):
        return self._mpi_status

    @status.setter
    def status(self, value):
        self._mpi_status = value

    @property
    def n_workers(self):
        return self._n_workers

    def __getattr__(self, attr):
        return getattr(self.mpi_comm, attr)


class MPIProcessingException(Exception):
    """
    Happens when something goes wrong during simulation/collection stage.
    """
    def __init__(self):
        super(MPIProcessingException, self).__init__()


class EarlyTerminationException(Exception):
    """
    Happens when something goes wrong during before MPI stages.
    """
    def __init__(self):
        super(EarlyTerminationException, self).__init__()


def generic_master(mpi_comm, n_simulation_problems,
                   n_simulation_problem_batches_per_process, solve_simulation_problem,
                   origin_simulation_problem, simulation_problem_variations,
                   predecessor_node_lists, truth_tables, target_node_set, target_substate_code,
                   store_simulation_result, empty_simulation_results, write_simulation_results_to_db,
                   db_conn, storing_all_states, max_t, output_dirpath, n_simulation_results_to_find):
    """
    Logic for master process.

    :param mpi_comm: MPI communicator object
    :param n_simulation_problems: total number of simulation problems
    :param n_simulation_problem_batches_per_process: number of batches
        to split single process load into
    :param solve_simulation_problem: [function] to solve simulation problem
    :param origin_simulation_problem: simulation problem to start variations from
    :param simulation_problem_variations: variations of simulation problems
    :param predecessor_node_lists: list of predecessor node lists
    :param truth_tables: list of dicts (key: tuple of predecessor node states,
        value: resulting node state)
    :param target_node_set: nodes of the substate to reach
    :param target_substate_code: code of the substate to reach
    :param store_simulation_result: [function] to store simulation result
    :param empty_simulation_results: empty data structure for storing simulation results
    :param write_simulation_results_to_db: [function] to write simulation results to DB
    :param db_conn: DB connection
    :param storing_all_states: whether to store all states
    :param max_t: maximum simulation time
    :param output_dirpath: output dir path
    :param n_simulation_results_to_find: stop after this many unique
        results obtained
    :return:
    """
    # Split all simulation problems into batches and wrap each batch as a task.
    n_processes = max(mpi_comm.n_workers, 1)
    n_simulation_problem_batches = count_simulation_problem_batches(
        n_processes, n_simulation_problems, n_simulation_problem_batches_per_process)
    tasks = generate_tasks(
        origin_simulation_problem, simulation_problem_variations, predecessor_node_lists,
        truth_tables, n_processes, n_simulation_problems,
        n_simulation_problem_batches_per_process)

    # Find simulation results from all simulation problems.
    if mpi_comm.n_workers == 0:
        generic_single_process_master(
            n_simulation_problem_batches, tasks, solve_simulation_problem, store_simulation_result,
            empty_simulation_results, write_simulation_results_to_db, db_conn, storing_all_states,
            max_t, target_node_set, target_substate_code, output_dirpath,
            n_simulation_results_to_find)
    else:
        generic_multi_process_master(
            n_simulation_problem_batches, tasks, mpi_comm, write_simulation_results_to_db, db_conn,
            target_node_set, target_substate_code, output_dirpath, n_simulation_results_to_find)


def generic_single_process_master(
        n_tasks, tasks, solve_simulation_problem, store_raw_simulation_result,
        empty_simulation_results, write_simulation_results_to_db, db_conn, storing_all_states, max_t,
        target_node_set, target_substate_code, output_dirpath, n_simulation_results_to_find):
    """
    Logic for master process when there are no workers.

    :param n_tasks: number of tasks to perform
    :param tasks: generator of tasks for the workers
    :param solve_simulation_problem: [function] to solve simulation problems
    :param store_raw_simulation_result: [function] to store unaggregated simulation result
    :param empty_simulation_results: empty mutable data structure for
        storing simulation results
    :param write_simulation_results_to_db: [function] to write
        simulation results to database, guaranteeing their uniqueness
    :param db_conn: database connection, for storing simulation results
    :param output_dirpath: output directory path
    :param storing_all_states: whether to store all states
    :param max_t: maximum simulation time
    :param target_node_set: nodes of the substate to reach
    :param target_substate_code: code of the substate to reach
    :param output_dirpath: output dir path
    :param n_simulation_results_to_find: stop after this many unique
        results obtained
    :return: None
    """
    _solve_simulation_problem = configure_solve_simulation_problem(
        solve_simulation_problem, storing_all_states, max_t,
        target_node_set, target_substate_code)

    n_simulation_results = 0
    n_executed_tasks = 0

    with TickingProgressBar((n_tasks, ), 1, output_dirpath, True,
                            stage_label='Performing simulations...') as progressbar:
        for task in tasks:
            # Compute batch of simulation problems.
            simulation_result_batch_index, simulation_results = execute_task(
                task, _solve_simulation_problem, store_raw_simulation_result, empty_simulation_results,
                n_simulation_results_to_find)

            if simulation_results:
                # Write simulation results to database and count
                # unique simulation results added.
                simulation_result_batch_size = write_simulation_results_to_db(
                    db_conn, simulation_results, simulation_result_batch_index)

                del simulation_results
                transaction.commit()
                n_simulation_results += simulation_result_batch_size

            n_executed_tasks += 1
            progressbar.update((1, ))

            if n_simulation_results == n_simulation_results_to_find:
                break

    if n_simulation_results >= n_simulation_results_to_find:
        logging.getLogger().info(
            "Goal reached at {:.2%} of simulations to perform.".format(n_executed_tasks / n_tasks))


def generic_multi_process_master(
        n_tasks, tasks, mpi_comm, write_simulation_results_to_db, db_conn, target_node_set,
        target_substate_code, output_dirpath, n_simulation_results_to_find):
    """
    Logic for master process when there are workers.

    :param n_tasks: number of tasks to perform
    :param tasks: generator of tasks for the workers
    :param mpi_comm: MPI communicator object
    :param write_simulation_results_to_db: [function] to write
        simulation results to database, guaranteeing their uniqueness
    :param db_conn: database connection, for storing simulation results
    :param target_node_set: nodes of the substate to reach
    :param target_substate_code: code of the substate to reach
    :param output_dirpath: output directory path
    :param n_simulation_results_to_find: stop after this many unique
        results obtained
    :return: None
    """
    n_released_workers = None
    n_simulation_results = 0
    n_simulation_result_batches = 0

    def generic_stage(progressbar, lookup_table):
        nonlocal n_released_workers
        n_released_workers = 0

        with progressbar:
            while n_released_workers < mpi_comm.n_workers:
                receive_and_handle_message(mpi_comm, MPI.ANY_SOURCE, lookup_table)

        # Signal to workers to proceed with next stage.
        mpi_comm.bcast(False, root=0)

    def simulation_stage():
        # Send simulation problems to workers.

        n_executed_tasks = 0

        nonlocal n_simulation_result_batches
        nonlocal n_simulation_results

        def handle_ready(msg, source):
            # Release the worker if found the number of unique
            # simulation results requested.
            if n_simulation_results >= n_simulation_results_to_find:
                mpi_comm.send(None, dest=source, tag=MpiTags.EXIT)
            # Send next batch of simulation problems to the worker otherwise.
            else:
                try:
                    task = next(tasks)
                except StopIteration:
                    mpi_comm.send(None, dest=source, tag=MpiTags.EXIT)
                else:
                    mpi_comm.send(task, dest=source, tag=MpiTags.START)

        def handle_done(msg, source):
            nonlocal n_simulation_results
            nonlocal n_simulation_result_batches
            nonlocal n_executed_tasks

            # MPI message contains number of simulation results yielded
            # from the task.
            n_simulation_results += msg
            if msg > 0:
                n_simulation_result_batches += 1
            n_executed_tasks += 1
            progressbar.update((1, ))

        def handle_exit(msg, source):
            nonlocal n_released_workers
            n_released_workers += 1

        def handle_error(msg, source):
            nonlocal n_released_workers
            n_released_workers += 1

            raise msg

        lookup_table = {
            MpiTags.READY: handle_ready,
            MpiTags.DONE: handle_done,
            MpiTags.EXIT: handle_exit,
            MpiTags.ERROR: handle_error
        }

        mpi_comm.bcast((target_node_set, target_substate_code), root=0)

        progressbar = TickingProgressBar((n_tasks, ), 1, output_dirpath, False,
                                         stage_label='Performing simulations...')

        generic_stage(progressbar, lookup_table)

        if n_simulation_results >= n_simulation_results_to_find:
            logging.getLogger().info(
                "Goal reached at {:.2%} of simulations to perform.".format(n_executed_tasks / n_tasks))

    def collection_stage():
        # Collect simulation results from workers.

        nonlocal n_released_workers

        def handle_ready(msg, source):
            mpi_comm.send(None, dest=source, tag=MpiTags.START)

        def handle_done(msg, source):
            simulation_result_batch_index, simulation_result_batch = msg
            simulation_result_batch_size = len(simulation_result_batch)
            if simulation_result_batch:
                write_simulation_results_to_db(
                    db_conn, simulation_result_batch, simulation_result_batch_index)
                del simulation_result_batch
                transaction.commit()

            progressbar.update((1, simulation_result_batch_size))

        def handle_exit(msg, source):
            nonlocal n_released_workers
            n_released_workers += 1

        def handle_error(msg, source):
            nonlocal n_released_workers
            n_released_workers += 1

            raise msg

        lookup_table = {
            MpiTags.READY: handle_ready,
            MpiTags.DONE: handle_done,
            MpiTags.EXIT: handle_exit,
            MpiTags.ERROR: handle_error
        }

        progressbar = TickingProgressBar(
            (n_simulation_result_batches, n_simulation_results), 1, output_dirpath, False,
            stage_label='Collecting results...')

        generic_stage(progressbar, lookup_table)

    for stage in [simulation_stage, collection_stage]:
        try:
            stage()
        except Exception as e:
            logging.getLogger().exception('Exception caught: {}'.format(e))

            release_workers(mpi_comm, mpi_comm.n_workers - n_released_workers)

            raise MPIProcessingException


def receive_and_handle_message(mpi_comm, mpi_source, lookup_table):
    mpi_msg = mpi_comm.recv(source=mpi_source, tag=MPI.ANY_TAG, status=mpi_comm.status)
    mpi_source = mpi_comm.status.Get_source()
    mpi_tag = mpi_comm.status.Get_tag()

    lookup_table[mpi_tag](mpi_msg, mpi_source)


def configure_solve_simulation_problem(solve_simulation_problem, storing_all_states, max_t,
                                       target_node_set, target_substate_code):
    """
    Partially applies function to solve simulation problems to the
    known parameters.

    :param solve_simulation_problem: [function] function to solve simulation problems
    :param storing_all_states: [bool]
    :param max_t:
    :param target_node_set:
    :param target_substate_code:
    :return: configured function to solve simulation problems
    """
    # Configure function to encode network state and substate, based
    # on the set of nodes of the substate.
    _encode_state = partial(encode_state, target_node_set)
    # Configure function to perform simulations, based on time cap,
    # an approach to storing states, and target substate.
    _simulate_until_attractor_or_target_substate_or_max_t = partial(
        simulate_until_attractor_or_target_substate_or_max_t, storing_all_states, max_t,
        _encode_state, target_substate_code)
    # Configure function to solve simulation problems.
    _solve_simulation_problem = partial(
        solve_simulation_problem, _simulate_until_attractor_or_target_substate_or_max_t)

    return _solve_simulation_problem


def generic_worker(
        mpi_comm, solve_simulation_problem, store_raw_simulation_result, empty_simulation_results,
        read_simulation_result_batches_from_db, write_simulation_results_to_db, db_conn,
        storing_all_states, max_t, n_simulation_results_to_find):
    """
    Describes worker process computing logic in any of the 3 modes.

    :param mpi_comm: MPI communicator object
    :param mpi_status: MPI status object
    :param store_raw_simulation_result: [function] for storing simulation result
    :param empty_simulation_results: empty mutable data structure for
        storing simulation results
    :param read_simulation_result_batches_from_db: [function] to read
        simulation results from database, based on their label
    :param write_simulation_results_to_db: [function] to write
        simulation results to database, guaranteeing their uniqueness
    :param db_conn: database connection, for storing simulation results
    :param n_simulation_results_to_find: stop after this many unique
        results obtained
    :return: None
    """
    exiting = None

    def generic_stage(lookup_table):
        nonlocal exiting
        exiting = False

        while exiting is False:
            # Report availability.
            mpi_comm.send(None, dest=0, tag=MpiTags.READY)
            # Wait to receive a message from master.
            receive_and_handle_message(mpi_comm, 0, lookup_table)

        # Report completion.
        mpi_comm.send(None, dest=0, tag=MpiTags.EXIT)

    def simulation_stage():
        def handle_start(msg, source):
            # Compute batch of simulation problems.
            simulation_result_batch_index, simulation_results = execute_task(
                msg, _solve_simulation_problem, store_raw_simulation_result,
                empty_simulation_results, n_simulation_results_to_find)

            if simulation_results:
                # Write simulation results to database and count
                # unique simulation results added.
                simulation_result_batch_size = write_simulation_results_to_db(
                    db_conn, simulation_results, simulation_result_batch_index)
                del simulation_results
                transaction.commit()
            else:
                simulation_result_batch_size = 0

            # Send # of unique simulation results to master.
            mpi_comm.send(simulation_result_batch_size, dest=0, tag=MpiTags.DONE)

        def handle_exit(msg, source):
            nonlocal exiting

            exiting = True

        lookup_table = {
            MpiTags.START: handle_start,
            MpiTags.EXIT: handle_exit
        }

        config = mpi_comm.bcast(None, root=0)

        if config:
            target_node_set, target_substate_code = config
        else:
            raise EarlyTerminationException

        _solve_simulation_problem = configure_solve_simulation_problem(
            solve_simulation_problem, storing_all_states, max_t, target_node_set,
            target_substate_code)

        generic_stage(lookup_table)

    def collection_stage():
        def handle_start(msg, source):
            nonlocal exiting

            try:
                indexed_batch_simulation_results = next(simulation_result_batches_generator)
            except StopIteration:
                exiting = True
            else:
                mpi_comm.send(indexed_batch_simulation_results, dest=0, tag=MpiTags.DONE)
                # Allow database to unload simulation result batch from
                # memory.
                del indexed_batch_simulation_results

        def handle_exit(msg, source):
            nonlocal exiting

            exiting = True

        if mpi_comm.bcast(None, root=0):
            raise EarlyTerminationException

        simulation_result_batches_generator = read_simulation_result_batches_from_db(db_conn)

        lookup_table = {
            MpiTags.START: handle_start,
            MpiTags.EXIT: handle_exit
        }

        generic_stage(lookup_table)

    try:
        for stage in [simulation_stage, collection_stage]:
            stage()
    except EarlyTerminationException:
        pass
    except Exception as e:
        mpi_comm.send(e, dest=0, tag=MpiTags.ERROR)


def execute_task(task, solve_simulation_problem, store_raw_simulation_result, empty_simulation_results,
                 n_simulation_results_to_find):
    """
    Solve simulation problems from a batch by applying the
    mode-dependent function provided.

    :param task: (batch seed, batch_index, predecessor node lists, truth tables)
    :param solve_simulation_problem: [function] to solve simulation problem
    :param store_raw_simulation_result: [function] for storing simulation result
    :param empty_simulation_results: empty mutable data structure for
        storing simulation results
    :param n_simulation_results_to_find: stop after this many unique
        results obtained
    :return: simulation results from the task, their number
    """
    try:
        simulation_problem_batch_seed, simulation_problem_batch_index, predecessor_node_lists, \
            truth_tables = task
        n_simulation_results = 0
        simulation_results = empty_simulation_results.copy()
        fixed_nodes_adjusted_for = None

        # Solve every simulation problem from the batch.
        for simulation_problem_index, (initial_state, fixed_nodes, perturbed_nodes_by_t) in \
                enumerate(generate_simulation_problems(*simulation_problem_batch_seed)):

            # Adjust update rule to fixed nodes, if needed.
            if fixed_nodes != fixed_nodes_adjusted_for:
                adjusted_predecessor_node_lists, adjusted_truth_tables = \
                    adjust_update_rules_for_fixed_nodes(predecessor_node_lists, truth_tables, fixed_nodes)
                fixed_nodes_adjusted_for = fixed_nodes

            raw_simulation_result = solve_simulation_problem(
                initial_state, perturbed_nodes_by_t, adjusted_predecessor_node_lists,
                adjusted_truth_tables)

            if raw_simulation_result is not None:
                n_simulation_results += store_raw_simulation_result(
                    raw_simulation_result, fixed_nodes, perturbed_nodes_by_t, simulation_results)
                if n_simulation_results >= n_simulation_results_to_find:
                    break

        return simulation_problem_batch_index, simulation_results

    except (KeyboardInterrupt, ValueError, Exception) as e:
        # Rethrow the exception into main worker function.
        raise e


def release_workers(mpi_comm, n_pending_workers):
    """
    Release workers from blocking MPI communication with master in
    both simulation and collection stages.

    :param mpi_comm: MPI communicator object
    :param n_pending_workers: number of worker processes pending
        in current stage
    :return: None
    """
    logging.getLogger().info("Releasing workers...")

    # Release workers in current stage.
    n_released_workers = 0
    while n_released_workers < n_pending_workers:
        mpi_comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi_comm.status)
        mpi_source = mpi_comm.status.Get_source()
        mpi_tag = mpi_comm.status.Get_tag()
        if mpi_tag in {MpiTags.EXIT, MpiTags.ERROR}:
            n_released_workers += 1
        elif mpi_tag == MpiTags.READY:
            mpi_comm.send(None, dest=mpi_source, tag=MpiTags.EXIT)

    # Signal to workers to abort instead of carrying out next stage.
    mpi_comm.bcast(True, root=0)


def release_workers_early(mpi_comm):
    """
    Signal to workers to abort before simulations.

    :param mpi_comm: MPI communicator object
    :return: None
    """
    mpi_comm.bcast(None, root=0)
