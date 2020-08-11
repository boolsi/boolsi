import os
import logging
import transaction

from ZODB import DB


class DatabaseInitializationException(Exception):
    """
    Happens when something goes wrong during db/connection creation.
    """
    pass


def init_db(db_dir, run_id, mpi_rank, unlink_db):
    """
    Initialize database and clean up its directory.

    :param db_dir: temporary storage directory
    :param run_id: BoolSi run identifier
    :param mpi_rank: MPI rank
    :param unlink_db: whether to unlink own DB files
    :return: DB connection
    """
    try:
        process_id = "{}_{}".format(run_id, mpi_rank)
        db_filename = 'simulation_results_{}'.format(process_id)
        db_filepath = os.path.join(db_dir, db_filename)

        logging.getLogger().info('Initializing temporary storage in "{}/".'.format(os.path.abspath(db_dir)))

        # Ensure temporary storage directory exists.
        os.makedirs(db_dir, exist_ok=True)

        # Create and open the database.
        db = DB(db_filepath, cache_size=-1)
        db_conn = db.open_then_close_db_when_connection_closes()

        # If possible, unlink own database files to ensure they get
        # deleted in the case of abrupt termination (POSIX only).
        if unlink_db:
            for filename in os.listdir(db_dir):
                filename_base, _ = os.path.splitext(filename)
                if filename_base.endswith(process_id):
                    try:
                        os.remove(os.path.join(db_dir, filename))
                    except OSError:
                        pass

        return db_conn

    except Exception as e:
        logging.getLogger().exception('Failed to initialize DB: {}'.format(e))

        raise DatabaseInitializationException


def close_and_cleanup_db(db_conn, db_dir):
    """
    Close connection and do cleanup.

    :param db_conn: DB connection
    :param db_dir: DB directory path
    :return: None
    """
    if db_conn:
        try:
            # Close database.
            db_storage = db_conn.db().storage

            # Abort active transactions if any.
            transaction.abort()

            # Remove database files.
            db_conn.close()
            db_storage.cleanup()
        except Exception as e:
            logging.getLogger().error('Failed to close or cleanup DB: {}'.format(e))


def batch_cache_reset(f):
    """
    Decorator that resets DB cache when starts processing new batch.

    :param f: function
    :return: decorated function
    """
    def wrapper(db_conn, resetting_cache=True):
        previous_simulation_result_index = None

        for simulation_result_index, bundled_simulation_result in f(db_conn):

            if resetting_cache and previous_simulation_result_index is not None and \
                    previous_simulation_result_index != simulation_result_index:
                db_conn.cacheMinimize()

            previous_simulation_result_index = simulation_result_index

            yield simulation_result_index, bundled_simulation_result

        if resetting_cache and previous_simulation_result_index is not None:
            db_conn.cacheMinimize()

    return wrapper
