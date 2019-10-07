import os
import sys
import click
import logging
import queue
import numpy as np
import threading
try:
    from math import gcd
except ImportError:
    from fractions import gcd

from time import time
from io import UnsupportedOperation
from click._termui_impl import ProgressBar
from collections import deque


class TickingProgressBar:
    """
    Context-manager for a progressbar with a background thread that updates it every second.
    Updates are performed using update method and are fed into the thread through a queue.
    """

    def __init__(self, n_progress_items_by_type, length, output_directory, is_singleprocess, stage_label=None):
        self.progressbar = create_progressbar(length, output_directory, is_singleprocess, stage_label=stage_label)
        self.progress_queue = queue.Queue()

        # Spawn a thread to update progressbar every second.
        self.progress_thread = threading.Thread(
            target=report_progress,
            args=(self.progressbar, self.progress_queue, n_progress_items_by_type),
            daemon=True)

        self.progress_thread.start()

    def __enter__(self):
        self.progressbar.__enter__()
        return self

    def __exit__(self, *args):
        self.progress_queue.put(None)
        self.progress_thread.join()
        self.progressbar.__exit__(*args)

    def update(self, value):
        """
        Used to feed the progressbar with updates. It pushed them into the queue that is listened to
        by a thread running in background.

        :param value: progress value
        :return: None
        """
        self.progress_queue.put(value)


class FileOutputProgressBar(ProgressBar):
    """
    Progressbar that outputs into a file and overwrites it.
    """
    def __init__(self, iterable, length=None, fill_char='#', empty_char=' ',
                 bar_template='%(bar)s', info_sep='  ', show_eta=True,
                 show_percent=None, show_pos=False, item_show_func=None,
                 label=None, file=None, color=None, width=30):

        # If the file is not a tty we don't need any control symbols.
        if file and not file.isatty():
            click._termui_impl.BEFORE_BAR = ''
            click._termui_impl.AFTER_BAR = ''

        ProgressBar.__init__(self, iterable, length, fill_char, empty_char,
                             bar_template, info_sep, show_eta,
                             show_percent, show_pos, item_show_func,
                             label, file, color, width)

        # This progress bar is never hidden
        self.is_hidden = False

    def update(self, n_steps):
        """

        :param n_steps: number of steps executed
        :return: None
        """
        # Seek to the beginning of the file if that's possible.
        try:
            self.file.seek(0)
        except UnsupportedOperation:
            pass

        ProgressBar.update(self, n_steps)


def create_progressbar(length, output_directory, is_single_process, iterable=None, show_pos=False, show_eta=True, stage_label=None):
    """
    Create progressbar that outputs into a file if output is not a TTY and seeks to the
    beginning of the file to overwrite it with current progress instead of appending. This looks better.

    :param length: length of the progressbar
    :param output_directory: output directory path
    :param is_single_process: whether BoolSi run has no workers
    :param iterable: iterable to iterate over (no need to manually call .update())
    :param show_pos: whether to show position
    :param show_eta: whether to show eta
    :param stage_label: label of the current stage
    :return: progressbar
    """
    if sys.stdout.isatty() and is_single_process:
        # Just don't use "mpiexec -n 1", it's stupid anyway.
        file = sys.stdout

        if stage_label is not None:
            logging.getLogger().info(stage_label)
    elif output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, 'progress.txt')
        file = open(filename, 'w+')

        if stage_label is not None:
            logging.getLogger().info('{} (see progressbar in "{}").'.format(stage_label, filename))
        else:
            logging.getLogger().info('Progressbar is being printed to "{}".'.format(filename))
    else:
        file = None

    bar = FileOutputProgressBar(iterable=iterable, length=length, label=None,
                                bar_template='%(label)s [%(bar)s] %(info)s',
                                file=file, show_pos=show_pos, show_eta=show_eta)

    return bar


def report_progress(progressbar, progress_queue, sparse_n_items_tuple):
    """
    Report progress to progressbar regularly based on fake pace,
    calculated so that it converges to the actual pace.

    :param progressbar: progressbar of unit length
    :param progress_queue: queue to which progress updates are put
    :param sparse_n_items_tuple: [tuple] numbers of items to complete for each
        item type, sorted roughly by importance desc
    :return: None
    """
    item_type_mask = [n_items > 0 for n_items in sparse_n_items_tuple]
    n_items_array = np.array(sparse_n_items_tuple)[item_type_mask]
    n_item_types = sum(item_type_mask)
    n_finished_items_arrays_by_progress_point = deque([], maxlen=n_item_types)
    elapsed_by_progress_point = deque([], maxlen=n_item_types)
    n_finished_items_array = np.zeros_like(n_items_array)
    sparse_n_finished_items_increase_tuple = tuple(0 for _ in sparse_n_items_tuple)
    progress_reported = 0
    eta_computation_time = None
    min_n_item_types_to_solve_for = 1

    start_time = time()
    last_report_time = start_time
    while sparse_n_finished_items_increase_tuple is not None:
        n_finished_items_increase_array = np.array(sparse_n_finished_items_increase_tuple)[item_type_mask]
        # Recalculate ETA (remaining time in seconds) at each progress point,
        # by solving for item times.
        if (n_finished_items_increase_array > 0).any():
            progress_point_time = time()
            elapsed = progress_point_time - start_time
            elapsed_by_progress_point.appendleft(elapsed)
            n_finished_items_array += n_finished_items_increase_array
            n_finished_items_arrays_by_progress_point.appendleft(n_finished_items_array.copy())
            # Each progress point is a linear equation in processing times of item types.
            if len(n_finished_items_arrays_by_progress_point) == n_item_types:
                A = np.array(n_finished_items_arrays_by_progress_point)
                b = np.array(elapsed_by_progress_point)
                # Throw away less important item types and older progress
                # points until system of equations is solvable for
                # processing times of remaining item types.
                n = n_item_types
                while n >= min_n_item_types_to_solve_for and np.linalg.matrix_rank(A[:n, :n]) < n:
                    n -= 1
                if n >= min_n_item_types_to_solve_for:
                    min_n_item_types_to_solve_for = n
                    eta_computation_time = progress_point_time
                    seconds_per_item_by_item_type = np.linalg.solve(A[:n, :n], b[:n])
                    eta_at_eta_computation = \
                        ((n_items_array - n_finished_items_array)[:n] * seconds_per_item_by_item_type).sum()

        elapsed_since_report = time() - last_report_time
        # Report progress not too frequently.
        if elapsed_since_report > .1 and eta_computation_time is not None:
            elapsed_since_eta_computation = time() - eta_computation_time
            eta = eta_at_eta_computation - elapsed_since_eta_computation
            # Only report progress if the processing is not overdue
            # (w.r.t. to ETA).
            if eta > 0:
                # Calculate fake pace and progress based on it, and report
                # the progress.
                pace_to_report = (1 - progress_reported) / eta
                progress_to_report = pace_to_report * elapsed_since_report
                progressbar.update(progress_to_report)
                progress_reported += progress_to_report
                last_report_time = time()

        # Complement reported progress if finished processing.
        if (n_finished_items_array == n_items_array).all():
            progressbar.update(1 - progress_reported)
            progress_reported = 1

        try:
            sparse_n_finished_items_increase_tuple = progress_queue.get(timeout=.01)
        except queue.Empty:
            sparse_n_finished_items_increase_tuple = tuple(0 for _ in sparse_n_items_tuple)


def lcm(n, m):
    """
    Find least common multiple of two integers.

    :param n: first integer
    :param m: second integer
    :return: least common multiple of theirs
    """
    return n * m // gcd(n, m)
