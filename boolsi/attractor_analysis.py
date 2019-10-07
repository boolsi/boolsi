import logging
import numpy as np
from scipy import stats

from .utils import TickingProgressBar


def validate_analysis_possible(f):
    """
    Decorator that validates that the amount of information is
    sufficient for attractor analysis.

    :param f: function
    :return: decorated function
    """
    def f_decorated(*args, **kwargs):
        db_conn, *_ = args
        if db_conn.root.n_aggregated_attractors() == 1 or \
                db_conn.root.total_frequency() <= 2:
            logging.getLogger().info('Not enough attractors to infer node correlations.')

            return None
        else:
            return f(*args, **kwargs)

    return f_decorated


@validate_analysis_possible
def find_node_correlations(db_conn, output_directory, is_single_process):
    """
    Calculate Spearman's rho between node activity in aggregated attractors for
    each pair of nodes, viewing aggregated attractors as observations in [0, 1]^n.

    :param db_conn: connection to database with aggregated attractors
    :param output_directory: output directory path
    :param is_single_process: whether BoolSi run has no workers
    :return: [2-D] pairwise node activity correlations, [2-D] their p-values
    """
    try:
        # Represent aggregated attractors as observations in [0, 1]^n; each entry is
        # computed by averaging the states of the corresponding node in the
        # attractor.
        observations = []
        frequencies = []
        with TickingProgressBar((db_conn.root.n_aggregated_attractors(), ), 1,
                                output_directory, is_single_process,
                                stage_label='Reading attractors to compute node correlations...') as progressbar:
            for aggregated_attractor in db_conn.root.aggregated_attractors.values():
                observations.append(np.mean(aggregated_attractor.states, axis=0))
                frequencies.append(aggregated_attractor.frequency)
                # Force database to unload aggregated attractor from memory.
                del aggregated_attractor
                db_conn.cacheMinimize()
                # Report progress.
                progressbar.update((1, ))

        logging.getLogger().info("Computing node correlations...")

        Rho, P = compute_frequency_spearmanrho(np.array(observations), np.array(frequencies))

        return Rho, P

    except MemoryError:
        logging.getLogger().warning('Cannot compute node correlations because attractors '
                                    'don\'t fit in memory.')
        return None


def compute_frequency_spearmanrho(data, frequencies):
    """
    Compute Spearman's rho between variables, given frequencies of
    the observations.

    :param data: [2-D] observations
    :param frequencies: [1-D] observation frequencies
    :return: [2-D] pairwise correlations, [2-D] pairwise p-values
    """
    # Rank the values in each column.
    ranks = np.apply_along_axis(lambda x: assign_frequency_ranks(x, frequencies), 0, data)
    # Compute Pearson's r correlation and p-values on the ranks.
    return compute_frequency_pearsonr(ranks, frequencies)


def assign_frequency_ranks(values, frequencies):
    """
    Assign ranks to 1-D array, given the frequency of each value. Equal
    values get same "averaged" ranks.

    :param values: [1-D] values
    :param frequencies: [1-D] value frequencies
    :return: [1-D] ranks
    """
    sorted_indices = values.argsort()
    # Construct indices to undo the sorting.
    unsorted_indices = np.zeros_like(sorted_indices)
    unsorted_indices[sorted_indices] = np.arange(len(sorted_indices))
    # Find unique values, their first indices in sorted values, and
    # indices to reconstruct the sorted values.
    unique_values, first_indices, inverse_indices = np.unique(
        values[sorted_indices], return_index=True, return_inverse=True)
    # Find total frequency of each unique value.
    unique_frequencies = np.add.reduceat(frequencies[sorted_indices], first_indices)
    # Find cumulative frequency of each unique value, shifted by 1 to
    # the left.
    shifted_unique_cumulative_frequencies = np.concatenate(
        (np.zeros(1, dtype=int), unique_frequencies[:-1])).cumsum()
    # Find "averaged" rank of each unique value.
    unique_ranks = shifted_unique_cumulative_frequencies + 0.5 * (unique_frequencies + 1)
    # Construct the ranks of sorted values.
    sorted_ranks = unique_ranks[inverse_indices]
    # Construct the ranks of unsorted values.
    ranks = sorted_ranks[unsorted_indices]
    return ranks


def compute_frequency_pearsonr(data, frequencies):
    """
    Calculate Pearson's r between columns (variables), given the
    frequencies of the rows (observations).

    :param data: [2-D] data
    :param frequencies: [1-D] frequencies
    :return: [2-D] pairwise correlations, [2-D] pairwise p-values
    """
    df = frequencies.sum() - 2
    Sigma = np.cov(data.T, fweights=frequencies)
    Sigma_diag = Sigma.diagonal()
    Sigma_diag_pairwise_products = np.multiply.outer(Sigma_diag, Sigma_diag)
    # Calculate matrix with pairwise correlations. Variables of a
    # single (repeatedly) observed value have covariance 0 with any
    # variable, so their correlation with any variable is 0/0 = NaN.
    with np.errstate(invalid='ignore'):
        R = Sigma / np.sqrt(Sigma_diag_pairwise_products)
    # Calculate matrix with pairwise t-statistics. Main diagonal gets
    # 1/0 = inf. Pairs with variables of a single (repeatedly)
    # observed value get NaN.
    with np.errstate(divide='ignore'):
        T = R / np.sqrt((1 - R * R) / df)
    # Calculate matrix with pairwise p-values. NaN t-statistics result
    # in NaN p-values.
    with np.errstate(invalid='ignore'):
        P = 2 * stats.t.sf(np.abs(T), df)

    return R, P
