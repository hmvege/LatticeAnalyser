import time
import os
import sys
import numpy as np
import numba as nb
from timing_function import timing_function
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

__all__ = ["BootstrapTimeSeries"]


class BootstrapTimeSeries:
    """
    Class for creating a bootstrap sample.
    """
    # @timing_function
    def __init__(self, data, N_bs, tau, index_lists=[], seed=None, axis=None):
        """
        Bootstrapping class. Creates N bootstrap samples for a given dataset.

        Args:
                data: numpy array, datasets.
                N_bs: int, number of bootstrap-samples.
                tau: the first point at whic the autocorrelation becomes negative.
                bootstrap_statistics: optional, function of statistics to run on 
                        bootstrap samples, default is numpy.mean().
                index_lists: optional, numpy array, randomly generated lists
                        that can be provided.
                seed: optional, int/float, seed for bootstrap sampling.

        Returns:
                Object containing bootstrapped values
        """
        N = len(data)

        tau = int(tau)  # Assures we are dealing with an integer

        if seed != None:  # Generates a seed if it is provided
            np.random.seed(seed=seed)
            self.seed = seed

        k = int(np.ceil(N / float(tau)))  # Number of blocks

        # Allows user to send in a predefined list if needed
        if len(index_lists) == 0:
            index_lists = np.random.randint(0, N-tau, size=(N_bs, k))

        self.bs_data = self._boot(data, index_lists, N_bs, tau, N, k)
        # self.bs_data = tsboot(data, N_bs, tau)

        # Performs regular statistics.
        self.bs_avg = np.average(self.bs_data, axis=axis)
        self.bs_var = np.var(self.bs_data, axis=axis)
        self.bs_std = np.sqrt(self.bs_var)

        # Sets some global class variables
        self.shape = self.bs_avg.shape
        self.N_bs = N_bs

    @staticmethod
    @nb.njit(cache=True)
    def _boot(data, index_lists, N_bs, tau, N, k):
        """
        Time series bootstrap.

        Args:
            data: numpy doubles array. Contains data. Shape of (N,).
            index_lists: numpy int array. Shape of (N_bs, k). Random integers
                selected between 0 and N-tau.
            N_bs: int, number of bootstrap sampels.
            tau: int, first lag where tau is zero.
            N: number of data points in data.
            k: number of blocks.
        """
        bs_data_raw = np.zeros(N_bs)
        summation_array = np.zeros(tau*k)

        for i_bs in xrange(N_bs):
            for i, j in enumerate(index_lists[i_bs]):
                summation_array[i*tau:(i+1)*tau] = data[j:j+tau]
            bs_data_raw[i_bs] = summation_array[0:N].mean()

        return bs_data_raw

    def __str__(self):
        """
        When object is printed, prints information about the bootstrap performed.
        """
        msg = "\n" + "="*61 + "\n"
        msg += "Bootstrap:     %10.10f " % self.bs_avg
        msg += "%10.10E " % self.bs_std
        msg += "\nN bootstraps   %d" % self.N_bs
        msg += "\n" + "="*61 + "\n"

        return msg


# @timing_function
def tsboot(data, R, l):
    t = np.empty(R)
    n = int(len(data))
    k = int(np.ceil(float(n)/l))
    # time series bootstrap
    for i in range(R):
        # construct bootstrap sample from
        # k chunks of data. The chunksize is l
        _data = np.concatenate([data[j:j+l]
                                for j in np.random.randint(0, n-l, k)])[0:n]
        t[i] = np.mean(_data)

    return t


def main():
    # Data to load and analyse
    test_data_filename = "tests/topc_beta6_2_t10.dat"
    data = np.loadtxt(test_data_filename, skiprows=0)

    from autocorrelation import PropagatedAutocorrelation
    from bootstrap import Bootstrap

    # Bootstrapping
    N_bootstraps = int(10000)

    ac1 = PropagatedAutocorrelation(data)
    test_data_figurename = os.path.join(
        test_data_filename.split(".")[0]
        + "_autocorrelation_before_bootstrap.png")

    # ac1.plot_autocorrelation(r"$Q$ autocorrelation before bootstrap",
    #                         test_data_figurename,
    #                         verbose=True, dryrun=False)

    h = np.where(ac1.R <= 0.0)[0][0]
    h = 90

    print "TIMING:"
    bs = BootstrapTimeSeries(data, N_bootstraps, h)#, timefunc=True)
    # data_tsboot = tsboot(data, N_bootstraps, h, timefunc=True)

    print "STATISTICS:"
    print "Autocorrelatioin: tau_int:", ac1.tau_int_optimal
    print "Chunk size: ", h
    print ""
    print "Original standard deviation(with ac-correction):   ", \
          np.std(data)/np.sqrt(len(data))*np.sqrt(2*ac1.tau_int_optimal)
    # print "Time series bootstrap(tsboot):                     ", np.std(data_tsboot)
    print "Time series bootstrap(BootstrapTimeSeries):        ", bs.bs_std

    bs_orig = Bootstrap(data, N_bootstraps)#, timefunc=True)
    print "Bootstrap (with ac-correction):                    ", \
          bs_orig.bs_std*np.sqrt(2*ac1.tau_int_optimal)
    print "Original bootstrap:"
    print bs_orig
    print "Timeseries bootstrap"
    print bs


if __name__ == '__main__':
    main()
