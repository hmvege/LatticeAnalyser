import numpy as np
import matplotlib.pyplot as plt
import sys
import os

__all__ = ["Bootstrap"]


class Bootstrap:
    """
    Class for creating a bootstrap sample.
    """

    def __init__(self, data, N_BS, index_lists=[], seed=None, axis=None):
        """
        Bootstrapping class. Creates N bootstrap samples for a given dataset.

        Args:
                data: numpy array, datasets.
                N_BS: int, number of bootstrap-samples.
                bootstrap_statistics: optional, function of statistics to run on 
                        bootstrap samples, default is numpy.mean().
                index_lists: optional, numpy array, randomly generated lists
                        that can be provided.
                seed: optional, int/float, seed for bootstrap sampling.

        Returns:
                Object containing bootstrapped values
        """
        N = len(data)

        if seed != None:  # Generates a seed if it is provided
            np.random.seed(seed=seed)
            self.seed = seed

        # Allows user to send in a predefined list if needed
        if len(index_lists) == 0:
            index_lists = np.random.randint(N, size=(N_BS, N))

        self.bs_data_raw = data[index_lists]
        self.bs_data = np.mean(self.bs_data_raw, axis=1)

        # Performing basic bootstrap statistics
        self.bs_avg = np.average(self.bs_data, axis=axis)
        self.bs_var = np.var(self.bs_data, axis=axis)
        self.bs_std = np.sqrt(self.bs_var)

        # Performing basic statistics on original data
        self.data_original = data
        self.avg_original = np.average(self.data_original, axis=axis)
        self.var_original = np.var(self.data_original, axis=axis)
        self.std_original = np.std(self.data_original, axis=axis)

        # Sets some global class variables
        self.shape = self.bs_avg.shape
        self.N_BS = N_BS

    def __add__(self, other):
        """
        Enables adding of two bootstrapped datasets.
        """
        if type(other) != Bootstrap:
            raise TypeError("%s should be of type Bootstrap." % other)
        new_data = np.concatenate((self.bs_avg, other.bs_avg), axis=0)
        return new_data

    def __call__(self):
        """
        When called, returns the bootstrapped samples
        """
        return self.bs_data

    def __len__(self):
        """
        Length given as number of bootstraps
        """
        return self.shape[0]

    def __str__(self):
        """
        When object is printed, prints information about the bootstrap performed.
        """
        msg = "BOOTSTRAP:"

        msg += "\n" + "="*61

        msg += "\nNon-bootstrap: "
        msg += "%10.10f " % self.avg_original
        msg += "%10.10E " % self.var_original
        msg += "%10.10E" % self.std_original

        msg += "\nBootstrap:     "
        msg += "%10.10f " % self.bs_avg
        msg += "%10.10E " % self.bs_var
        msg += "%10.10E " % self.bs_std

        msg += "\nN bootstraps   %d" % self.N_BS

        return msg


def main():
    # Data to load and analyse
    data = np.loadtxt("tests/plaq_beta6_2_t10.dat", skiprows=8)

    # Histogram bins
    N_bins = 20

    # Bootstrapping
    N_bootstraps = int(500)
    bs = Bootstrap(data, N_bootstraps)
    bs_data = bs()

    print bs


if __name__ == '__main__':
    main()
