import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import multiprocessing
import re
import time
import numba as nb

nb.njit(cache=True)


def block_core(data, N, N_block_size):
    """
    Blocking method.
    """

    # Gets the size of each block
    block_size = N / N_block_size

    blocked_values = np.empty(N_block_size)
    for j in range(0, N_block_size):
        blocked_values[j] = np.sum(data[j*block_size:(j+1)*block_size])
        blocked_values[j] /= block_size

    X = np.sum(blocked_values)/float(N_block_size)
    XX = np.sum(blocked_values**2)/float(N_block_size)

    return blocked_values, (XX - X*X)/N_block_size, block_size, N_block_size


class Blocking:
    """
    Blocking method av implemented by Flyvebjerg
    """

    def __init__(self, data, N_proc=1):
        """
        Args:
            data: numpy array, datasets.
            N_proc: int, optional, number of threads to run in parallel for.
        """

        # TODO: Include bootstrap analysis inside here as well? Or afterwards?

        N = len(data)

        # Setting up blocks
        N_block_sizes = []  # Number of blocks we divide into
        temp_N = N
        while temp_N > 1e6:
            temp_N /= 10
        N_block_sizes = self.factors(temp_N)[::-1]

        # N_block_sizes = N_block_sizes[:-2]
        N_blocks = len(N_block_sizes)

        # Blocks in parallel
        # pool = multiprocessing.Pool(processes=N_proc)
        # res = pool.map(block_core, N_block_sizes)
        # pool.close()

        self.blocked_values = []
        self.blocked_variances = []
        self.block_sizes = []
        self.N_block_sizes = []

        for N_block_size in N_block_sizes:
            _res = block_core(data, N, N_block_size)
            self.blocked_values.append(_res[0])
            self.blocked_variances.append(_res[1])
            self.block_sizes.append(_res[2])
            self.N_block_sizes.append(_res[3])


        # self.blocked_values = res[:, 0]
        # self.blocked_variances = res[:, 1]
        # self.block_sizes = res[:, 2]
        # self.N_block_sizes = res[:, 3]


    @staticmethod
    def factors(number):
        b = np.arange(1, number+1)
        res, = np.where((number % b) == 0)
        return np.array(res + 1)

    def plot(self):
        """
        Plots the variance vs block sizes.
        """
        plt.semilogx(self.block_sizes[::-1], self.blocked_variances[::-1])
        plt.xlabel("Block size")
        plt.ylabel(r"$\sigma^2$")
        plt.grid(True)
        plt.show()
        plt.close()


def block(x):
    # preliminaries
    d = np.log2(len(x))
    if (d - np.floor(d) != 0):
        print("Warning: Data size = %g, is not a power of 2." % np.floor(2**d))
        print("Truncating data to %g." % 2**np.floor(d))
        x = x[:2**int(np.floor(d))]
    d = int(np.floor(d))
    n = 2**d
    s, gamma = np.zeros(d), np.zeros(d)
    mu = np.mean(x)

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in np.arange(0, d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*np.sum((x[0:(n-1)]-mu)*(x[1:n]-mu))
        # estimate variance of x
        s[i] = np.var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (np.cumsum(((gamma/s)**2*2**np.arange(1, d+1)[::-1])[::-1]))[::-1]

    # we need a list of magic numbers
    q = np. array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
                   16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
                   24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
                   31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
                   38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
                   45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in np.arange(0, d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        print("Warning: Use more data")
    return s[k]/2**(d-k)
