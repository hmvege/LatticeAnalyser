import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import multiprocessing
import re
import time
import numba as nb


@nb.njit(cache=True)
def block_core(data, block_size):
    """
    Blocking method.
    """

    # Gets the size of each block
    num_blocks = len(data) / block_size

    blocked_values = np.empty(num_blocks)
    for j in range(0, num_blocks):
        blocked_values[j] = np.sum(data[j*block_size:(j+1)*block_size])
        blocked_values[j] /= block_size

    X = np.sum(blocked_values)/float(num_blocks)
    XX = np.sum(blocked_values**2)/float(num_blocks)

    # return blocked_values, (XX - X*X)/num_blocks, block_size, num_blocks
    return blocked_values, np.var(blocked_values)/float(num_blocks)


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
        self.block_sizes = self.factors(N)[::-1][1:-1]

        self.blocked_values = []
        self.blocked_variances = []

        for block_size in self.block_sizes:
            _res = block_core(data, block_size)
            self.blocked_values.append(_res[0])
            self.blocked_variances.append(_res[1])

    @staticmethod
    def factors(number):
        b = np.arange(1, number+1)
        res, = np.where((number % b) == 0)
        return np.array(res + 1)

    def plot(self):
        """
        Plots the variance vs block sizes.
        """
        plt.semilogx(self.block_sizes[::-1], self.blocked_variances[::-1],
                     "o-", color="#225ea8")
        plt.xlabel(r"Block size")
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

    vals = []

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in np.arange(0, d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*np.sum((x[0:(n-1)]-mu)*(x[1:n]-mu))
        # estimate variance of x
        s[i] = np.var(x)
        # perform blocking transformation
        vals.append(x)
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (np.cumsum(((gamma/s)**2*2**np.arange(1, d+1)[::-1])[::-1]))[::-1]

    # we need a list of magic numbers
    q = np.array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
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

    return s[k]/2**(d-k), 2**(d-k), vals[k]


def test():
    test_data = np.fromfile("tests/blocking_test_data")

    print "Test data: %g +/- %g" % (np.mean(test_data), np.std(test_data))
    print "Autoblocking:", np.sqrt(block(test_data)[0])

    b = Blocking(test_data)
    # b.plot()
    print b.block_sizes


if __name__ == '__main__':
    test()
