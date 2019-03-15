#!/usr/bin/env python2

from matplotlib import rc, rcParams
import numba as nb
import types
import time
import os
import numpy as np
from timing_function import timing_function

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

rc("text", usetex=True)
rcParams["font.family"] += ["serif"]

__all__ = ["Autocorrelation", "PropagatedAutocorrelation"]


class _AutocorrelationCore(object):
    """
    Core autocorrelation method object.
    """
    def __init__(self, data, function_derivative=lambda x: 1.0,
                 function_parameters={}, method="correlate"):
        """
        Base method for the auto correlation modules.

        Args:
                data: numpy array of dataset to get autocorrelation for, replicum
                        R=1.
                function_derivative: function of the derivative of function to
                        propagate data through.
                function_parameters: python dictionary of function derivative
                        parameters.
                method: optional, string method of performing autocorrelation:
                        "corroeff", "correlate", "manual".

        Returns:
                Object containing the autocorrelation values
        """

        # Retrieves relevant functions for later
        self.function_derivative = function_derivative
        self.function_parameters = function_parameters

        # Autocorrelation variables
        self.data = data
        self.N = len(self.data)
        self.C0 = np.var(self.data)
        self.R = np.zeros(int(self.N/2))
        self.G = np.zeros(int(self.N/2))
        self.R_error = np.zeros(int(self.N/2))
        self.tau_int = 0

        # Gets the autocorrelations
        if method == "corrcoef":
            self._numpy_autocorrelation(self.data, self.data)
        elif method == "correlate":
            self._numpy2_autocorrelation(self.data, self.data)
        elif method == "manual":
            self._autocorrelation(self.data, self.data)
        else:
            raise KeyError("Method {0:s} of autocorrelation not recognized "
                           "among: corrcoef, correlate, manual".format(method))

    def __call__(self):
        """
        Returns:
                (numpy double array) the auto-correlation
        """
        return self.R

    def __len__(self):
        """
        Returns:
                (int) the length of the auto-correlation results.
        """
        return len(self.R)

    def _autocorrelation(self, data_x, data_y):
        """
        Gets the autocorrelation from a dataset.
        Args:
                Data, (numpy array): dataset to find the autocorrelation in
        Returns:
                C(t)  (numpy array): normalized autocorrelation times
        """
        avg_data_x = np.average(data_x)
        avg_data_y = np.average(data_y)
        for t in range(0, int(self.N/2)):
            for i in range(0, self.N - t):
                self.R[t] += (data_x[i] - avg_data_x) * \
                    (data_y[i+t] - avg_data_y)

            self.R[t] /= (self.N - t)
        self.G = self.R * 2
        self.R /= self.C0

    def _numpy_autocorrelation(self, data_x, data_y):
        """
        Numpy method for finding autocorrelation in a dataset. t is the lag.
        Args:
                Data, (numpy array): dataset to find the autocorrelation in
        Returns:
                C(t)  (numpy array): normalized autocorrelation times 
        """
        for t in range(0, int(self.N/2)):
            self.R[t] = np.corrcoef(np.array([data_x[0:self.N - t],
                                              data_y[t:self.N]]))[0, 1]
        self.G = self.R * self.C0 * 2

    def _numpy2_autocorrelation(self, x, y):
        """
        http://stackoverflow.com/q/14297012/190597
        http://en.wikipedia.org/wiki/Autocorrelation#Estimation
        """
        n = len(x)
        # variance = x.var()
        x = x-x.mean()
        y = y-y.mean()
        self.G = np.correlate(x, y, mode="full")[-n:]
        self.G /= np.arange(n, 0, -1)
        self.G = self.G[:int(self.N/2)]
        self.R = self.G/self.G[0]

    def integrated_autocorrelation_time(self, plot_cutoff=False):
        raise NotImplemented("integrated_autocorrelation_time() not "
                             "implemented for base class")

    def integrated_autocorrelation_time_error(self):
        raise NotImplemented("integrated_autocorrelation_time_error() not "
                             "implemented for base class")

    def _probe_and_plot_S_param(self):
        """
        Method for investigating the S param for the automatic windowing 
        procedure.
        """
        for S in np.linspace(1.0, 2.0, 20):
            self.W = self._gW(
                self._automatic_windowing_procedure(S, self.tau_int))
            if np.isnan(self.W):
                continue

            n = len(self.tau_int)/2
            x = np.arange(n)
            y = self.tau_int[:n]
            y_std = self.integrated_autocorrelation_time_error()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, y)
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.5)
            ax.set_title(r"$W = %d$, $S_\mathrm{param} = %.2f$" % (self.W, S))
            ax.set_ylim(0, 1.25*np.max(self.tau_int[:len(self.tau_int)/4]))
            ax.set_xlabel(r"$W$")
            ax.set_ylabel(r"$\tau_\mathrm{int}(W)$")
            ax.axvline(self.W)
            # plt.show()
            plt.close(fig)

    def plot_autocorrelation(self, title, filename, lims=1, dryrun=False,
                             verbose=False):
        """
        Plots the autocorrelation.
        """
        x = range(len(self.R))
        y = self.R
        y_std = self.R_error

        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111)
        ax.plot(x, y, color="0", label="Autocorrelation")
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.5, edgecolor='',
                        facecolor='r')
        ax.set_ylim(-lims, lims)
        ax.set_xlim(0, int(self.N/2))
        ax.set_xlabel(r"Lag $t$")
        ax.set_ylabel(r"$R = \frac{C_t}{C_0}$")
        ax.set_title(title, fontsize=16)
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, 0.2))
        ax.grid(True)
        ax.legend()
        if not dryrun:
            if filename.startswith("tests"):
                fig.savefig(filename)
            else:
                filename = "tests/autocorrelation_%s.pdf" % filename
                fig.savefig(filename)

            if verbose:
                print "Test data saved at: %s" % filename
        plt.close(fig)


class Autocorrelation(_AutocorrelationCore):
    """
    Class for performing an autocorrelation analysis based on Luscher
    """
    @timing_function
    def __init__(self, *args, **kwargs):
        """
        Base method for the auto correlation modules.

        Args:
                data: numpy array of dataset to get autocorrelation for, replicum 
                        R=1.
                function_derivative: function of the derivative of function to 
                        propagate data through.
                function_parameters: python dictionary of function derivative 
                        parameters.
                method: optional, string method of performing autocorrelation: 
                        "corroeff", "correlate", "manual".

        Returns:
                Object containing the autocorrelation values
        """

        # Calls parent
        super(Autocorrelation, self).__init__(*args, **kwargs)

        # Lambda cutoff
        self.LAMBDA = 100  # As in paper

        # Gets the autocorrelation errors
        list(map(self._autocorrelation_error, range(int(self.N/2))))

    def _autocorrelation_error(self, t):
        """
        Function for calculating the autocorrelation error.
        Equation E.11 in Luscher

        Args:
                R: numpy array of autocorrelations.
        Returns:
                R_error: numpy array of error related to the autocorrelation.
        """

        for k in range(1, self.LAMBDA + t):
            if k+t >= int(self.N/2):
                break
            else:
                # abs since gamma(t) = gamma(-t)
                self.R_error[t] += (self.R[k+t] + self.R[np.abs(k-t)] -
                                    2*self.R[k]*self.R[t])**2

        self.R_error[t] = np.sqrt(self.R_error[t] / float(self.N))
        return self.R_error[t]

    def _get_optimal_w(self):
        """Equation E.13 in Luscher(2004)."""
        for t in range(1, int(self.N/2)):
            if np.abs(self.R[t]) <= self.R_error[t]:
                self.W = t
                break
        else:
            self.W = np.nan

    def integrated_autocorrelation_time(self, plot_cutoff=False):
        """
        Finds the integrated autocorrelation time, and returns it in order 
        to correct the standard deviation.

        Returns:
                2*tau_int (float)
        """
        if self.R[-1] == 0 or self.R[0] == 0:
            print("Error: autocorrelation has not been performed yet!")

        self.tau_int = np.array([0.5 + np.sum(self.R[1:iW])
                                 for iW in range(1, int(self.N/2))])

        self._get_optimal_w()

        # Sums the integrated autocorrelation time, eq E.12 Luscher(2004)
        # self.tau_int = 0.5 + np.sum(self.R[1:self.W])

        self.tau_int_optimal = self.tau_int[self.W]

        # Plots cutoff if prompted
        if plot_cutoff:
            tau = np.zeros(len(self.R) - 1)
            for iW in range(1, len(self.R)-1):
                tau[iW] = (0.5 + np.sum(self.R[1:iW]))

            plt.figure()
            plt.plot(range(1, len(self.R)), tau)
            plt.title(r"Cutoff $W = %d$" % (self.W))
            plt.xlabel(r"$W$")
            plt.ylabel(r"$\tau_{int}(W)$")
            plt.axvline(self.W)
            # plt.show()

        return self.tau_int_optimal

    def integrated_autocorrelation_time_error(self):
        """
        Equation E.14 in Luscher(2004)
        """
        self.tau_int_error = np.sqrt(
            (4*self.W + 2)/float(self.N) * self.tau_int**2)
        # self.tau_int_error = np.sqrt(
        #   4/float(self.N) * (self.W + 0.5 - self.tau_int) * self.tau_int**2)
        self.tau_int_optimal_error = self.tau_int_error[self.W]
        return self.tau_int_optimal_error


class PropagatedAutocorrelation(_AutocorrelationCore):
    """
    Class for performing a semi-generall autocorrelation analysis according to
    2006 paper by Wolff.

    Assumptions throughout the program:
    - only have 1 alpha, that is only one observable. This simplifies quite alot.
    """

    @timing_function
    def __init__(self, *args, **kwargs):
        # Calls parent
        super(PropagatedAutocorrelation, self).__init__(*args, **kwargs)

        # Gets the autocorrelation errors
        self._autocorrelation_error()

    def _autocorrelation_error(self, SParam=1.11):
        # Eq. 6, 7
        avg_data = np.mean(self.data)

        # Eq. 14
        derfun_avg = self.function_derivative(avg_data,
                                              **self.function_parameters)

        # Eq. 33
        self.G *= derfun_avg**2

        # Eq. 35, array with different integration cutoffs
        CfW = np.array([self.G[0]] + [self.G[0] + 2.0*np.sum(self.G[1:W+1])
                                      for W in range(1, int(self.N/2))])
        # CfW = np.array([0.5 + np.sum(self.R[1:iW]) for iW in range(1,self.N/2)])

        # Eq. 49, bias correction
        CfW = self._correct_bias(CfW)

        # Eq. 34
        sigma0 = CfW[0]

        # Eq. 41
        self.tau_int = CfW / (2*sigma0)

        if SParam == False:
            self._probe_and_plot_S_param()
        else:
            self.W = self._gW(
                self._automatic_windowing_procedure(SParam, self.tau_int))

        self.tau_int_optimal = self.tau_int[self.W]

    def _correct_bias(self, CfW):
        """
        Eq. 49, bias correction
        """
        return CfW*((2*np.arange(len(CfW)) + 1.0)/float(self.N) + 1.0)

    def _gW(self, tau):
        """
        Eq. 52, getting optimal W
        """
        for iW, itau in enumerate(tau):
            if iW == 0:
                continue
            if np.exp(-iW/itau) - itau/np.sqrt(iW*float(self.N)) < 0.0:
                return iW
        else:
            return np.nan

    @staticmethod
    @nb.njit(cache=True)
    def _automatic_windowing_procedure(S, tau_int):
        """
        Automatic windowing as described in Wolff paper, section 3.3 
        """
        tau = np.zeros(len(tau_int))
        for it, itauint, in enumerate(tau_int):
            if itauint <= 0.5:
                tau[it] = 0.00000001
            else:
                # Eq. 51
                tau[it] = S/np.log((2*itauint + 1) / (2*itauint - 1))

        return tau

    def integrated_autocorrelation_time_error(self):
        """
        Eq. 42, standard deviation of tau_int
        """
        self.tau_int_error = np.asarray(
            [np.sqrt(4/float(self.N)*(float(iW) + 0.5 - itau)*itau**2)
             for iW, itau in enumerate(self.tau_int)])
        # self.tau_int_error = np.sqrt((4*self.W + 2)/float(self.N) * self.tau_int**2)
        self.tau_int_optimal_error = self.tau_int_error[self.W]
        return self.tau_int_optimal_error

    # @staticmethod
    # @nb.njit(cache=True)
    # def integrated_autocorrelation_time_error(N, tau_int, W):
    #     """
    #     Eq. 42, standard deviation of tau_int
    #     """
    #     tau_int_error = np.asarray(
    #         [np.sqrt(4/float(N)*(float(iW) + 0.5 - itau)*itau**2)
    #          for iW, itau in enumerate(tau_int)])
    #     # self.tau_int_error = \
    #     #     np.sqrt((4*self.W + 2)/float(self.N) * self.tau_int**2)
    #     return tau_int_error, tau_int_error[W]

    def integrated_autocorrelation_time(self):
        return self.tau_int_optimal


class FullAutocorrelation(_AutocorrelationCore):
    """
    Class for performing a general autocorrelation analysis according to 2006 
    paper by Wolff.

    Include average of several replicums, many observables given by greek 
    indices, as well as propagated errors.
    """

    @timing_function
    def __init__(self, data, function_derivative=[lambda x: 1.0],
                 function_parameters={}, numerical_derivative=False):
        """
        Base method for the auto correlation modules.

        Args:
                data: numpy array of dataset to get autocorrelation for, replicum 
                        R=1. Structure of input: [replicums][observables][data]
                function_derivative: list of functions of the derivative of 
                        function to propagate data through.
                function_parameters: python dictionary of function derivative 
                        parameters.
                method: optional, string method of performing autocorrelation: 
                        "corroeff", "correlate", "manual".

        Returns:
                Object containing the autocorrelation values
        """
        assert isinstance(data, list), "data is not a list of replicums."
        for d in data:
            assert isinstance(d, np.ndarray), (
                "replicum is not of type numpy ndarray.")

        assert isinstance(function_derivative, list), (
            "Missing list of derivatives of observables.")
        for func in function_derivative:
            assert isinstance(func, types.FunctionType), (
                "Derivative is not of function type: %s" % type(func))

        if numerical_derivative:
            raise NotImplementedError(
                "TODO: implement option for taking a numerical derivative")

        # N replicums and a check
        self.NReplicums = len(data)
        assert self.NReplicums != 0, "No replicums provided."

        # Makes sure data have shape (observable type, time series data)
        for ir in range(len(data)):
            data[ir] = np.atleast_2d(data[ir])

        # Gets the number of observables
        num_obs = set(list(map(len, data)))
        assert len(num_obs) == 1, ("unequal number of observables in "
                                   "replicums: %s" % num_obs)
        self.N_obs = list(num_obs)[0]

        # Retrieves relevant functions for later
        self.function_derivative = function_derivative
        self.function_parameters = function_parameters

        # Autocorrelation variables
        self.data = np.asarray(data)

        # Sets up data set lengths
        self.NR = np.asarray(list(map(np.shape, self.data)))[:, 1]
        self.N = np.sum(self.NR)
        self.R_error = np.zeros(int(self.N/2))

        # Gets the autocorrelations
        self.G_ab_t = np.empty((self.N_obs, self.N_obs, int(self.N//2)))
        for ialpha in range(self.N_obs):
            for ibeta in range(self.N_obs):
                self.G_ab_t[ialpha, ibeta] = \
                    self._autocorrelation_with_replicums(
                        self.data[:, ialpha], self.data[:, ibeta])

        # Gets the autocorrelation errors
        self._autocorrelation_error()

    def _autocorrelation_with_replicums(self, x, y):
        """Eq. 31. Autocorrelation function with replicums."""
        self.G = np.zeros(self.N)
        for ir in range(self.NReplicums):
            # variance = x.var()
            _x = x[ir] - x[ir].mean()
            _y = y[ir] - y[ir].mean()
            self.G += np.correlate(_x, _y, mode="full")[-self.NR[ir]:]

        self.G /= (self.N - self.NReplicums*np.arange(0, self.N, 1))
        self.G = self.G[:int(self.N/2)]
        self.R = self.G/self.G[0]
        return self.G

    # @staticmethod
    # @nb.njit(cache=True)
    # def _get_total_average(avg_replicums, NR, N_obs, N):
    #     # Eq. 6: gets the average of each replicum, avg_replicums

    #     # Eq. 7: gets the total average
    #     avg_data = np.empty(N_obs)
    #     for ia in range(N_obs):
    #         # tmp = np.zeros(avg_replicums.shape[0])
    #         # for _nr in NR:
    #         #     tmp += _nr*np.transpose(avg_replicums)[ia]
    #         # avg_data[ia] = np.mean(tmp)
    #         avg_data[ia] = np.mean(np.multiply(NR, avg_replicums[ia]))
    #     return avg_data

    def _autocorrelation_error(self, SParam=1.11):
        """Computes the autocorrelation with error propagation."""

        # t0 = time.time()
        # self.avg_data = self._get_total_average(np.mean(self.data, axis=2).T,
        #                                         self.NR, self.N_obs, self.N)
        # print("Jit: ", time.time() - t0)

        # Eq. 6: gets the average of each replicum
        self.avg_replicums = np.mean(self.data, axis=2)

        # Eq. 7: gets the total average
        self.avg_data = np.empty(self.N_obs)
        for ia in range(self.N_obs):
            self.avg_data[ia] = np.mean(
                np.dot(self.NR, self.avg_replicums[:, ia]))

        # Eq. 33, computing derivatives
        derfun = np.zeros(self.N_obs)
        for ia in range(self.N_obs):  # alpha
            derfun[ia] = \
                self.function_derivative[ia](self.avg_data,
                                             **self.function_parameters)

        # Eq. 33, computed G_F
        self.G_F = np.zeros(int(self.N/2))
        for ia in range(self.N_obs):  # alpha
            for ib in range(self.N_obs):  # beta
                self.G_F += derfun[ia]*derfun[ib]*self.G_ab_t[ia, ib]

        # Eq. 35, array with different integration cutoffs
        CfW = np.array([self.G_F[0]]
                       + [self.G_F[0] + 2.0*np.sum(self.G_F[1:W+1])
                          for W in range(1, int(self.N/2))])
        # CfW = np.array([0.5 + np.sum(self.R[1:iW]) for iW in range(1,int(self.N/2))])

        # Eq. 49, bias correction
        CfW = self._correct_bias(CfW, self.N)

        # Eq. 34
        sigma0 = CfW[0]

        # Eq. 41
        self.tau_int = CfW / (2*sigma0)

        if SParam == False:
            self._probe_and_plot_S_param()
        else:
            self.W = self._gW(
                self._automatic_windowing_procedure(SParam, self.tau_int))

        if np.isnan(self.W):
            raise ValueError("W is nan.")

        self.tau_int_optimal = self.tau_int[self.W]

    @staticmethod
    @nb.njit(cache=True)
    def _correct_bias(CfW, N):
        """
        Eq. 49, bias correction
        """
        return CfW*((2*np.arange(len(CfW)) + 1.0)/float(N) + 1.0)

    def _gW(self, tau):
        """
        Eq. 52, getting optimal W
        """
        for iW, itau in enumerate(tau):
            if iW == 0:
                continue
            if (np.exp(-iW/itau) - itau/np.sqrt(iW*float(self.N))) < 0.0:
                return iW
        else:
            return np.nan

    @staticmethod
    @nb.njit(cache=True)
    def _automatic_windowing_procedure(S, tau_int):
        """
        Automatic windowing as described in Wolff paper, section 3.3 
        """
        tau = []
        for it, itauint, in enumerate(tau_int):
            if itauint <= 0.5:
                tau.append(0.00000001)
            else:
                # Eq. 51
                tau.append(S/np.log((2*itauint + 1) / (2*itauint - 1)))

        return tau

    def integrated_autocorrelation_time_error(self):
        """
        Eq. 42, standard deviation of tau_int
        """
        self.tau_int_error = np.asarray(
            [np.sqrt(4/float(self.N)*(float(iW) + 0.5 - itau)*itau**2)
             for iW, itau in enumerate(self.tau_int)])
        # self.tau_int_error = np.sqrt((4*self.W + 2)/float(self.N) * self.tau_int**2)
        self.tau_int_optimal_error = self.tau_int_error[self.W]
        return self.tau_int_optimal_error

    def integrated_autocorrelation_time(self):
        return self.tau_int_optimal


def _testRegularAC(data, N_bins, store_plots, time_ac_functions):
    """Function for testing default autocorrelation method."""

    def chi_beta6_2_derivative(Q_squared):
        const = 0.0763234462734
        return 0.25*const / Q_squared**(0.75)

    def print_values(observable, method, values, autocorr, autocorr_error):
        value_string = observable
        value_string += "\nMethod:                       {0:<s}".format(method)
        value_string += "\nAverage:                      {0:<.8f}".format(
            np.average(values))
        value_string += "\nStd:                          {0:<.8f}".format(
            np.std(values))
        value_string += "\nStd with ac-time correction:  {0:<.8f}".format(
            np.std(data)*np.sqrt(2*autocorr))
        value_string += "\nsqrt(2*tau_int):              {0:<.8f}".format(
            np.sqrt(2*autocorr))
        value_string += "\nIntegrated ac-time:           {0:<.8f}".format(
            autocorr)
        value_string += "\nIntegrated ac-time error:     {0:<.8f}".format(
            autocorr_error)
        print(value_string)

    print("="*20, "RUNNING DEFAULT TEST", "="*20)

    # Autocorrelation
    ac = Autocorrelation(data, method="manual",
                         timefunc=True)
    ac.plot_autocorrelation((r"Autocorrelation for Plaquette "
                             r"$\beta = 6.2, \tau=10.0$"), "beta6_2",
                            dryrun=(not store_plots))
    ac_manual = ac.integrated_autocorrelation_time()
    ac_manual_err = ac.integrated_autocorrelation_time_error()

    # Autocorrelation with numpy corrcoef
    ac_corrcoef = Autocorrelation(data, method="corrcoef",
                                  time_autocorrelation=time_ac_functions)
    ac_corrcoef.plot_autocorrelation(
        (r"Autocorrelation for Plaquette "
         r"$\beta = 6.2, \tau=10.0$ using np\.corrcoef"), "beta6_2",
        dryrun=(not store_plots))
    ac_autocorr = ac_corrcoef.integrated_autocorrelation_time()
    ac_autocorr_err = ac_corrcoef.integrated_autocorrelation_time_error()

    ac_correlate = Autocorrelation(data, method="correlate",
                                   time_autocorrelation=time_ac_functions)
    ac_correlate.plot_autocorrelation(
        (r"Autocorrelation for Plaquette "
         r"$\beta = 6.2, \tau=10.0$ using np\.correlate"), "beta6_2",
        dryrun=(not store_plots))
    ac_autocorr2 = ac_correlate.integrated_autocorrelation_time()
    ac_autocorr_err2 = ac_correlate.integrated_autocorrelation_time_error()

    ac_semi_propagated = PropagatedAutocorrelation(
        data, method="correlate", time_autocorrelation=time_ac_functions)
    ac_semi_propagated.plot_autocorrelation(
        (r"Autocorrelation for Plaquette "
         r"$\beta = 6.2, \tau=10.0$ using semi-full propagated np\.correlate"),
        "beta6_2", dryrun=(not store_plots))
    ac_tau_int_semi = ac_semi_propagated.integrated_autocorrelation_time()
    ac_tau_int_semi_err = \
        ac_semi_propagated.integrated_autocorrelation_time_error()

    ac_propagated = FullAutocorrelation([data],
                                        time_autocorrelation=time_ac_functions)
    ac_propagated.plot_autocorrelation(
        (r"Autocorrelation for Plaquette "
         r"$\beta = 6.2, \tau=10.0$ using full propagated  np\.correlate"),
        "beta6_2", dryrun=(not store_plots))
    ac_tau_int_full = ac_propagated.integrated_autocorrelation_time()
    ac_tau_int_full_err = ac_propagated.integrated_autocorrelation_time_error()

    print_values("Plaquette", "manual", data, ac_manual, ac_manual_err)
    print("")
    print_values("Plaquette", "corrcoef", data, ac_autocorr, ac_autocorr_err)
    print("")
    print_values("Plaquette", "correlate", data,
                 ac_autocorr2, ac_autocorr_err2)
    print("")
    print_values("Plaquette", "semi-full propagated", data,
                 ac_tau_int_semi, ac_tau_int_semi_err)
    print("")
    print_values("Plaquette", "full propagated", data,
                 ac_tau_int_full, ac_tau_int_full_err)
    print("")

    # Differences in time
    print("""
Time used by default method:                    {0:<.8f}
Time used by numpy corrcoef:                    {1:<.8f}
Time used by numpy correlate:                   {2:<.8f}
Time used by semi-full propagated correlate:    {3:<.8f}
Time used by propagated correlate:              {4:<.8f}
Improvement(default/corrcoef):                  {5:<.3f}
Improvement(default/correlate):                 {6:<.3f}
Improvement(corrcoef/correlate):                {7:<.3f}
Improvement(semi-propagated/corrcoef):          {8:<.3f}
Improvement(propagated/corrcoef):               {9:<.3f}
Improvement(propagated/semi-propagated):        {10:<.3f}""".format(
        ac.time_used, ac_corrcoef.time_used, ac_correlate.time_used,
        ac_semi_propagated.time_used, ac_propagated.time_used,
        ac.time_used/ac_corrcoef.time_used,
        ac.time_used/ac_correlate.time_used,
        ac_corrcoef.time_used/ac_correlate.time_used,
        ac_semi_propagated.time_used/ac_corrcoef.time_used,
        ac_propagated.time_used/ac_corrcoef.time_used,
        ac_propagated.time_used/ac_semi_propagated.time_used))

    # Plotting difference
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    ax.semilogy(np.abs(ac.R - ac_corrcoef.R))
    ax.set_title(("Relative difference between numpy "
                  "method and standard method"),
                 fontsize=14)
    ax.grid(True)
    if store_plots:
        fig.savefig("tests/relative_differences_in_ac_methods.pdf")


def _testFullAC(data, N_bins, store_plots, time_ac_functions):
    """Function for testing autocorrelation with error propagation."""

    print("="*20 + "RUNNING FULL AC TEST" + "="*20)

    def chi_beta6_2_derivative(Q_squared):
        const = 0.0763234462734
        return 0.25*const / Q_squared**(0.75)

    print "Autocorrelation:"
    ac1 = Autocorrelation(data, method="corrcoef", timefunc=True)
    ac1.plot_autocorrelation((r"Autocorrelation for Topological "
                              r"Suscpetibility $\beta = 6.2$"),
                             "beta6_2_topc", dryrun=(not store_plots))
    print(ac1.integrated_autocorrelation_time(plot_cutoff=True))
    print(ac1.integrated_autocorrelation_time_error())
    print(ac1.W)

    print "PropagatedAutocorrelation:"
    ac2 = PropagatedAutocorrelation(data,
                                    function_derivative=chi_beta6_2_derivative,
                                    method="corrcoef", timefunc=True)
    ac2.plot_autocorrelation((r"Autocorrelation for Topological Suscpetibility"
                              r" $\beta = 6.2$"), "beta6_2_topc",
                             dryrun=(not store_plots))
    print(ac2.integrated_autocorrelation_time())
    print(ac2.integrated_autocorrelation_time_error())
    print(ac2.W)

    print "FullAutocorrelation:"
    ac3 = FullAutocorrelation([data],
                              function_derivative=[chi_beta6_2_derivative], 
                              timefunc=True)
    ac3.plot_autocorrelation((r"Autocorrelation for Topological Suscpetibility"
                              r" $\beta = 6.2$"), "beta6_2_topc",
                             dryrun=(not store_plots))
    print(ac3.integrated_autocorrelation_time())
    print(ac3.integrated_autocorrelation_time_error())
    print(ac3.W)


def main():
    # Data to load and analyse
    data_plaq = np.loadtxt("tests/plaq_beta6_2_t10.dat")
    data_topc = (np.loadtxt("tests/topc_beta6_2_t10.dat"))**2

    # Histogram bins
    N_bins = 20
    store_plots = True
    time_ac_functions = True

    # _testRegularAC(data_plaq, N_bins, store_plots, time_ac_functions)
    _testFullAC(data_topc, N_bins, store_plots, time_ac_functions)
    # plt.show()


if __name__ == '__main__':
    main()
