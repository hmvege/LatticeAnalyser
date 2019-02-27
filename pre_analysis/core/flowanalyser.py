#!/usr/bin/env python2

from matplotlib import rc, rcParams
import types
import copy
import multiprocessing
from tools.folderreadingtools import FlowDataReader
from tools.folderreadingtools import check_folder
from tools.folderreadingtools import write_data_to_file
from tools.folderreadingtools import write_raw_analysis_to_file
from tools.latticefunctions import get_lattice_spacing
from statistics.jackknife import Jackknife
from statistics.bootstrap import Bootstrap
from statistics.autocorrelation import Autocorrelation, FullAutocorrelation, \
    PropagatedAutocorrelation
import statistics.parallel_tools as ptools
import os
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

__all__ = ["FlowAnalyser"]

rc("text", usetex=True)
# rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
rcParams["font.family"] += ["serif"]


class FlowAnalyser(object):
    observable_name = "Missing_Observable_Name"
    observable_name_compact = "missing_obs_name"
    x_label = "Missing x-label"
    y_label = "Missing y-label"
    mark_interval = 5
    error_mark_interval = 5
    autocorrelations_limits = 1
    figures_folder = "figures"
    fname_addon = ""
    section_seperator = "="*160
    hbarc = 0.19732697  # eV micro m

    include_title = False

    # Function derivative to be used in the autocorrelation class
    function_derivative = [ptools._default_return]

    # Resolution in figures created
    dpi = 350

    # Default figure label is None
    fig_label = None

    # Hline and Vline plots. Internal variables, accesible through children.
    plot_hline_at = None
    plot_vline_at = None

    def __init__(self, data, mc_interval=None, figures_folder=False,
                 parallel=False, numprocs=4, dryrun=False, verbose=False):
        """
        Parent class for analyzing flowed observables.

        Args:
                data: DataReader([observable_name]), an DataReader object
                        called with the compact observable name. Options:
                        "plaq", "energy", "topc".
                mc_interval: optional, tuple, will only look at Monte Carlo
                        history inside interval. Default is using all
                        Monte Carlo of the Monte Carlo history available.
                figures_folder: optional argument for where to place the
                        figures created. Default is "../figures".
                parallel: optinal argument if we are to run analysis in
                        parallel. Default is False.
                numprocs: optional argument for the number of processors to
                        use. Default is 4.
                dryrun: optional dryrun mode. Default is False.
                verbose: optional argument for a more verbose run. Default is
                        False.

        Returns:
                Object for analyzing flow.
        """
        # Retrieves data from data
        self.batch_name = data["batch_name"]
        self.batch_data_folder = data["batch_data_folder"]
        self.x = data["t"]
        self.y = data["obs"]
        self.flow_epsilon = data["FlowEpsilon"]

        # Sets lattice parameters
        self.beta = data["beta"]
        self.a, self.a_err = get_lattice_spacing(self.beta)
        self.r0 = 0.5  # The Sommer parameter

        # Sets the lattice sizes if one is provided
        self.lattice_size = data["lattice_size"]

        # Initializes up global constants
        self.N_bs = None
        self.dryrun = dryrun
        self.verbose = verbose
        if figures_folder != False:  # Default is just figures
            self.figures_folder = figures_folder

        # Parallel variables
        self.parallel = parallel
        self.numprocs = numprocs

        # Checks that a figures folder exists
        check_folder(self.figures_folder, self.dryrun, verbose=self.verbose)

        # Check that a data run folder exist, so one data anlysis performed on
        # different data sets do not mix
        self.data_batch_folder_path = \
            os.path.join(self.figures_folder,
                         os.path.split(self.batch_data_folder)[-1])
        check_folder(self.data_batch_folder_path, self.dryrun,
                     verbose=self.verbose)

        # Checks that a batch folder exists
        self.batch_name_folder_path = os.path.join(self.data_batch_folder_path,
                                                   self.batch_name)
        check_folder(self.batch_name_folder_path, self.dryrun,
                     verbose=self.verbose)

        # Checks that observable output folder exist, and if not will create it
        self.observable_output_folder_path = os.path.join(
            self.batch_name_folder_path, self.observable_name_compact)
        check_folder(self.observable_output_folder_path, self.dryrun,
                     verbose=self.verbose)

        # Sets up the post analysis folder, but do not create it till its
        # needed.
        self.post_analysis_folder_base = os.path.join(self.batch_data_folder,
                                                      self.batch_name,
                                                      "post_analysis_data")
        check_folder(self.post_analysis_folder_base, self.dryrun,
                     verbose=self.verbose)

        # Checks that {post_analysis_folder}/{observable_name} exists
        self.post_analysis_folder = \
            os.path.join(self.post_analysis_folder_base,
                         self.observable_name_compact)
        check_folder(self.post_analysis_folder, self.dryrun,
                     verbose=self.verbose)

        # Sets the MC interval
        self._set_mc_interval(mc_interval)

        # Makes a backup, for later use
        self.post_analysis_folder_old = self.post_analysis_folder

        # Checks if we already have scaled the x values or not
        # print(np.all(np.abs(np.diff(self.x) - self.flow_epsilon) > 1e-14))
        if np.all(np.abs(np.diff(self.x) - self.flow_epsilon) > 1e-14):
            self.x = self.x * self.flow_epsilon
            self.pre_scale = False
        else:
            self.pre_scale = True

        # Max plotting window variables
        self.y_limits = [None, None]

        # Default type of observables, one per configuration per flow
        self.N_configurations, self.NFlows = self.y.shape[:2]

        self._analysis_arrays_setup()

    def _analysis_arrays_setup(self):
        """Initializes the arrays used for storing analaysis results in."""
        # Sets up the function derivative parameters as an empty dictionary
        self.function_derivative_parameters = [{} for tf in range(self.NFlows)]

        # Non-bootstrapped data
        self.unanalyzed_y = np.zeros(self.NFlows)
        self.unanalyzed_y_std = np.zeros(self.NFlows)
        self.unanalyzed_y_data = np.zeros((self.NFlows, self.N_configurations))

        # Bootstrap data
        self.bootstrap_performed = False
        self.bs_y = np.zeros(self.NFlows)
        self.bs_y_std = np.zeros(self.NFlows)

        # Jackknifed data
        self.jackknife_performed = False
        self.jk_y = np.zeros(self.NFlows)
        self.jk_y_std = np.zeros(self.NFlows)

        # Autocorrelation data
        self.autocorrelation_performed = False
        self.autocorrelations = np.zeros(
            (self.NFlows, self.N_configurations/2))
        self.autocorrelations_errors = np.zeros(
            (self.NFlows, self.N_configurations/2))
        self.integrated_autocorrelation_time = np.ones(self.NFlows)
        self.integrated_autocorrelation_time_error = np.zeros(self.NFlows)
        self.autocorrelation_error_correction = np.ones(self.NFlows)

    def _set_q0_time_and_index(self, q0_flow_time):
        """
        Internal method for setting q0_flow_time and finding the correct q0
        flow index. To be used in conjunction with qtq0-type variables.

        Args:
                q0_flow_time: float, time at which we set q0 at.

        Raises:
                AssertionError: if q0_flow_time is less than maximum value of
                        a*sqrt(8t).
        """

        # assert q0_flow_time < (self.a * np.sqrt(8*self.x))[-1], \
        #     "q0_flow_time %f exceed maximum flow time value %f." % \
        #     (q0_flow_time, (self.a * np.sqrt(8*self.x))[-1])

        assert q0_flow_time < (self.a * self.x)[-1], \
            "q0_flow_time %f exceed maximum flow time value %f." % \
            (q0_flow_time, (self.a * self.x)[-1])

        # Stores the q0 flow time value
        self.q0_flow_time = q0_flow_time

        # Selects index closest to q0_flow_time
        self.q0_flow_time_index = np.argmin(
            np.abs(self.a * self.x - self.q0_flow_time))

        if self.verbose:
            print("Extracting values at flow time %.2f with index %d for"
                  " observable %s" % (self.q0_flow_time,
                                      self.q0_flow_time_index,
                                      self.observable_name_compact))

    def _check_skip_wolff_condition(self):
        """
        Method for check if we are to skip performing the Wolff
        autocorrelation method. E.g. if we have a negative mean value.
        By default, will assume this is not a problem for observable and
        return True.
        """
        return True

    def _set_mc_interval(self, mc_interval):
        """
        Selects a Monte Carlo interval in the MC history to use based on the
        tuple provided.
        """
        if isinstance(mc_interval, types.NoneType):
            self.mc_interval = mc_interval
            return

        assert isinstance(mc_interval, (tuple, list)), \
            "invalid type: " + type(mc_interval)

        self.mc_interval = mc_interval

        # Sets the new y config range
        self.y = self.y[mc_interval[0]:mc_interval[1], :]

        # Updates the title name or label to include range
        self.fig_label = r"MC interval: $[%d, %d)$" % mc_interval
        # Checks and creates file folders for mc interval
        self.observable_output_folder_path = os.path.join(
            self.observable_output_folder_path,
            "MCint%03d-%03d" % mc_interval)
        check_folder(self.observable_output_folder_path,
                     self.dryrun, self.verbose)

        # Checks that {post_analysis_folder}/{observable_name}/{mc-interval}
        # exists.
        self.post_analysis_folder = \
            os.path.join(self.post_analysis_folder,
                         "%03d-%03d" % self.mc_interval)
        check_folder(self.post_analysis_folder, self.dryrun, self.verbose)

    def __check_ac(self, fname):
        """
        Internal method to check if autocorrelation has been performed, if
        false it will add "_noErrorCorrection" to the filename.

        Args:
                fname: str, filename for autocorrelation.
        """
        head, ext = os.path.splitext(fname)
        fname_addon = "_noErrorCorrection"
        if not self.autocorrelation_performed:
            return head + "_noErrorCorrection" + ext
        else:
            return head + ext

    def save_post_analysis_data(self, save_as_txt=False):
        """Saves post analysis data to a file."""
        if self.bootstrap_performed:
            write_data_to_file(self, save_as_txt=save_as_txt)

    def save_raw_analysis_data(self, data, analysis_type):
        """Saves the raw analysis data to post analysis folder as binary."""
        write_raw_analysis_to_file(data, analysis_type,
                                   self.observable_name_compact,
                                   self.post_analysis_folder,
                                   dryrun=self.dryrun, verbose=self.verbose)

    def boot(self, N_bs, F=None, F_error=None, store_raw_bs_values=True,
             index_lists=None):
        """
        Bootstrap caller for the flow analysis.

        Args:
                N_bs: number of bootstraps to perform.
                F: optional argument for function that will modify data.
                    Default is None.
                F_error: optional argument for F function for propagating
                    error. Default is None.
                store_raw_bs_values: optional argument for storing raw
                    bootstrap datasets, default is True.
        """

        # Sets default parameters
        if F == None:
            F = ptools._default_return
        if F_error == None:
            F_error = ptools._default_error_return

        # Stores number of bootstraps
        self.N_bs = N_bs

        # Sets up raw bootstrap and unanalyzed data array
        self.bs_y_data = np.zeros((self.NFlows, self.N_bs))
        self.unanalyzed_y_data = np.zeros((self.NFlows, self.N_configurations))

        # Generates random lists to use in resampling
        if isinstance(index_lists, types.NoneType):
            index_lists = np.random.randint(self.N_configurations,
                                            size=(self.N_bs,
                                                  self.N_configurations))

        if self.parallel:
            # Sets up jobs for parallel processing
            input_values = zip([self.y[:, i] for i in xrange(self.NFlows)],
                               [N_bs for i in xrange(self.NFlows)],
                               [index_lists for i in xrange(self.NFlows)])

            # Initializes multiprocessing
            pool = multiprocessing.Pool(processes=self.numprocs)

            # Runs parallel processes. Can this be done more efficiently?
            results = pool.map(ptools._bootstrap_parallel_core, input_values)

            # Garbage collection for multiprocessing instance
            pool.close()

            # Populating bootstrap data
            for i in xrange(self.NFlows):
                self.bs_y[i] = results[i][0]
                self.bs_y_std[i] = results[i][1]
                self.unanalyzed_y[i] = results[i][2]
                self.unanalyzed_y_std[i] = results[i][3]

                # Stores last data for histogram and post analysis
                self.bs_y_data[i] = results[i][4]
                self.unanalyzed_y_data[i] = results[i][5]
        else:
            if self.verbose:
                print("Not running parallel bootstrap "
                      "for %s!" % self.observable_name_compact)

            # Non-parallel method for calculating bootstrap
            for i in xrange(self.NFlows):
                bs = Bootstrap(self.y[:, i], N_bs, index_lists=index_lists)
                self.bs_y[i] = bs.bs_avg
                self.bs_y_std[i] = bs.bs_std
                self.unanalyzed_y[i] = bs.avg_original
                self.unanalyzed_y_std[i] = bs.std_original

                # Stores last data for plotting in histogram later
                self.bs_y_data[i] = bs.bs_data
                self.unanalyzed_y_data[i] = bs.data_original

        # Runs bs and unanalyzed data through the F and F_error
        self.bs_y_std = F_error(self.bs_y, self.bs_y_std)
        self.bs_y = F(self.bs_y)

        self.unanalyzed_y_std = F_error(
            self.unanalyzed_y, self.unanalyzed_y_std)
        self.unanalyzed_y = F(self.unanalyzed_y)

        # The error is in that I am taking F, which is not a linear operation
        # on the bootstrap data, which will be averaged afterwards and
        # is therefore not equal the data bootstrap samples.

        # print(self.bs_y.shape, self.bs_y_data.shape)
        # print(self.bs_y[100])
        # print(F(np.mean(self.bs_y_data, axis=1)[100]))
        # exit(1)

        if store_raw_bs_values:
            self.save_raw_analysis_data(self.bs_y_data, "bootstrap")
            self.save_raw_analysis_data(self.unanalyzed_y_data, "unanalyzed")

        # Sets performed flag to true
        self.bootstrap_performed = True

    def jackknife(self, F=None, F_error=None, store_raw_jk_values=True):
        """
        When called, performs either a parallel or non-parallel jackknife,
        using the Jackknife class.

        Args:
                F: optional argument for function that will modify data.
                    Default is None.
                F_error: optional argument for F function for propagating
                    error. Default is None.
                store_raw_jk_values: optional argument for storing raw
                    jackknifed datasets. Default is True.

        Raises:
                AssertError will be made if F and F_error are not None or
                        of types.FunctionType.
        """

        # Sets default parameters
        if F == None:
            F = ptools._default_return
        if F_error == None:
            F_error = ptools._default_error_return

        # Sets up raw jackknife
        self.jk_y_data = np.zeros((self.NFlows, self.N_configurations))

        if self.parallel:
            # Sets up jobs for parallel processing
            input_values = [self.y[:, i] for i in xrange(self.NFlows)]

            # Initializes multiprocessing
            pool = multiprocessing.Pool(processes=self.numprocs)

            # Runs parallel processes. Can this be done more efficiently?
            results = pool.map(ptools._jackknife_parallel_core, input_values)

            # Closes multiprocessing instance for garbage collection
            pool.close()

            # Populating jackknife results
            for i in xrange(self.NFlows):
                self.jk_y[i] = results[i][0]
                self.jk_y_std[i] = results[i][1]
                self.jk_y_data[i] = results[i][2]
        else:
            if self.verbose:
                print("Not running parallel jackknife "
                      "for %s!" % self.observable_name_compact)

            # Non-parallel method for calculating jackknife
            for i in xrange(self.NFlows):
                jk = Jackknife(self.y[:, i])
                self.jk_y[i] = jk.jk_avg
                self.jk_y_std[i] = jk.jk_std
                self.jk_y_data[i] = jk.jk_data

        # Runs data through the F and F_error
        self.jk_y_std = F_error(self.jk_y, self.jk_y_std)
        self.jk_y = F(self.jk_y)

        if store_raw_jk_values:
            self.save_raw_analysis_data(self.jk_y_data, "jackknife")

        # Sets performed flag to true
        self.jackknife_performed = True

    def autocorrelation(self, store_raw_ac_error_correction=True,
                        method="wolff", funder=None, funder_params=None):
        """
        Function for running the autocorrelation routine.

        Args:
                store_raw_ac_error_correction: optional argument for storing
                    the autocorrelation error correction to file.
                method: type of autocorrelation to be performed. Choose from:
                    "wolff"(default), "luscher", "wolff_full".

        Raises:
                KeyError: if method is not a valid one.
        """

        available_ac_methods = ["wolff", "luscher", "wolff_full"]
        if method not in available_ac_methods:
            raise KeyError("%s not a receognized method. Choose from: %s." % (
                method, ", ".join(available_ac_methods)))

        if self.verbose:
            print("Running autocorrelation with %s ac-method" % method)

        # Gets autocorrelation
        if self.parallel:
            # Sets up parallel job
            pool = multiprocessing.Pool(processes=self.numprocs)

            skip_condition = self._check_skip_wolff_condition()

            if method == "wolff" and skip_condition:

                # Sets up function derivative parameters
                fderparams = \
                    [fdparam
                     for fdparam in self.function_derivative_parameters]

                # Sets up jobs for parallel processing
                input_values = zip(
                    [self.y[:, i] for i in xrange(self.NFlows)],
                    [self.function_derivative[0]
                     for i in xrange(self.NFlows)],
                    fderparams)

                # Initiates parallel jobs
                results = pool.map(
                    ptools._autocorrelation_propagated_parallel_core,
                    input_values)

            elif method == "wolff_full" and skip_condition:
                # Sets up jobs for parallel processing
                input_values = zip(
                    [[rep for rep in self.ac_data[i]]
                     for i in xrange(self.NFlows)],
                    [self.function_derivative for i in xrange(
                        self.NFlows)],
                    self.function_derivative_parameters)

                # Initiates parallel jobs
                results = pool.map(
                    ptools._autocorrelation_full_parallel_core,
                    input_values)
            else:
                # Sets up jobs for parallel processing
                input_values = zip([self.y[:, i] for i in xrange(self.NFlows)])

                # Initiates parallel jobs
                results = pool.map(ptools._autocorrelation_parallel_core,
                                   input_values)

            # Closes multiprocessing instance for garbage collection
            pool.close()

            # Populating autocorrelation results
            for i in xrange(self.NFlows):
                self.autocorrelations[i] = results[i][0]
                self.autocorrelations_errors[i] = results[i][1]
                self.integrated_autocorrelation_time[i] = results[i][2]
                self.integrated_autocorrelation_time_error[i] = results[i][3]
                self.autocorrelation_error_correction[i] = \
                    np.sqrt(2*self.integrated_autocorrelation_time[i])

        else:
            if self.verbose:
                print("Not running parallel autocorrelation "
                      "for %s!" % self.observable_name_compact)

            # Non-parallel method for calculating autocorrelation
            for i in xrange(self.NFlows):
                if method == "wolff":
                    fderparams_ = self.function_derivative_parameters[i]
                    ac = PropagatedAutocorrelation(
                        self.y[:, i],
                        function_derivative=self.function_derivative[0],
                        function_parameters=fderparams_)
                if method == "wolff_full":
                    fderparams_ = self.function_derivative_parameters[i]
                    ac = FullAutocorrelation(
                        [rep for rep in self.ac_data[i]],
                        function_derivative=self.function_derivative,
                        function_parameters=fderparams_)
                else:
                    ac = Autocorrelation(self.y[:, i])

                self.autocorrelations[i] = ac.R
                self.autocorrelations_errors[i] = ac.R_error
                self.integrated_autocorrelation_time[i] = \
                    ac.integrated_autocorrelation_time()
                self.integrated_autocorrelation_time_error[i] = \
                    ac.integrated_autocorrelation_time_error()
                self.autocorrelation_error_correction[i] = \
                    np.sqrt(2*self.integrated_autocorrelation_time[i])

        # Stores the ac error correction
        if store_raw_ac_error_correction:
            self.save_raw_analysis_data(self.autocorrelation_error_correction,
                                        "autocorrelation")

        # Sets performed flag to true
        self.autocorrelation_performed = True

    def plot_jackknife(self, x=None, correction_function=lambda x: x,
                       error_correction_function=None):
        """
        Function for plotting the jackknifed data.

        Args:
                x: optional values to plot along the x-axis. Default is plotting
                        self.a*np.sqrt(8*t).
                correction_function: function to correct y-axis values with.
                        Default is to leave them unmodified.
                error_correction_function: function that corrects the error term.
                        Takes y and y_err. Default is this being equal to correction 
                        function.

        Raises:
                ValueError: if jackknife has not been performed yet.
        """

        # Checks that jacknifing has been performed.
        if not self.jackknife_performed:
            raise ValueError("Jackknifing has not been performed yet.")

        # Sets up the x axis array to be plotted
        if isinstance(x, types.NoneType):
            # Default x axis points is the flow time
            x = self.a * np.sqrt(8*self.x)

        if isinstance(error_correction_function, types.NoneType):
            def error_correction_function(
                q, qerr): return correction_function(qerr)

        # Copies data over to arrays to be plotted
        y = self.jk_y
        y_std = self.jk_y_std*self.autocorrelation_error_correction

        # Sets up the title and filename strings
        title_string = r"Jacknife of %s" % self.observable_name
        fname_path = os.path.join(
            self.observable_output_folder_path,
            "{0:<s}_jackknife_beta{1:<s}{2:<s}.pdf".format(
                self.observable_name_compact, str(
                    self.beta).replace('.', '_'),
                self.fname_addon))

        # Plots the jackknifed data
        self.__plot_error_core(x, correction_function(y),
                               error_correction_function(y, y_std),
                               title_string, fname_path)

    def plot_boot(self, x=None, correction_function=lambda x: x,
                  error_correction_function=None, _plot_bs=True):
        """
        Function for plotting the bootstrapped data.

        Args:
                x: optional values to plot along the x-axis. Default is plotting
                        self.a*np.sqrt(8*t).
                correction_function: function to correct y-axis values with.
                        Default is to leave them unmodified.
                error_correction_function: function that corrects the error term.
                        Takes y and y_err. Default is this being equal to correction 
                        function.
                _plot_bs: internval argument for plotting the bootstrapped and
                        non-bootstrapped values. Default is True.

        Raises:
                ValueError: if bootstrap has not been performed yet.
        """

        # Checks that the bootstrap has been performed.
        if not self.bootstrap_performed and _plot_bs:
            raise ValueError("Bootstrap has not been performed yet.")

        # Retrieves relevant data and sets up the arrays to be plotted
        if isinstance(x, types.NoneType):
            # Default x axis points is the flow time
            x = self.a * np.sqrt(8*self.x)

        if isinstance(error_correction_function, types.NoneType):
            def error_correction_function(
                q, qerr): return correction_function(qerr)

        # Determines if we are to plot bootstrap or original and retrieves data
        if _plot_bs:
            y = self.bs_y
            y_std = self.bs_y_std*self.autocorrelation_error_correction
        else:
            y = self.unanalyzed_y
            y_std = self.unanalyzed_y_std*self.autocorrelation_error_correction

        # Sets up the title and filename strings
        if _plot_bs:
            title_string = ""
            if len(self.observable_name) != 0:
                title_string = r"%s, " % self.observable_name
            title_string += r"$N_\mathrm{bootstraps}=%d$" % self.N_bs

            fname_path = os.path.join(
                self.observable_output_folder_path,
                "{0:<s}_bootstrap_Nbs{2:<d}_beta{1:<s}{3:<s}.pdf".format(
                    self.observable_name_compact,
                    str(self.beta).replace('.', '_'),
                    self.N_bs, self.fname_addon))
        else:
            title_string = r"%s" % self.observable_name
            fname_path = os.path.join(
                self.observable_output_folder_path,
                "{0:<s}_original_beta{1:<s}{2:<s}.pdf".format(
                    self.observable_name_compact,
                    str(self.beta).replace('.', '_'), self.fname_addon))

        # Plots either bootstrapped or regular stuff
        self.__plot_error_core(x, correction_function(y),
                               error_correction_function(y, y_std),
                               title_string, fname_path)

    def plot_original(self, x=None, correction_function=lambda x: x,
                      error_correction_function=None):
        """
        Plots the default analysis, mean and std of the observable.

        Args:
                x: optional values to plot along the x-axis. Default is plotting
                        self.a*np.sqrt(8*t).
                correction_function: function to correct y-axis values with.
                        Default is to leave them unmodified.
                error_correction_function: function that corrects the error term.
                        Takes y and y_err. Default is this being equal to correction 
                        function.

        """
        self.plot_boot(x=x, correction_function=correction_function,
                       error_correction_function=error_correction_function,
                       _plot_bs=False)

    def __plot_error_core(self, x, y, y_std, title_string, fname):
        """
        Interternal plotter function. Forms basis for unanalyzed, jackknifed 
        and bootstrapped plotters.

        Args:
                x: values to plot along the x-axis.
                y: values to plot along the y-axis.

        """
        # Plots the jackknifed data
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plots the error bar
        ax.errorbar(x, y, yerr=y_std, fmt=".", color="0", ecolor="r",
                    markevery=self.mark_interval,
                    errorevery=self.error_mark_interval)

        # Plots hline/vline at specified position.
        # Needed for plotting at e.g. t0.
        if not (self.plot_hline_at, types.NoneType):
            ax.axhline(self.plot_hline_at, linestyle="--",
                       color="0", alpha=0.3)
        if not isinstance(self.plot_vline_at, types.NoneType):
            ax.axvline(self.plot_vline_at, linestyle="--",
                       color="0", alpha=0.3)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_ylim(self.y_limits)
        ax.grid(True)
        if self.include_title:
            ax.set_title(title_string)
        if not isinstance(self.fig_label, types.NoneType):
            ax.legend([self.fig_label])
        if not self.dryrun:
            fig.savefig(self.__check_ac(fname), dpi=self.dpi)
        if self.verbose or self.dryrun:
            print("Figure created in %s" % fname)

        # plt.show()
        plt.close(fig)

    def plot_autocorrelation(self, flow_time_index):
        """
        Plots the autocorrelation at a given flow time.

        Args:
                flow_time_index: integer of the flow time index.

        Raises:
                ValueError: if no autocorrelation has been performed yet.
                AssertionError: if the provided flow_time_index is out of bounds.
        """

        # Checks that autocorrelations has been performed.
        if not self.autocorrelation_performed:
            raise ValueError("Autocorrelation has not been performed yet.")

        # sets up the autocorrelation
        N_autocorr = self.N_configurations / 2

        # Converts flow_time_index if it is minus 1
        if flow_time_index == -1:
            flow_time_index = self.NFlows - 1

        # Ensures flow time is within bounds.
        assert flow_time_index < self.NFlows, ("Flow time %d is out of "
                                               "bounds." % flow_time_index)

        # Finds the maximum value at each MC time and sets up the y array
        x = range(N_autocorr)
        y = self.autocorrelations[flow_time_index, :]
        y_std = self.autocorrelations_errors[flow_time_index, :]

        # Sets up the title and filename strings
        title_string = r"Autocorrelation of %s, $t_f=%.2f$" % (
            self.observable_name, flow_time_index*self.flow_epsilon)
        fname_path = os.path.join(
            self.observable_output_folder_path,
            "{0:<s}_autocorrelation_flowt{1:<d}_beta{2:<s}{3:<s}.pdf".format(
                self.observable_name_compact, flow_time_index,
                str(self.beta).replace('.', '_'), self.fname_addon))

        # Plots the autocorrelations
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot(x,y,color="0",label=self.observable_name)
        ax.errorbar(x, y, yerr=y_std, color="0", ecolor="r")
        ax.set_ylim(-self.autocorrelations_limits,
                    self.autocorrelations_limits)
        ax.set_xlim(0, N_autocorr)
        ax.set_xlabel(r"Lag $h$")
        ax.set_ylabel(r"$R = \frac{C_h}{C_0}$")
        if self.include_title:
            ax.set_title(title_string)
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, 0.2))
        ax.grid(True)
        # ax.legend()
        if not self.dryrun:
            fig.savefig(fname_path, dpi=self.dpi)
        if self.verbose or self.dryrun:
            print("Figure created in %s" % fname_path)
        plt.close(fig)

    def plot_integrated_correlation_time(self):
        """
        Plots the integrated correlation through the flowing.
        """
        # Sets up values to be plotted
        y = self.integrated_autocorrelation_time
        y_std = self.integrated_autocorrelation_time_error
        x = self.a*np.sqrt(8*self.x)

        # Gives title and file name
        title_string = r"$\tau_\mathrm{int}$ of %s, $N_\mathrm{cfg}=%2d$" % (
            self.observable_name, self.N_configurations)
        fname_path = os.path.join(
            self.observable_output_folder_path,
            "{0:<s}_integrated_ac_time_beta{1:<s}{2:<s}.pdf".format(
                self.observable_name_compact,
                str(self.beta).replace('.', '_'), self.fname_addon))

        # Sets up the plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot with error
        ax.plot(x, y, color="0")
        ax.fill_between(x, y - y_std, y + y_std,
                        alpha=0.5, edgecolor='', facecolor='#6699ff')
        ax.set_xlabel(r"$\sqrt{8t_f}$")
        ax.set_ylabel(r"$\tau_\mathrm{int}$")
        if self.include_title:
            ax.set_title(title_string)
        ax.grid(True)
        if not self.dryrun:
            fig.savefig(fname_path, dpi=self.dpi)
        if self.verbose or self.dryrun:
            print("Figure created in %s" % fname_path)
        plt.close(fig)

    def plot_histogram(self, flow_time_index, x_label=None, NBins=None,
                       x_limits="equal", F=lambda x: x):
        """
        Function for creating histograms of the original, bootstrapped and 
        jackknifed datasets together.

        Args:
                flow_time_index: int, Flow time to plot.
                x_label: str, x-axis label for plot.
                NBins: int, optional, number of histogram bins.
                x_limits: str, optional, type of x-axis limits. Default: 'auto'.
                        Choices: 'equal','auto','analysis'.
                F: function, optional, function to run data through. Default is
                        a f(x): x.

        Raises:
                AssertionError: if the lengths of the different analyzed data sets 
                        differ.
                AssertionError: if the flow index exceeds the available total flow
                        time.
        """
        N_unanalyzed = len(self.unanalyzed_y_data)

        # Checks if we have a derived histogram_bins instance of class.
        if not hasattr(self, "histogram_bins"):
            if isinstance(NBins, types.NoneType):
                NBins = 30
        else:
            NBins = self.histogram_bins

        # Setting proper flow-time
        if flow_time_index < 0:
            flow_time_index = N_unanalyzed - abs(flow_time_index)
            assert N_unanalyzed == len(self.bs_y_data) == len(self.jk_y_data),\
                "Flow lengths of data sets is not equal!"

        # X-label set as the default y-label
        if isinstance(x_label, types.NoneType):
            x_label = self.y_label

        # Ensures flow time is within bounds.
        assertion_str = "Flow time %d is out of bounds." % flow_time_index
        assert flow_time_index < N_unanalyzed, assertion_str

        # Sets up title and file name strings
        title_string = r"Spread of %s, $t_f=%.2f$" % (
            self.observable_name, flow_time_index*self.flow_epsilon)
        fname_path = os.path.join(
            self.observable_output_folder_path,
            "{0:<s}_histogram_flowt{1:>04d}_beta{2:<s}{3:<s}.pdf".format(
                self.observable_name_compact, abs(
                    flow_time_index),
                str(self.beta).replace('.', '_'), self.fname_addon))

        # Sets up plot
        fig = plt.figure()

        # Adds unanalyzed data
        ax1 = fig.add_subplot(311)
        x1, y1, _ = ax1.hist(F(self.unanalyzed_y_data[flow_time_index]),
                             bins=NBins, label="Unanalyzed", density=True)
        ax1.legend(loc="upper right")
        ax1.grid(True)
        ax1.set_title(title_string)

        # Adds bootstrapped data
        ax2 = fig.add_subplot(312)
        x2, y2, _ = ax2.hist(F(self.bs_y_data[flow_time_index]),
                             bins=NBins, label="Bootstrap", density=True)
        ax2.grid(True)
        ax2.legend(loc="upper right")
        ax2.set_ylabel("Hits(normalized)")

        # Adds jackknifed histogram
        ax3 = fig.add_subplot(313)
        x3, y3, _ = ax3.hist(F(self.jk_y_data[flow_time_index]),
                             bins=NBins, label="Jackknife", density=True)
        ax3.legend(loc="upper right")
        ax3.grid(True)
        ax3.set_xlabel(r"%s" % x_label)

        if x_limits == "auto":
            # Lets matplotlib decide on axes
            xlim_positive = None
            xlim_negative = None
        elif x_limits == "equal":
            # Sets the x-axes to be equal
            xlim_positive = np.max([y1, y2, y3])
            xlim_negative = np.min([y1, y2, y3])

            # Sets the axes limits
            ax1.set_xlim(xlim_negative, xlim_positive)
        elif x_limits == "analysis":
            # Sets only the analysises axes equal
            xlim_positive = np.max([y2, y3])
            xlim_negative = np.min([y2, y3])
        else:
            raise KeyError(("%s not recognized.\nOptions: 'equal', 'auto', "
                            "'analysis'." % x_limits))

        # Sets the axes limits
        ax2.set_xlim(xlim_negative, xlim_positive)
        ax3.set_xlim(xlim_negative, xlim_positive)

        # Saves figure
        if not self.dryrun:
            plt.savefig(fname_path, dpi=self.dpi)
        if self.verbose or self.dryrun:
            print("Figure created in %s" % fname_path)

        # Closes figure for garbage collection
        plt.close(fig)

    def plot_multihist(self, histogram_slices, NBins=None, x_label=None,
                       F=lambda x: x):
        """
        Method for plotting multiple histograms in the same plot sequentially.

        Args:
                histogram_slices: list containing 3 flow time points of where to plot.
                x_label: str for x-axis label, optional.
                NBins: int, optional. Histogram binning. Default is 30 bins.
                F: function, optional. Function to run data through. Default is    
                        f(x): x.
        """

        # Checks if we have a derived histogram_bins instance of class.
        if isinstance(NBins, types.NoneType):
            NBins = 30

        # X-label set as the default y-label
        if isinstance(x_label, types.NoneType):
            x_label = self.y_label

        # Sorting the histogram points
        _hist_slices = []
        for iHist in histogram_slices:
            if iHist < 0:
                _hist_slices.append(len(self.unanalyzed_y_data) - abs(iHist))
            else:
                _hist_slices.append(iHist)
        histogram_slices = sorted(_hist_slices)

        # Ensures flow time is within bounds.
        assert len(histogram_slices) == 3, (
            "Too many histogram slices provided is out of bounds: "
            " [%s]" % ", ".join([str(iHist) for iHist in histogram_slices]))

        # Sets up title and file name strings
        title_string = r"Evolution of %s" % self.observable_name
        fname_path = os.path.join(
            self.observable_output_folder_path,
            "{0:<s}_multihistogram_beta{1:<s}{2:<s}.pdf".format(
                self.observable_name_compact, str(
                    self.beta).replace('.', '_'),
                self.fname_addon))

        # Plots histograms
        fig, axes = plt.subplots(3, 1, sharey=True, sharex=True)
        for iHist, ax in zip(histogram_slices, axes):
            t_f = iHist * self.flow_epsilon
            ax.hist(self.unanalyzed_y_data[iHist],
                    bins=NBins, label=r"$t_f=%.2f$" % t_f, density=True)
            ax.grid(True)
            ax.legend(loc="upper right")
            if iHist == histogram_slices[1]:
                ax.set_ylabel("Hits(normalized)")
            if iHist == histogram_slices[2]:
                ax.set_xlabel(x_label)
        fig.suptitle(title_string)

        # Saves figure
        if not self.dryrun:
            plt.savefig(fname_path, dpi=self.dpi)
        if self.verbose or self.dryrun:
            print("Figure created in %s" % fname_path)

        # Closes figure for garbage collection
        plt.close(fig)

    def plot_mc_history(self, flow_time_index,
                        correction_function=lambda x: x):
        """
        Plots the Monte Carlo history at a given flow time .

        Args:
                flow_time_index: index of the flow-time we are to plot for. -1 
                        gives the last flow time element.

        Raises:
                AssertionError: if the flow_time_index is out of bounds.
        """

        # Converts flow_time_index if it is minus 1
        if flow_time_index == -1:
            flow_time_index = self.NFlows - 1

        # Ensures flow time is within bounds.
        assertion_str = "Flow time %d is out of bounds." % flow_time_index
        assert flow_time_index < len(self.unanalyzed_y_data), assertion_str

        # Sets up title and file name strings
        title_string = r"Monte Carlo history at $t_f = %.2f$" % (
            flow_time_index*self.flow_epsilon)
        fname_path = os.path.join(
            self.observable_output_folder_path,
            "{0:<s}_mchistory_flowt{1:>04d}_beta{2:<s}{3:<s}.pdf".format(
                self.observable_name_compact, flow_time_index,
                str(self.beta).replace('.', '_'), self.fname_addon))

        # Sets up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(correction_function(self.unanalyzed_y_data[flow_time_index]),
                color="0")
        ax.set_xlabel(r"Monte Carlo time")
        ax.set_ylabel(self.y_label)
        if self.include_title:
            ax.set_title(title_string)
        ax.grid(True)
        if not isinstance(self.fig_label, types.NoneType):
            ax.legend([self.fig_label])
        if not self.dryrun:
            fig.savefig(fname_path, dpi=self.dpi)
        if self.verbose or self.dryrun:
            print("Figure created in %s" % fname_path)
        plt.close(fig)

    def __str__(self):
        def info_string(s1, s2): return "\n{0:<20s}: {1:<20s}".format(s1, s2)
        return_string = ""
        return_string += "\n" + "="*160
        return_string += info_string("Data batch folder",
                                     self.batch_data_folder)
        return_string += info_string("Batch name", self.batch_name)
        return_string += info_string("Observable",
                                     self.observable_name_compact)
        return_string += info_string("Beta", "%.2f" % self.beta)
        if not isinstance(self.mc_interval, types.NoneType):
            return_string += info_string("MC interval",
                                         "[%d,%d)" % self.mc_interval)

        # Add parallel run info
        return_string += info_string("Parallel", "{}".format(self.parallel))
        if self.parallel:
            return_string += info_string("Numprocs",
                                         "{}".format(self.numprocs))

        return_string += "\n" + "="*160
        return return_string


def main():
    exit("Module FlowAnalyser not intended for standalone usage.")


if __name__ == '__main__':
    main()
