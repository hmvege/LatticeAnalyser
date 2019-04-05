import statistics.parallel_tools as ptools
from statistics.autocorrelation import Autocorrelation
from statistics.linefit import LineFit
from scipy.optimize import curve_fit
from tools.sciprint import sciprint
from tools.latticefunctions import get_lattice_spacing
from tools.folderreadingtools import check_folder
from post_analysis.core.multiplotcore import MultiPlotCore
import multiprocessing
import types
import os
import itertools
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt


class QtQ0EffectiveMassPostAnalysis(MultiPlotCore):
    """Post-analysis of the effective mass."""
    observable_name = r"Effective mass, "
    observable_name += r"$am_\textrm{eff} = \log \frac{C(t_e)}{C(t_e+1)}$, "
    observable_name += r"$C(t_e)=\langle q_t q_0\rangle$"
    observable_name_compact = "qtq0eff"
    x_label = r"$t_e$ [fm]"
    y_label = r"$r_0 m_\textrm{eff}$"
    sub_obs = True
    hbarc = 0.19732697  # eV micro m or GeV fm
    dpi = None
    fold = True
    fold_range = 16
    subfolder_type = "tflow"

    y_label_continuum = r"$m_\mathrm{eff}$ [GeV]"
    x_label_continuum = r"$a^2/t_{0,\mathrm{cont}}$"

    meff_plot_type = "ma"  # Default
    meff_plot_types = ["ma", "m", "r0ma"]

    meff_labels = {
        "ma": r"$m_\mathrm{eff}$",
        "m": r"$am_\mathrm{eff}$",
        "r0ma": r"$r_0 m_\mathrm{eff}$",
    }

    meff_hist_x_limits = {
        "ma": [0.5, 2.0],
        "m": [0, 1.0],
        "r0ma": [1.5, 4],
    }

    meff_hist_y_limits = {
        "ma": [0.0, 3],
        "m": [0, 1.0],
        "r0ma": [1.0, 5],
    }

    meff_y_limits = {
        "ma": [-0.5, 4],
        "m": [-0.5, 2],
        "r0ma": [-1, 10],
    }

    meff_unit_labels = {
        "ma": r"[GeV]",
        "m": r"",
        "r0ma": r"",
    }

    def __init__(self, *args, **kwargs):
        # Ensures we load correct data
        self.observable_name_compact_old = self.observable_name_compact

        super(QtQ0EffectiveMassPostAnalysis, self).__init__(*args, **kwargs)

        # Resets the observable name after data has been loaded.
        self.observable_name_compact = self.observable_name_compact_old

    def _setup_analysis_types(self, atypes):
        """
        Stores the number of analysis types from the data container, while 
        removing the autocorrelation one.
        """
        self.analysis_types = atypes
        if "autocorrelation" in self.analysis_types:
            self.analysis_types.remove("autocorrelation")
        if "autocorrelation_raw" in self.analysis_types:
            self.analysis_types.remove("autocorrelation_raw")
        if "autocorrelation_raw_error" in self.analysis_types:
            self.analysis_types.remove("autocorrelation_raw_error")
        if "blocked" in self.analysis_types:
            self.analysis_types.remove("blocked")
        if "blocked_bootstrap" in self.analysis_types:
            self.analysis_types.remove("blocked_bootstrap")

    def fold_array(self, arr, axis=0):
        """Method for folding an array by its last values."""
        # OLD FOLD METHOD - DO NOT INCREASE STATISTICS BY THIS WAY:|
        # folded_array = np.roll(arr, self.fold_range, axis=axis)
        # folded_array = folded_array[:self.fold_range*2]
        # folded_array[:self.fold_range] *= -1

        # fold_range = int(arr.shape[-1]/2) - 1
        fold_range = arr.shape[-1]/2
        folded_array = arr[:fold_range+1]
        last_part = arr[fold_range+1:] * (-1)
        folded_array[1:-1] = (folded_array[1:-1] +
                              np.flip(last_part, axis=0))*0.5
        # folded_array[1:-1] *= 0.5
        return folded_array

    def fold_error_array(self, arr, axis=0):
        """Method for folding an array by its last values."""
        # OLD FOLD METHOD - DO NOT INCREASE STATISTICS BY THIS WAY:|
        # folded_array = np.roll(arr, self.fold_range, axis=axis)
        # folded_array = folded_array[:self.fold_range*2]
        # folded_array[:self.fold_range] *= -1

        fold_range = arr.shape[-1]/2
        folded_array = arr[:fold_range+1]
        last_part = arr[fold_range+1:] * (-1)
        folded_array[1:-1] = (folded_array[1:-1] +
                              np.flip(last_part, axis=0))*0.5
        folded_array[1:-1] = np.sqrt((0.5*folded_array[1:-1])**2
                                     + (0.5*np.flip(last_part, axis=0))**2)
        return folded_array

    def _convert_label(self, lab):
        return float(lab[-6:])

    def effMass(self, Q, axis=0):
        """Correlator for qtq0."""
        return np.log(Q/np.roll(Q, -1, axis=axis))  # C(t)/C(t+1)

    def effMass_err(self, Q, dQ, axis=0):
        """Correlator for qtq0 with error propagation."""
        q = np.roll(Q, -1, axis=axis)
        dq = np.roll(dQ, -1, axis=axis)
        # return np.sqrt((dQ/Q)**2 + (dq/q)**2 - 2*dq*dQ/(q*Q))
        return np.sqrt((dQ/Q)**2 + (dq/q)**2 - 2*dq*dQ/(q*Q))

    def analyse_raw(self, data, data_raw):
        """
        Method for analysis <QteQ0>_i where i is index of bootstrapped,
        jackknifed or unanalyzed samples.
        """

        # Using bs samples
        y = self.effMass(data["y"])
        y_err = self.effMass_err(data["y"], data["y_error"])

        # NEucl, NCfgs = data_raw.shape
        # if self.analysis_data_type=="unanalyzed":
        #   N_BS = 500
        #   y_raw = np.zeros((NEucl, N_BS))     # Correlator, G
        #   index_lists = np.random.randint(NCfgs, size=(N_BS, NCfgs))
        #   # Performing the bootstrap samples
        #   for i in xrange(NEucl):
        #       for j in xrange(N_BS):
        #           y_raw[i,j] = np.mean(data_raw[i][index_lists[j]])
        # else:
        #   y_raw = data_raw

        # y_raw = np.log(y_raw/np.roll(y_raw, -1, axis=0))
        # y = np.mean(y_raw, axis=1)
        # y_err = np.std(y_raw, axis=1)

        # # Runs parallel processes
        # input_values = zip(   [data_raw[iEucl] for iEucl in range(NEucl)],
        #                   [None for _d in range(NEucl)],
        #                   [{} for _d in range(NEucl)])

        # pool = multiprocessing.Pool(processes=8)
        # res = pool.map(ptools._autocorrelation_propagated_parallel_core, input_values)
        # pool.close()

        # error_correction = np.ones(NEucl)
        # for i, _data in enumerate(data_raw):
        #   error_correction[i] = np.sqrt(2*res[i][2])

        # y_err *= error_correction

        # print "\n"
        # print y[:10]
        # print y_err[:10],"\n"

        # for _res in results:
        #   y_err *= np.sqrt(2*_res[2])

        # C = np.mean(data_raw, axis=1)
        # C_err = np.std(data_raw, axis=1)
        # y = self.effMass(C, axis=0)
        # y_err = self.effMass_err(C, C_err, axis=0)

        return y, y_err

    def analyse_data(self, data):
        """Method for analysis <QteQ0>."""
        return self.effMass(data["y"]), self.effMass_err(data["y"],
                                                         data["y_error"])

    def _get_plot_figure_name(self, output_folder=None,
                              figure_name_appendix=""):
        """Retrieves appropriate figure file name."""
        if isinstance(output_folder, types.NoneType):
            output_folder = os.path.join(self.output_folder_path, "slices")
        check_folder(output_folder, False, True)
        fname = "post_analysis_%s_%s_tf%s%s.pdf" % (
                self.observable_name_compact, self.analysis_data_type,
                str(self.interval_index).replace(".", "_"),
                figure_name_appendix)
        return os.path.join(output_folder, fname)

    def _initiate_plot_values(self, data, data_raw, flow_index=None):
        """interval_index: int, should be in euclidean time."""

        # Sorts data into a format specific for the plotting method
        for bn in self.batch_names:
            values = {}

            if flow_index == None:
                # Case where we have sub sections of observables, e.g. in
                # euclidean time.
                for sub_obs in self.observable_intervals[bn]:
                    sub_values = {}
                    sub_values["a"], sub_values["a_err"] = \
                        get_lattice_spacing(self.beta_values[bn])
                    sub_values["x"] = np.linspace(0,
                                                  self.lattice_sizes[bn][1] *
                                                  sub_values["a"],
                                                  self.lattice_sizes[bn][1])

                    sub_values["y"], sub_values["y_err"] = self.analyse_raw(
                        data[bn][sub_obs],
                        data_raw[bn][self.observable_name_compact][sub_obs])

                    sub_values["label"] = self.ensemble_names[bn]

                    sub_values["raw"] = \
                        data_raw[bn][self.observable_name_compact][sub_obs]

                    if self.fold:
                        sub_values["x"] = np.linspace(
                            0,
                            (int(sub_values["y"].shape[0]/2))*sub_values["a"],
                            int(sub_values["y"].shape[0]/2)+1)

                        sub_values["y"], sub_values["y_err"] = \
                            self._folder_and_propagate(sub_values)

                        self.fold_position = sub_values["x"][self.fold_range]

                    if self.with_autocorr:
                        sub_values["tau_int"] = \
                            data[bn][sub_obs]["ac"]["tau_int"]
                        sub_values["tau_int_err"] = \
                            data[bn][sub_obs]["ac"]["tau_int_err"]

                    values[sub_obs] = sub_values
                self.plot_values[bn] = values

            else:
                tf_index = "tflow%04.4f" % flow_index
                values["a"], values["a_err"] = \
                    get_lattice_spacing(self.beta_values[bn])

                # For exact box sizes
                values["x"] = np.linspace(0,
                                          self.lattice_sizes[bn][1] *
                                          values["a"],
                                          self.lattice_sizes[bn][1])

                values["y_raw"] = \
                    data_raw[bn][self.observable_name_compact][tf_index]

                if self.with_autocorr:
                    values["tau_int"] = data[bn][tf_index]["ac"]["tau_int"]
                    values["tau_int_err"] = \
                        data[bn][tf_index]["ac"]["tau_int_err"]

                values["y"], values["y_err"] = \
                    self.analyse_data(data[bn][tf_index])

                if self.fold:
                    values["x"] = \
                        np.linspace(
                            0, (int(values["y"].shape[0]/2))*values["a"],
                            int(values["y"].shape[0]/2)+1)

                    values["y"], values["y_err"] = self._folder_and_propagate(
                        values)

                    # values["y_raw"] = self.fold_array(values["y_raw"], axis=0)
                    self.fold_position = values["x"][self.fold_range]

                values["label"] = self.ensemble_names[bn]

                self.plot_values[bn] = values

    def _folder_and_propagate(self, values):
        """Depending on what we are plotting, 'm/a', 'm', 'r0m/a',
        will set up the correct fold."""
        # tmp = self.hbarc
        # self.hbarc = 1

        if self.meff_plot_type == "ma":
            # y / a
            y = self.fold_array(values["y"])/values["a"] * self.hbarc
            y_err = np.sqrt(
                (self.fold_error_array(values["y_err"])/values["a"])**2 +
                (y/values["a"]**2*values["a_err"])**2) * self.hbarc

        elif self.meff_plot_type == "m":
            # y
            y = self.fold_array(values["y"])
            y_err = self.fold_error_array(values["y_err"])

        elif self.meff_plot_type == "r0ma":
            # y * r0 / a
            y = \
                self.fold_array(values["y"])/values["a"]*self.r0
            y_err = np.sqrt(
                (self.fold_error_array(values["y_err"])/values["a"])**2 +
                (y/values["a"]**2*values["a_err"])**2)*self.r0

        else:
            raise KeyError(("Effective mass plot type '%s' not recognized "
                            "among %s" % (self.meff_plot_type,
                                          self.meff_plot_types)))

        # self.hbarc = tmp
        return y, y_err

    def plot_interval(self, flow_index, **kwargs):
        """
        Sets and plots only one interval.

        Args:
                flow_index: flow time integer
                euclidean_index: integer for euclidean time
        """
        for meff_plot_type in self.meff_plot_types:

            self.meff_plot_type = meff_plot_type

            self.plot_values = {}
            self.interval_index = flow_index
            self._initiate_plot_values(self.data[self.analysis_data_type],
                                       self.data_raw[self.analysis_data_type],
                                       flow_index=flow_index)

            # Sets the x-label to proper units
            x_label_old = self.x_label
            self.x_label = r"$t_e[fm]$"

            # Makes it a global constant so it can be added in plot figure name
            self.plot(**kwargs)
            self.plot_with_article_masses(**kwargs)

            self.x_label = x_label_old

    def plot(self, *args, **kwargs):
        """Ensuring I am plotting with formule in title."""
        kwargs["plot_with_formula"] = True
        kwargs["error_shape"] = "bars"
        kwargs["x_label"] = self.x_label
        kwargs["y_label"] = self.meff_labels[self.meff_plot_type]
        kwargs["y_limits"] = self.meff_y_limits[self.meff_plot_type]
        kwargs["figure_name_appendix"] = "_" + self.meff_plot_type
        kwargs["legend_position"] = "best"
        kwargs["x_limits"] = [-0.1, 0.8]

        super(QtQ0EffectiveMassPostAnalysis, self)._plot_core(
            self.plot_values, **kwargs)

    def _plateau_fit(self, plateau_range):
        """Method that performs a plateau fit on plateau_range."""

        def _f(x, a):
            """Model function for plateau fitting."""
            return a

        # Used bootstrap as the extrapolation method
        refvals = self.reference_values["bootstrap"][self.analysis_data_type]

        a, a_err, t0, t0_err, meff, meff_err = \
            [[] for i in range(6)]

        for bn in self.sorted_batch_names:

            # Retrieves values for ensemble
            _x = self.plot_values[bn]["x"]
            _y = self.plot_values[bn]["y"]
            _yerr = self.plot_values[bn]["y_err"]

            # Sets up the plateau indexes
            # Lowest index
            min_ind = np.where(plateau_range[0] <= _x)[0][0]
            # Plateau index range
            pind = np.where(_x[min_ind:] <= plateau_range[1])

            # Skips in case of nan -> bad statistics
            if np.isnan(_y[min_ind:][pind]).any():
                continue

            # Performs a curve fit, which returns an actual estimate on the
            _res = curve_fit(
                _f, _x[min_ind:][pind], _y[min_ind:][pind],
                sigma=_yerr[min_ind:][pind])

            # Stores the results
            a.append(self.plot_values[bn]["a"])
            a_err.append(self.plot_values[bn]["a_err"])
            meff.append(_res[0][0])
            meff_err.append(np.sqrt(_res[1][0][0]))
            t0.append(refvals[bn]["t0"])
            t0_err.append(refvals[bn]["t0err"])

        # Converts lists to arrays
        a, a_err, meff, meff_err, t0, t0_err = \
            map(lambda _k: np.array(_k)[::-1],
                [a, a_err, meff, meff_err, t0, t0_err])

        return a, a_err, meff, meff_err, t0, t0_err

    def get_plateau_value(self, flow_index, plateau_ranges,
                          meff_plot_type):
        """Calculates the extrapolated plateau value with systematic error
        estimates."""
        self.meff_plot_type = meff_plot_type
        self.plot_values = {}
        self.interval_index = flow_index
        self._initiate_plot_values(self.data[self.analysis_data_type],
                                   self.data_raw[self.analysis_data_type],
                                   flow_index=flow_index)

        # Sets the x-label to proper units
        x_label_old = self.x_label
        self.x_label = r"$t_e[fm]$"

        # Systematic error retrieved by going through:
        plateau_ranges = []

        range_start = 0.1
        range_stop = 0.8
        range_step_size = 0.1

        # Range start
        for _prange_start in np.arange(range_start, range_stop,
                                       range_step_size):

            # Range stop
            for _prange_stop in np.arange(_prange_start + range_step_size,
                                          range_stop + range_step_size,
                                          range_step_size):
                plateau_ranges.append([_prange_start, _prange_stop])

        meff_values, meff_err_values, chi2_values = [], [], []

        for i, _prange in enumerate(plateau_ranges):

            # Performs a plateau fit
            a, a_err, meff, meff_err, t0, t0_err = \
                self._plateau_fit(_prange)

            if len(a) == 0:
                print("Too few values retrieved from plateau fit for"
                      " range", _prange)
                return

            # Propagates error a
            a_squared = a**2 / t0

            # Continuum limit arrays
            N_cont = 1000
            a_squared_cont = np.linspace(-0.025, a_squared[-1]*1.1, N_cont)

            # Performs a continuum extrapolation of the effective mass
            continuum_fit = LineFit(a_squared, meff, meff_err)
            y_cont, y_cont_err, fit_params, chi_squared = \
                continuum_fit.fit_weighted(a_squared_cont)

            cont_fit_params = fit_params

            # Gets the continuum value and its error
            y0_cont, y0_cont_err, _, _, = \
                continuum_fit.fit_weighted(0.0)

            # Matplotlib requires 2 point to plot error bars at
            y0 = [y0_cont[0], y0_cont[0]]
            y0_err = [y0_cont_err[0][0], y0_cont_err[1][0]]

            # Stores the continuum mass
            meff_cont = y0[0]
            meff_cont_err = (y0_err[1] - y0_err[0])/2.0

            _lowlim = self.meff_hist_y_limits[self.meff_plot_type][0]
            _upplim = self.meff_hist_y_limits[self.meff_plot_type][1]
            if (meff_cont < _lowlim or meff_cont > _upplim):
                # print "Skipping bad interval for {}".format(_prange)
                continue

            if np.isnan(meff_cont_err) or np.isnan(meff_cont):
                continue

            # Store chi^2
            chi2_values.append(chi_squared)

            # Store mass + error
            meff_values.append(meff_cont)
            meff_err_values.append(meff_cont_err)

        meff_values, meff_err_values, chi2_values = map(
            np.asarray,
            [meff_values, meff_err_values, chi2_values])

        systematic_error = np.std(meff_values)  # /len(meff_values)

        # print "Systematic error for %s: %g" % (
        #     self.meff_plot_type,
        #     systematic_error)
        if self.skipped_plateau_plot:
            print("Skipping get_plateau_value since we could not "
                  "perform a proper plteau plot.")
            return

        assert hasattr(self, "meff_cont"), "Run plot_plateau."
        assert hasattr(self, "meff_cont_err"), "Run plot_plateau."

        # Gets the systematic error in correct str format
        sys_error_str = sciprint(self.meff_cont, systematic_error, prec=3)
        sys_error_str = "(" + sys_error_str.split("(")[-1]

        # Sets up string for effective mass
        eff_mass_str = "{}{}".format(
            sciprint(self.meff_cont, self.meff_cont_err, prec=3),
            sys_error_str)

        msg = "Effective mass for {}: {}".format(
            self.meff_plot_type, eff_mass_str)
        print msg

        # Method 1
        # Make histogram of mass
        # Take std of histogram
        fig1, ax1 = plt.subplots()
        ax1.hist(meff_values,
                 label=r"%s$=%s$%s" % (
                     self.meff_labels[self.meff_plot_type], eff_mass_str,
                     self.meff_unit_labels[self.meff_plot_type]),
                 density=True)
        ax1.grid(True)
        ax1.set_xlabel("%s%s" % (
            self.meff_labels[self.meff_plot_type],
            self.meff_unit_labels[self.meff_plot_type]))
        ax1.set_xlim(self.meff_hist_x_limits[self.meff_plot_type])
        fig1.legend(loc="upper right")

        # Saves and closes figure
        fname = self._get_plot_figure_name(
            output_folder=None,
            figure_name_appendix="_{0:s}_syserror_unweighted".format(
                self.meff_plot_type))
        plt.savefig(fname)
        if self.verbose:
            print "Figure saved in %s" % fname
        plt.close(fig1)

        # # Method 2
        # # Make histgram weighted by chi^2
        # # Take std of weighted histogram
        # fig2, ax2 = plt.subplots()
        # ax2.hist(meff_values,
        #          label=r"%s$=%s$%s" % (
        #              meff_labels[self.meff_plot_type], eff_mass_str,
        #              meff_unit_labels[self.meff_plot_type]),
        #          density=True)
        # ax2.grid(True)
        # ax2.set_ylabel("%s%s"%(meff_labels[self.meff_plot_type]
        #     meff_unit_labels[self.meff_plot_type]))
        # ax2.set_xlim(self.meff_hist_x_limits[self.meff_plot_type])
        # fig2.legend(loc="upper right")

        # # Saves and closes figure
        # fname = self._get_plot_figure_name(
        #     output_folder=None,
        #     figure_name_appendix="_{0:s}_syserror_weighted".format(
        #         self.meff_plot_type))
        # plt2.savefig(fname)
        # if self.verbose:
        #     print "Figure saved in %s" % fname
        # plt.close(fig1)

    def _get_continuum_mass_estimate(self):
        """Method for getting single continuum estimate."""

        pass

    def plot_plateau(self, flow_index, plateau_limits, meff_plot_type="ma"):
        """Method for extracting the glueball mass and plot plateau."""
        self.meff_plot_type = meff_plot_type
        self.plot_values = {}
        self.interval_index = flow_index
        self._initiate_plot_values(self.data[self.analysis_data_type],
                                   self.data_raw[self.analysis_data_type],
                                   flow_index=flow_index)

        # Sets the x-label to proper units
        x_label_old = self.x_label
        self.x_label = r"$t_e[fm]$"

        # print plateau_limits
        plateau_limits = [0.3, 0.6]

        a, a_err, meff, meff_err, t0, t0_err = \
            self._plateau_fit(plateau_limits)

        # Propagates error a
        a_squared = a**2 / t0
        a_squared_err = np.sqrt((2*a*a_err/t0)**2
                                + (a**2*t0_err/t0**2)**2)

        if np.any(meff == np.nan) or np.any(meff_err == np.nan):
            print "nan value discovered in:", meff, meff_err
            self.skipped_plateau_plot = True
            return

        if len(meff) <= 2:
            print "Too few values for continuum fit:", len(meff)
            self.skipped_plateau_plot = True
            return

        # Continuum limit arrays
        N_cont = 1000
        a_squared_cont = np.linspace(-0.025, a_squared[-1]*1.1, N_cont)

        # Performs a continuum extrapolation of the effective mass
        continuum_fit = LineFit(a_squared, meff, meff_err)
        y_cont, y_cont_err, fit_params, chi_squared = \
            continuum_fit.fit_weighted(a_squared_cont)
        self.cont_chi_squared = chi_squared
        self.cont_fit_params = fit_params

        # Gets the continuum value and its error
        y0_cont, y0_cont_err, _, _, = \
            continuum_fit.fit_weighted(0.0)

        # Matplotlib requires 2 point to plot error bars at
        a0_squared = [0, 0]
        y0 = [y0_cont[0], y0_cont[0]]
        y0_err = [y0_cont_err[0][0], y0_cont_err[1][0]]

        # Stores the chi continuum
        self.meff_cont = y0[0]
        self.meff_cont_err = (y0_err[1] - y0_err[0])/2.0

        # Prepares plotting
        y0_err = [self.meff_cont_err, self.meff_cont_err]

        # if self.verbose:
        #     print "The effective mass for {} is: {}".format(
        #         self.meff_plot_type,
        #         sciprint(self.meff_cont, self.meff_cont_err, prec=3))
        #     print "Chi^2: {}".format(self.cont_chi_squared)

        if self.meff_plot_type != "ma":
            return

        # Creates figure and plot window
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plots an ax-line at 0
        ax.axvline(0, linestyle="dashed",
                   color=self.cont_axvline_color, linewidth=0.5)

        # Plots the fit
        ax.plot(a_squared_cont, y_cont, color=self.fit_color, alpha=0.5)
        ax.fill_between(a_squared_cont, y_cont_err[0], y_cont_err[1],
                        alpha=0.5, edgecolor='',
                        label=r"$\chi^2/\mathrm{d.o.f.}=%.2f$" % chi_squared,
                        facecolor=self.fit_fill_color)

        # Plot lattice points
        ax.errorbar(a_squared, meff, xerr=a_squared_err, yerr=meff_err,
                    fmt="o", capsize=5, capthick=1,
                    color=self.lattice_points_color,
                    ecolor=self.lattice_points_color)

        # plots continuum limit, 5 is a good value for cap size
        ax.errorbar(a0_squared, y0,
                    yerr=y0_err, fmt="o", capsize=5,
                    capthick=1, color=self.cont_error_color,
                    ecolor=self.cont_error_color,
                    label=r"$m_\mathrm{eff}=%.3f\pm%.3f$" % (
                        self.meff_cont, self.meff_cont_err))

        ax.set_ylabel(self.y_label_continuum)
        ax.set_xlabel(self.x_label_continuum)
        ax.set_xlim(a_squared_cont[0], a_squared_cont[-1])
        ax.set_ylim(0, 3)
        ax.legend()
        ax.grid(True)
        # Saves figure
        fname = os.path.join(
            self.output_folder_path,
            "post_analysis_%s_%s_%s.pdf" % (
                self.observable_name_compact, self.analysis_data_type,
                self.meff_plot_type))
        fig.savefig(fname)
        if self.verbose:
            print "Continuum plot of %s created in %s" % (
                self.observable_name_compact, fname)

        plt.close(fig)
        self.skipped_plateau_plot = False
        # Plot an overlay?

    def _get_article_values(self):
        """Retrieves the article values for the set effetive mass type."""
        # https://arxiv.org/pdf/hep-lat/0510074.pdf
        gb1_Mr0 = 6.25
        gb1_Mr0_error = 0.06
        gb1_Mr0_syserror = 0.06
        gb1_M = 2.560
        gb1_M_error = 0.035
        gb1_M_syserror = 0.120
        gb1_label = r"Chen et al."
        gb1_color = "#ff7f00"
        gb1_ls = "--"

        # https://arxiv.org/pdf/1409.6459.pdf
        gb2_M = 2.563
        gb2_M_error = 0.034
        gb2_label = r"Chowdhury et al."
        gb2_color = "#ffff33"
        gb2_ls = "-."

        # https://arxiv.org/pdf/hep-lat/9901004.pdf
        gb3_M = 2.590
        gb3_M_error = 0.040
        gb3_M_syserror = 0.130
        gb3_Mr0 = 6.33
        gb3_Mr0_error = 0.07
        gb3_Mr0_syserror = 0.06
        gb3_label = r"Morningstar et al."
        gb3_color = "#a65628"
        gb3_ls = ":"

        eff_masses = []
        if self.meff_plot_type == "m":
            print "No data for 'm'. Continuing."

        elif self.meff_plot_type == "ma":
            eff_masses.append({
                "mass": gb1_M,
                "mass_error": gb1_M_error,
                "label": gb1_label,
                "color": gb1_color,
                "ls": gb1_ls,
            })
            eff_masses.append({
                "mass": gb2_M,
                "mass_error": gb2_M_error,
                "label": gb2_label,
                "color": gb2_color,
                "ls": gb2_ls,
            })
            eff_masses.append({
                "mass": gb3_M,
                "mass_error": gb3_M_error,
                "label": gb3_label,
                "color": gb3_color,
                "ls": gb3_ls,
            })

        elif self.meff_plot_type == "r0ma":
            eff_masses.append({
                "mass": gb1_Mr0,
                "mass_error": gb1_Mr0_error,
                "label": gb1_label,
                "color": gb1_color,
                "ls": gb1_ls,
            })
            eff_masses.append({
                "mass": gb3_Mr0,
                "mass_error": gb3_Mr0_error,
                "label": gb3_label,
                "color": gb3_color,
                "ls": gb3_ls,
            })

        return eff_masses

    def plot_with_article_masses(self, **kwargs):
        """Plots the effective mass together with different masses from other 
        papers.

        The gluon mass goes through the A_1^{-+} channel
        """

        xlimits = [-0.1, 0.8]

        kwargs["plot_with_formula"] = True
        kwargs["error_shape"] = "bars"
        kwargs["x_label"] = self.x_label

        kwargs["y_label"] = self.meff_labels[self.meff_plot_type]
        kwargs["y_limits"] = self.meff_y_limits[self.meff_plot_type]

        kwargs["figure_name_appendix"] = "_" + self.meff_plot_type
        kwargs["legend_position"] = "best"
        kwargs["x_limits"] = xlimits
        kwargs["return_axes"] = True

        fig, ax = super(QtQ0EffectiveMassPostAnalysis, self)._plot_core(
            self.plot_values, **kwargs)

        eff_masses = self._get_article_values()
        if self.meff_plot_type == "m":
            print "No data for 'm'. Continuing."
            return
        elif self.meff_plot_type == "ma":
            ylimits = [-0.5, 4.0]
        elif self.meff_plot_type == "r0ma":
            ylimits = [-1, 10]

        # fig, ax = plt.subplots()
        for mass_dict in eff_masses:
            x = np.linspace(xlimits[0], xlimits[1], self.num_overlay_points)
            y = np.ones(self.num_overlay_points)*mass_dict["mass"]
            y_err = np.ones(self.num_overlay_points)*mass_dict["mass_error"]
            ax.plot(x, y, mass_dict["ls"], label=mass_dict["label"],
                    color=mass_dict["color"])
            ax.fill_between(x, y - y_err, y + y_err, alpha=0.5,
                            edgecolor="", facecolor=mass_dict["color"])

        # Sets axes limits
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)

        plt.legend(loc="upper left", prop={"size": 7})

        # Saves and closes figure
        kwargs["figure_name_appendix"] += "_overlay"
        fname = self._get_plot_figure_name(
            output_folder=None,
            figure_name_appendix=kwargs["figure_name_appendix"])
        plt.savefig(fname)
        if self.verbose:
            print "Figure saved in %s" % fname

        plt.close(fig)

    def plot_series(self, indexes, x_limits=None,
                    y_limits=None, plot_with_formula=False,
                    error_shape="band"):
        """
        Method for plotting 4 axes together.

        Args:
                indexes: list containing integers of which intervals to plot
                        together.
                x_limits: limits of the x-axis. Default is False.
                y_limits: limits of the y-axis. Default is False.
                plot_with_formula: bool, default is false, is True will look for
                        formula for the y-value to plot in title.
                error_shape: plot with error bands or with error bars.
                        Options: band, bars
        """

        for meff_plot_type in self.meff_plot_types:

            self.meff_plot_type = meff_plot_type

            self.plot_values = {}
            self._initiate_plot_values(self.data[self.analysis_data_type],
                                       self.data_raw[self.analysis_data_type])

            y_limits = self.meff_y_limits[self.meff_plot_type]
            _tmp_ylabel = self.y_label
            self.y_label = "%s%s" % (
                self.meff_labels[self.meff_plot_type],
                self.meff_unit_labels[self.meff_plot_type])

            _tmp_sub_values = sorted(self.observable_intervals.values()[0])
            sub_titles = [
                r"$\sqrt{8t_f}=%.2f$" % (
                    self._convert_label(_tsv))
                for _tsv in _tmp_sub_values]

            overlay_masses = self._get_article_values()

            self._series_plot_core(indexes, x_limits=[-0.05, 1.0],
                                   y_limits=y_limits,
                                   plot_with_formula=plot_with_formula,
                                   error_shape=error_shape,
                                   filename_addendum="_" + self.meff_plot_type,
                                   legend_loc="upper right",
                                   legend_size=8,
                                   use_common_legend=True,
                                   plot_overlay=overlay_masses,
                                   sub_titles=sub_titles)

            self.y_label = _tmp_ylabel


def main():
    exit(("Exit: QtQ0EffectiveMassPostAnalysis not intended to be a "
          "standalone module."))


if __name__ == '__main__':
    main()
