import statistics.parallel_tools as ptools
import multiprocessing
from statistics.autocorrelation import Autocorrelation
import types
import os
import itertools
from post_analysis.core.multiplotcore import MultiPlotCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

# Temporary needed for assessing the autocorrelation in the eff mass data.


class QtQ0EffectiveMassPostAnalysis(MultiPlotCore):
    """Post-analysis of the effective mass."""
    observable_name = r"Effective mass, "
    observable_name += r"$am_\textrm{eff} = \log \frac{C(t_e)}{C(t_e+1)}$, "
    observable_name += r"$C(t_e)=\langle q_t q_0\rangle$"
    observable_name_compact = "qtq0eff"
    x_label = r"$t_e$ [fm]"
    y_label = r"$r_0 m_\textrm{eff}$"
    sub_obs = True
    hbarc = 0.19732697  # eV micro m
    dpi = 400
    fold = True
    fold_range = 16
    subfolder_type = "tflow"

    meff_plot_type = "ma" # Default
    meff_plot_types = ["ma", "m", "r0ma"]

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
        fname = "post_analysis_%s_%s_tf%f%s.pdf" % (
                self.observable_name_compact, self.analysis_data_type,
                self.interval_index, figure_name_appendix)
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

                    sub_values["label"] = (
                        r"%s, $\beta=%2.2f$, $\sqrt{8t_f}=%.2f$" % (
                            self.size_labels[bn], self.beta_values[bn],
                            self._convert_label(sub_obs)))

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
                        np.linspace(0, (int(values["y"].shape[0]/2))*values["a"],
                                    int(values["y"].shape[0]/2)+1)

                    values["y"], values["y_err"] = self._folder_and_propagate(
                        values)

                    # values["y_raw"] = self.fold_array(values["y_raw"], axis=0)
                    self.fold_position = values["x"][self.fold_range]

                values["label"] = r"%s $\beta=%2.2f$, $\sqrt{8t_f}=%.2f$" % (
                    self.size_labels[bn], self.beta_values[bn], flow_index)

                self.plot_values[bn] = values

    def _folder_and_propagate(self, values):
        """Depending on what we are plotting, 'm/a', 'm', 'r0m/a',
        will set up the correct fold."""

        if self.meff_plot_type == "ma":
            # y / a
            y = self.fold_array(values["y"])/values["a"] * self.hbarc
            y_err = np.sqrt(
                (self.fold_error_array(values["y_err"])*self.hbarc/values["a"])**2 +
                (y/values["a"]**2*values["a_err"]*self.hbarc)**2)

        elif self.meff_plot_type == "m":
            # y
            y = self.fold_array(values["y"])
            y_err = self.fold_error_array(values["y_err"])

        elif self.meff_plot_type == "r0ma":
            # y * r0 / a
            y = \
                self.fold_array(values["y"])*self.r0/values["a"]
            y_err = np.sqrt(
                (self.fold_error_array(values["y_err"])*self.r0/values["a"])**2 +
                (self.r0*y/values["a"]**2*values["a_err"])**2)

        else:
            raise KeyError(("Effective mass plot type '%s' not recognized "
                        "among %s" % (self.meff_plot_type, self.meff_plot_types)))

        return y, y_err

    def plot_interval(self, flow_index, **kwargs):
        """
        Sets and plots only one interval.

        Args:
                flow_index: flow time integer
                euclidean_index: integer for euclidean time
        """
        meff_plot_types = ["ma", "m", "r0ma"]
        for meff_plot_type in meff_plot_types:

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

            self.x_label = x_label_old

    def plot(self, *args, **kwargs):
        """Ensuring I am plotting with formule in title."""
        kwargs["plot_with_formula"] = True
        kwargs["error_shape"] = "bars"
        kwargs["x_label"] = self.x_label
        if self.meff_plot_type == "ma":
            # y / a
            kwargs["y_label"] = r"$m_\mathrm{eff}$[GeV]"
            kwargs["y_limits"] = [-10, 10]  # When plotting Y=M/a
        elif self.meff_plot_type == "m":
            # y
            kwargs["y_label"] = r"$am_\mathrm{eff}$"
            kwargs["y_limits"] = [-2, 2]  # When plotting Y=aM
        elif self.meff_plot_type == "r0ma":
            # y * r0 / a
            kwargs["y_label"] = r"$r_0 m_\mathrm{eff}$"
            kwargs["y_limits"] = [-10, 10]  # When plotting Y=M/a*r0
        else:
            raise KeyError(("Effective mass plot type %s not recognized"
                            % self.meff_plot_type))

        kwargs["figure_name_appendix"] = "_" + self.meff_plot_type

        # if self.fold:
        #   kwargs["plot_vline_at"] = self.fold_position
        # else:
        #   kwargs["x_limits"] = [-0.1,4.7]

        kwargs["x_limits"] = [-0.1, 1]

        super(QtQ0EffectiveMassPostAnalysis, self)._plot_core(
            self.plot_values, **kwargs)

    def plot_plateau(self, plateau_limits):
        """Method for extracting the glueball mass and plot plateau."""
        pass


def main():
    exit(("Exit: QtQ0EffectiveMassPostAnalysis not intended to be a "
          "standalone module."))


if __name__ == '__main__':
    main()
