import subprocess
import types
import itertools
from post_analysis.core.multiplotcore import MultiPlotCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import numpy as np
import copy
import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt


class QtQ0EuclideanPostAnalysis(MultiPlotCore):
    """Class for plotting different QteQte0 a specific flow time together."""
    observable_name = ""
    observable_name_compact = "qtq0e"
    x_label = r"$t_e$ [fm]"
    y_label = r"$\langle Q_{t_e} Q_{t_{e,0}} \rangle$"  # $\chi_t^{1/4}[GeV]$
    sub_obs = True
    sub_sub_obs = True
    subfolder_type = "tf"

    def __init__(self, *args, **kwargs):
        super(QtQ0EuclideanPostAnalysis, self).__init__(*args, **kwargs)

    def _get_euclidean_index(self, euclidean_percent, batch_name):
        """Internal method for getting the euclidean index."""
        euclidean_index = int(
            self.lattice_sizes[batch_name][1] * euclidean_percent)
        if euclidean_index != 0:
            euclidean_index -= 1
        return euclidean_index

    def _convert_label(self, lab):
        return float(lab[-6:])

    def _initiate_plot_values(self, data, data_raw, euclidean_percent,
                              q0_flow_time=None):
        """interval_index: int, should be in euclidean time."""

        # Sorts data into a format specific for the plotting method
        for bn in self.sorted_batch_names:
            euclidean_index = self._get_euclidean_index(euclidean_percent, bn)
            te_index = "te%04d" % euclidean_index
            values = {}
            if q0_flow_time == None:
                # Case where we have sub sections of observables,
                # e.g. in euclidean time.
                for sub_obs in self.observable_intervals[bn]:
                    sub_values = {}
                    sub_values["a"], sub_values["a_err"] = \
                        get_lattice_spacing(self.beta_values[bn])
                    sub_values["x"] = np.linspace(0,
                                                  self.lattice_sizes[bn][1] *
                                                  sub_values["a"],
                                                  self.lattice_sizes[bn][1])

                    sub_values["y"] = data[bn][sub_obs][te_index]["y"]
                    sub_values["y_err"] = \
                        data[bn][sub_obs][te_index]["y_error"]

                    sub_values["y_raw"] = \
                        data_raw[bn][self.observable_name_compact][sub_obs]

                    if self.with_autocorr:
                        sub_values["tau_int"] = \
                            data[bn][sub_obs][te_index]["ac"]["tau_int"]
                        sub_values["tau_int_err"] = \
                            data[bn][sub_obs][te_index]["ac"]["tau_int_err"]

                    sub_values["label"] = (
                        r"%s, $\beta=%2.2f$, $\sqrt{8t_{f}}=%.2f$"
                        % (self.size_labels[bn],
                           self.beta_values[bn],
                           self._convert_label(sub_obs)))

                    values[sub_obs] = sub_values
                self.plot_values[bn] = values

            else:
                tf_index = "tflow%04.4f" % q0_flow_time
                values = {}
                values["a"], values["a_err"] = \
                    get_lattice_spacing(self.beta_values[bn])

                # FOR EXACT BOX SIZE:
                values["x"] = np.linspace(0,
                                          self.lattice_sizes[bn][1] *
                                          values["a"],
                                          self.lattice_sizes[bn][1])

                values["y"] = data[bn][tf_index][te_index]["y"]
                values["y_err"] = data[bn][tf_index][te_index]["y_error"]

                values["y_raw"] = \
                    data_raw[bn][self.observable_name_compact][tf_index]

                if self.with_autocorr:
                    values["tau_int"] = \
                        data[bn][tf_index][te_index]["ac"]["tau_int"]
                    values["tau_int_err"] = \
                        data[bn][tf_index][te_index]["ac"]["tau_int_err"]

                values["label"] = (
                    r"%s $\beta=%2.2f$, $\sqrt{8t_f}=%.2f$, $t_{e,0}/a=%d$" % (
                        self.size_labels[bn], self.beta_values[bn],
                        q0_flow_time, euclidean_index))

                self.plot_values[bn] = values

    def set_analysis_data_type(self, euclidean_percent,
                               analysis_data_type="bootstrap"):
        self.plot_values = {}

        # Makes it a global constant so it can be added in plot figure name
        self.analysis_data_type = analysis_data_type

        # Initiates plot values
        self._initiate_plot_values(self.data[self.analysis_data_type],
                                   self.data_raw[self.analysis_data_type],
                                   euclidean_percent=euclidean_percent)

    def _get_plot_figure_name(self, output_folder=None,
                              figure_name_appendix=""):
        """Retrieves appropriate figure file name."""
        if isinstance(output_folder, types.NoneType):
            # Sets up slices folder containing all euclidean times
            output_folder = os.path.join(self.output_folder_path, "slices")
            check_folder(output_folder, False, True)

            # Sets up euclidean time folder
            output_folder = os.path.join(
                output_folder, "te%04d" % (int(100*self.interval_index[1])))

        check_folder(output_folder, False, True)

        fname = "post_analysis_%s_%s_tf%s%s.pdf" % (
                self.observable_name_compact,
                self.analysis_data_type,
                str("%4.4f" % self.interval_index[0]).replace(".", "_"),
                figure_name_appendix)
        return os.path.join(output_folder, fname)

    def plot_interval(self, q0_flow_time, euclidean_percent, **kwargs):
        """
        Sets and plots only one interval.

        Args:
                q0_flow_time: float, flow time.
                euclidean_percent: float, percentage of the euclidean time.
        """
        self.interval_index = [q0_flow_time, euclidean_percent]
        self.plot_values = {}
        self._initiate_plot_values(self.data[self.analysis_data_type],
                                   self.data_raw[self.analysis_data_type],
                                   euclidean_percent=euclidean_percent,
                                   q0_flow_time=q0_flow_time)

        self._center_values(euclidean_percent)

        # Sets the x-label to proper units
        x_label_old = self.x_label
        self.x_label = r"$t_f[fm]$"

        kwargs["legend_position"] = "best"

        # Makes it a global constant so it can be added in plot figure name
        self.plot(**kwargs)

        self.x_label = x_label_old

    def _center_values(self, euclidean_percent, indexes=None):
        """Offsets the x axis to be centered around the correlator."""
        flow_times = sorted(self.observable_intervals.values()[0])
        max_val = []

        if isinstance(indexes, type(None)):
            # In case no indexes has been provided, we are not plotting a
            # series.
            for bn in self.sorted_batch_names:
                max_val.append(self.plot_values[bn]
                               ["x"][self._get_euclidean_index(
                                   euclidean_percent, bn)])
            for j, bn in enumerate(self.sorted_batch_names[:-1]):
                offset = max_val[-1] - max_val[j]
                self.plot_values[bn]["x"] += offset
        else:
            # If indexes are provided we are plotting a series.
            for i in indexes:
                for bn in self.sorted_batch_names:
                    max_val.append(self.plot_values[bn][flow_times[i]]
                                   ["x"][self._get_euclidean_index(
                                       euclidean_percent, bn)])

            for j, bn in enumerate(self.sorted_batch_names[:-1]):
                offset = max_val[-1] - max_val[j]
                for ftime in flow_times:
                    self.plot_values[bn][ftime]["x"] += offset

    def plot_series(self, euclidean_percent, indexes, x_limits=False,
                    y_limits=False, plot_with_formula=False):
        """
        Method for plotting 4 axes together.

        Args:
                indexes: list containing integers of which intervals to 
                        plot together.
                x_limits: limits of the x-axis. Default is False.
                y_limits: limits of the y-axis. Default is False.
                plot_with_formula: bool, default is false, is True will look 
                        for formula for the y-value to plot in title.
        """

        self.plot_values = {}
        self._initiate_plot_values(self.data[self.analysis_data_type],
                                   self.data_raw[self.analysis_data_type],
                                   euclidean_percent=euclidean_percent)

        self._center_values(euclidean_percent, indexes=indexes)

        fname = "post_analysis_%s_%s_te%04d.pdf" % (
                self.observable_name_compact,
                self.analysis_data_type, int(euclidean_percent*100))

        self._series_plot_core(indexes, x_limits=x_limits,
                               y_limits=y_limits, plot_with_formula=False,
                               fname=fname)


def main():
    exit("%s not intendd for standalone usage." % __name__)


if __name__ == '__main__':
    main()
