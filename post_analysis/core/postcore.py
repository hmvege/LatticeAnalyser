from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import types
import os
import numpy as np
from tools.postanalysisdatareader import PostAnalysisDataReader
from tools.latticefunctions import get_lattice_spacing
from tools.folderreadingtools import check_folder, get_NBoots
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

# For zooming in on particular part of plot

rc("text", usetex=True)
# rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rcParams["font.family"] += ["serif"]


class PostCore(object):
    """Post analysis base class."""
    observable_name = "Observable"
    observable_name_compact = "obs"
    obs_name_latex = "MISSING LATEX NAME FOR OBSERVABLE"
    formula = ""
    x_label = r""
    y_label = r""
    section_seperator = "="*160
    dpi = 350
    r0 = 0.5
    print_latex = False
    sub_obs = False
    sub_sub_obs = False
    interval = []

    # Color setup for continuum plots
    cont_error_color = "#0c2c84" # For the single continuum point
    cont_axvline_color = "#000000"
    fit_color = "#225ea8"
    fit_fill_color = "#225ea8"
    lattice_points_color = "#000000"


    def __init__(self, data, with_autocorr=True, figures_folder="../figures",
                 verbose=False, dryrun=False):
        """
        Base class for analysing beta values together after initial analysis.

        Args:
                data: PostAnalysisDataReader object, contains all of the 
                        observable data.
                with_autocorr: bool, optional. Will perform analysis on data
                        corrected by autocorrelation sqrt(2*tau_int). Default is True.
                figures_folder: str, optional. Default output folder is ../figures.
                verbose: bool, optional. A more verbose output. Default is False.
                dryrun: bool, optional. No major changes will be performed. 
                        Default is False.
        """

        if with_autocorr:
            self.ac = "with_autocorr"
        else:
            self.ac = "without_autocorr"

        self.with_autocorr = with_autocorr
        self.reference_values = data.reference_values
        observable = self.observable_name_compact

        self.verbose = verbose
        self.dryrun = dryrun

        self.beta_values = sorted(data.beta_values)
        self.colors = data.colors
        self.lattice_sizes = data.lattice_sizes
        self.size_labels = data.labels
        self._setup_analysis_types(data.analysis_types)
        self.print_latex = data.print_latex

        self.data = {atype: {beta: {} for beta in self.beta_values}
                     for atype in self.analysis_types}

        self.flow_epsilon = {b: data.flow_epsilon[b] for b in self.beta_values}

        # Only sets this variable if we have sub-intervals in order to avoid bugs.
        if self.sub_obs:
            self.observable_intervals = {b: {} for b in self.beta_values}
            # for beta in self.beta_values:
            # 	self.observable_intervals[beta] = {}

        # Checks that the observable is among the available data
        assert_msg = ("%s is not among current data(%s). Have the pre analysis"
                      " been performed?" % (observable, ", ".join(data.observable_list)))
        assert observable in data.observable_list, assert_msg

        for atype in self.analysis_types:
            for beta in self.beta_values:
                _tmp = data.data_observables[observable][beta]
                if self.sub_obs:
                    if self.sub_sub_obs:
                        for subobs in _tmp:
                            # Sets sub-sub intervals
                            self.observable_intervals[beta][subobs] = \
                                _tmp[subobs].keys(
                            )

                            # Sets up additional subsub-dictionaries
                            self.data[atype][beta][subobs] = {}

                            for subsubobs in _tmp[subobs]:

                                self.data[atype][beta][subobs][subsubobs] = \
                                    _tmp[subobs][subsubobs][self.ac][atype]

                                if self.with_autocorr:
                                    self.data[atype][beta][subobs][subsubobs]["ac"] = \
                                        _tmp[subobs][subsubobs]["with_autocorr"]["autocorr"]

                    else:
                        # Fills up observable intervals
                        self.observable_intervals[beta] = \
                            _tmp.keys()

                        for subobs in _tmp:

                            self.data[atype][beta][subobs] = \
                                _tmp[subobs][self.ac][atype]

                            if self.with_autocorr:
                                self.data[atype][beta][subobs]["ac"] = \
                                    _tmp[subobs]["with_autocorr"]["autocorr"]

                else:
                    self.data[atype][beta] = _tmp[self.ac][atype]

                    if self.with_autocorr:
                        self.data[atype][beta]["ac"] = \
                            _tmp["with_autocorr"]["autocorr"]

        self.data_raw = {}
        for atype in data.raw_analysis:
            if atype == "autocorrelation":
                self.ac_raw = data.raw_analysis[atype]
            else:
                self.data_raw[atype] = data.raw_analysis[atype]

        # Small test to ensure that the number of bootstraps and number of
        # different beta batches match
        err_msg = ("Number of bootstraps do not match number "
                   "of different beta values")
        assert np.asarray([get_NBoots(self.data_raw["bootstrap"][i])
                           for i in self.data_raw["bootstrap"].keys()]).all(), err_msg

        self.NBoots = get_NBoots(self.data_raw["bootstrap"])

        # Creates base output folder for post analysis figures
        self.figures_folder = figures_folder
        check_folder(self.figures_folder, dryrun=self.dryrun,
                     verbose=self.verbose)
        check_folder(os.path.join(self.figures_folder, data.batch_name),
                     dryrun=self.dryrun, verbose=self.verbose)

        # Creates output folder
        self.post_anlaysis_folder = os.path.join(self.figures_folder,
                                                 data.batch_name, "post_analysis")
        check_folder(self.post_anlaysis_folder, dryrun=self.dryrun,
                     verbose=self.verbose)

        # Creates observable output folder
        self.output_folder_path = os.path.join(self.post_anlaysis_folder,
                                               self.observable_name_compact)
        check_folder(self.output_folder_path, dryrun=self.dryrun,
                     verbose=self.verbose)

    def _setup_analysis_types(self, atypes):
        """
        Stores the number of analysis types from the data container, while 
        removing the autocorrelation one.
        """
        self.analysis_types = atypes
        if "autocorrelation" in self.analysis_types:
            self.analysis_types.remove("autocorrelation")

    def _check_plot_values(self):
        """Checks if we have set the analysis data type yet."""
        if not hasattr(self, "plot_values"):
            raise AttributeError(
                "set_analysis_data_type() has not been set yet.")

    def set_analysis_data_type(self, analysis_data_type="bootstrap"):
        """Sets the analysis type and retrieves correct analysis data."""

        # Makes it a global constant so it can be added in plot figure name
        self.analysis_data_type = analysis_data_type

        self.plot_values = {}  # Clears old plot values
        self._initiate_plot_values(self.data[analysis_data_type],
                                   self.data_raw[analysis_data_type])

    def print_estimates(self, flow_times=[]):
        """
        Prints the topsus values given the flow time.
        """
        for bval, tf0 in zip(self.beta_values, flow_times):
            print self.beta_values[bval]["y"][tf0]

    def _initiate_plot_values(self, data, data_raw):
        """Sorts data into a format specific for the plotting method."""
        for beta in sorted(data):
            values = {}
            values["a"], values["a_err"] = get_lattice_spacing(beta)
            values["sqrt8t"] = values["a"]*np.sqrt(8*data[beta]["x"])
            values["x"] = values["a"] * np.sqrt(8*data[beta]["x"])
            values["y"] = data[beta]["y"]
            values["y_err"] = data[beta]["y_error"]
            values["y_raw"] = data_raw[beta][self.observable_name_compact]
            values["y_uraw"] = self.data_raw["unanalyzed"][beta][self.observable_name_compact]

            # print values["y_raw"].shape, values["y_uraw"].shape
            # print "%s beta: %f" % (self.observable_name_compact, beta)
            # print np.mean(values["y_raw"][-1,:]), np.std(values["y_raw"][-1,:])
            # print np.mean(values["y_uraw"][-1,:]), np.std(values["y_uraw"][-1,:])

            if self.with_autocorr:
                values["tau_int"] = data[beta]["ac"]["tau_int"]
                values["tau_int_err"] = data[beta]["ac"]["tau_int_err"]
            else:
                values["tau_int"] = None
                values["tau_int_err"] = None
            values["label"] = r"%s $\beta=%2.2f$" % (
                self.size_labels[beta], beta)
            values["color"] = self.colors[beta]

            self.plot_values[beta] = values

    def plot(self, x_limits=False, y_limits=False, plot_with_formula=False,
             error_shape="band", figure_folder=None, plot_vline_at=None,
             plot_hline_at=None, figure_name_appendix="", show_plot=False,
             zoom_box=None):
        """
        Function for making a basic plot of all the different beta values
        together.

        Args:
                x_limits: limits of the x-axis. Default is False.
                y_limits: limits of the y-axis. Default is False.
                plot_with_formula: bool, default is false, is True will look for 
                        formula for the y-value to plot in title.
                figure_folder: optional, default is None. If default, will place
                        figures in figures/{batch_name}/post_analysis/{observable_name}
                plot_vline_at: optional, float. If present, will plot a vline at 
                        position given position.
                plot_hline_at: optional, float. If present, will plot a hline at 
                        position given position.
                figure_name_appendix: optional, str, adds provided string to 
                        filename. Default is adding nothing.
                show_plot: optional, bool, will show plot figure.
                zoom_box, optional, nested list of floats, will create a zoomed 
                        in subplot in figure at location [[xmin, xmax], [ymin, ymax]].
        """

        self._plot_core(self.plot_values, x_label=self.x_label,
                        y_label=self.y_label, x_limits=x_limits, y_limits=y_limits,
                        plot_with_formula=plot_with_formula, error_shape=error_shape,
                        figure_folder=figure_folder, plot_vline_at=plot_vline_at,
                        plot_hline_at=plot_hline_at,
                        figure_name_appendix=figure_name_appendix,
                        show_plot=show_plot, zoom_box=zoom_box)

    def plot_autocorrelation(self, x_limits=False, y_limits=False,
                             figure_folder=None, plot_vline_at=None, plot_hline_at=None,
                             show_plot=False):
        """
        Method for plotting the autocorrelations in a single window.
        """
        if self.sub_obs or self.sub_sub_obs:
            print ("Skipping AC-plot for %s due to containing "
                   "subobs.".format(self.observable_name))
            return
            
        if self.verbose:
            print "Plotting %s autocorrelation for betas %s together" % (
                self.observable_name_compact,
                ", ".join([str(b) for b in self.plot_values]))

        fig = plt.figure(dpi=self.dpi)
        ax = fig.add_subplot(111)

        self._check_plot_values()

        # Retrieves values to plot
        for beta in sorted(self.plot_values):
            value = self.plot_values[beta]
            x = value["sqrt8t"]
            y = value["tau_int"]
            y_err = value["tau_int_err"]
            ax.plot(x, y, "-", label=value["label"], color=self.colors[beta])
            ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, edgecolor="",
                            facecolor=self.colors[beta])

        # Basic plotting commands
        ax.grid(True)
        ax.set_xlabel(r"$\sqrt{8 t_f}$")
        ax.set_ylabel(r"$\tau_\mathrm{int}$")
        ax.legend(loc="lower right", prop={"size": 8})

        # Sets axes limits if provided
        if x_limits != False:
            ax.set_xlim(x_limits)
        if y_limits != False:
            ax.set_ylim(y_limits)

        # Plots a vertical line at position "plot_vline_at"
        if not isinstance(plot_vline_at, types.NoneType):
            ax.axvline(plot_vline_at, linestyle="--", color="0", alpha=0.3)

        # Plots a horizontal line at position "plot_hline_at"
        if not isinstance(plot_hline_at, types.NoneType):
            ax.axhline(plot_hline_at, linestyle="--", color="0", alpha=0.3)

        # Saves and closes figure
        fname = self._get_plot_figure_name(output_folder=figure_folder,
                                           figure_name_appendix="_autocorr")
        plt.savefig(fname)
        if self.verbose:
            print "Figure saved in %s" % fname

        if show_plot:
            plt.show()

        plt.close(fig)

    def plot_autocorrelation_at(self, t_flow=0):
        """
        Plots autocorrelation at a given flow time t_flow.
        """
        pass

    def plot_mc_history_at(self, t_flow=0):
        """
        Plots the Monte-Carlo history at a given flow time t_flow.
        """
        pass

    def _get_plot_figure_name(self, output_folder=None,
                              figure_name_appendix=""):
        """Retrieves appropriate figure file name."""
        if isinstance(output_folder, types.NoneType):
            output_folder = self.output_folder_path
        fname = "post_analysis_%s_%s%s.pdf" % (self.observable_name_compact,
                                               self.analysis_data_type, figure_name_appendix)
        return os.path.join(output_folder, fname)

    def get_values(self, tf, atype, extrap_method=None):
        """
        Method for retrieving values a given flow time t_f.

        Args:
                tf: float or str. (float) flow time at a given t_f/a^2. If 
                        string is "t0", will return the flow time at reference value
                        t0/a^2 from t^2<E>=0.3 for a given beta. "tfbeta" will use the
                        t0 value for each particular beta value.
                atype: str, type of analysis we have performed.
                extrap_method: str, type of extrapolation technique used. If None,
                        will use method which is present.

        Returns:
                {beta: {t0, y0, y0_error}
        """

        # Checks that the extrapolation method exists
        if isinstance(extrap_method, types.NoneType):
            # If module has the extrapolation method, but it is not set, the
            # one last used will be the method of choice.
            if hasattr(self, "extrapolation_method"):
                extrap_method = self.extrapolation_method

        values = {beta: {} for beta in self.beta_values}
        self.t0 = {beta: {} for beta in self.beta_values}

        self._get_tf_value(tf, atype, extrap_method)

        for beta in self.beta_values:
            a = self.plot_values[beta]["a"]

            # Selects index closest to q0_flow_time
            tf_index = np.argmin(
                np.abs(self.plot_values[beta]["x"] - self.t0[beta]))

            values[beta]["t0"] = self.t0[beta]
            values[beta]["y0"] = self.plot_values[beta]["y"][tf_index]
            values[beta]["y_err0"] = self.plot_values[beta]["y_err"][tf_index]
            values[beta]["tau_int0"] = self.plot_values[beta]["tau_int"][tf_index]
            values[beta]["tau_int_err0"] = self.plot_values[beta]["tau_int_err"][tf_index]

        return_dict = {"obs": self.observable_name_compact, "data": values}
        return return_dict

    def _plot_core(self, plot_values, observable_name_compact=None,
                   x_label="x", y_label="y", x_limits=False, y_limits=False,
                   plot_with_formula=False, error_shape="band", figure_folder=None,
                   plot_vline_at=None, plot_hline_at=None, figure_name_appendix="",
                   show_plot=False, zoom_box=None):
        """
        Function for making a basic plot of all the different beta values
        together.

        Args:
                plot_values: from _initiate_plot_values().
                x_label: str, x label.
                x_label: str, y label.
                x_limits: limits of the x-axis. Default is False.
                y_limits: limits of the y-axis. Default is False.
                plot_with_formula: bool, default is false, is True will look for 
                        formula for the y-value to plot in title.
                figure_folder: optional, default is None. If default, will place
                        figures in figures/{batch_name}/post_analysis/{observable_name}
                plot_vline_at: optional, float. If present, will plot a vline at 
                        position given position.
                plot_hline_at: optional, float. If present, will plot a hline at 
                        position given position.
                figure_name_appendix: optional, str, adds provided string to 
                        filename. Default is adding nothing.
                show_plot: optional, bool, will show plot figure.
                zoom_box, optional, nested list of floats, will create a zoomed 
                        in subplot in figure at location [[xmin, xmax], [ymin, ymax]].
        """

        if type(observable_name_compact) == type(None):
            observable_name_compact = self.observable_name_compact

        if self.verbose:
            print "Plotting %s for betas %s together" % (
                observable_name_compact,
                ", ".join([str(b) for b in plot_values]))

        fig = plt.figure(dpi=self.dpi)
        ax = fig.add_subplot(111)

        self._check_plot_values()

        if not isinstance(zoom_box, type(None)):
            # 2.5: zoom factor, loc=2: upper left
            axins = zoomed_inset_axes(ax, zoom_box["zoom_factor"], loc=2)

        # Retrieves values to plot
        for beta in sorted(plot_values):
            value = plot_values[beta]
            x = value["x"]
            y = value["y"]
            y_err = value["y_err"]
            if error_shape == "band":
                ax.plot(
                    x, y, "-", label=value["label"], color=self.colors[beta])
                ax.fill_between(x, y - y_err, y + y_err, alpha=0.5,
                                edgecolor="", facecolor=self.colors[beta])
            elif error_shape == "bars":
                ax.errorbar(x, y, yerr=y_err, capsize=5, fmt="_", ls=":",
                            label=value["label"], color=self.colors[beta],
                            ecolor=self.colors[beta])
            else:
                raise KeyError("%s not a recognized plot type" % error_shape)

            if not isinstance(zoom_box, type(None)):
                if error_shape == "band":
                    axins.plot(x, y, "-", label=value["label"],
                               color=self.colors[beta])
                    axins.fill_between(x, y - y_err, y + y_err, alpha=0.5,
                                       edgecolor="", facecolor=self.colors[beta])
                elif error_shape == "bars":
                    axins.errorbar(x, y, yerr=y_err, capsize=5, fmt="_", ls=":",
                                   label=value["label"], color=self.colors[beta],
                                   ecolor=self.colors[beta])

        if not isinstance(zoom_box, type(None)):
            axins.set_xlim(zoom_box["xlim"])
            axins.set_ylim(zoom_box["ylim"])
            # axins.grid(True)
            plt.yticks(visible=False)
            plt.xticks(visible=False)

            # Plots a vertical line at position "plot_vline_at"
            if not isinstance(plot_vline_at, types.NoneType):
                axins.axvline(plot_vline_at, linestyle="--",
                              color="0", alpha=0.3)

            # Plots a horizontal line at position "plot_hline_at"
            if not isinstance(plot_hline_at, types.NoneType):
                axins.axhline(plot_hline_at, linestyle="--",
                              color="0", alpha=0.3)

            mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

        # Basic plotting commands
        ax.grid(True)
        # ax.set_title(r"%s" % title_string)
        ax.set_xlabel(r"%s" % x_label)
        ax.set_ylabel(r"%s" % y_label)
        ax.legend(loc="lower right", prop={"size": 8})

        # # Sets the title string
        # title_string = r"%s" % self.observable_name
        # if plot_with_formula:
        # 	title_string += r" %s" % self.formula

        # if self.observable_name_compact == "energy":
        # 	ax.ticklabel_format(style="sci", axis="y", scilimits=(1,10))

        # Sets axes limits if provided
        if x_limits != False:
            ax.set_xlim(x_limits)
        if y_limits != False:
            ax.set_ylim(y_limits)

        # Plots a vertical line at position "plot_vline_at"
        if not isinstance(plot_vline_at, types.NoneType):
            ax.axvline(plot_vline_at, linestyle="--", color="0", alpha=0.3)

        # Plots a horizontal line at position "plot_hline_at"
        if not isinstance(plot_hline_at, types.NoneType):
            ax.axhline(plot_hline_at, linestyle="--", color="0", alpha=0.3)

        # Saves and closes figure
        fname = self._get_plot_figure_name(output_folder=figure_folder,
                                           figure_name_appendix=figure_name_appendix)
        plt.savefig(fname)
        if self.verbose:
            print "Figure saved in %s" % fname

        if show_plot:
            plt.show()

        plt.close(fig)

    def __str__(self):
        """Class string representation method."""
        msg = "\n" + self.section_seperator
        msg += "\nPost analaysis for:        " + self.observable_name_compact
        msg += "\n" + self.__doc__
        msg += "\nAnalysis-type:             " + self.analysis_data_type
        msg += "\nIncluding autocorrelation: " + self.ac
        msg += "\nOutput folder:             " + self.output_folder_path
        msg += "\n" + self.section_seperator
        return msg


def main():
    exit("Exit: PostCore is not intended to be used as a standalone module.")


if __name__ == "__main__":
    main()
