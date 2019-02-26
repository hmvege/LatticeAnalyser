#!/usr/bin/env python2

from default_analysis_params import get_default_parameters
import copy
import os
import sys
import re
import copy as cp
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
    from matplotlib import rc, rcParams

rc("text", usetex=True)
rcParams["font.family"] += ["serif"]


try:
    from pre_analysis.pre_analyser import pre_analysis
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder, \
        load_observable, check_relative_path
except ImportError:
    sys.path.insert(0, "../")
    from pre_analysis.pre_analyser import pre_analysis
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder, \
        load_observable, check_relative_path


def thermalization_analysis():
    """Runs the thermalization analysis."""

    verbose = True
    run_pre_analysis = True
    mark_every = 50
    mc_cutoff = -1 # Skip every 100 points with 2000 therm-steps!!
    batch_folder = check_relative_path("data/thermalization_data")
    base_figure_folder = check_relative_path("figures/")
    base_figure_folder = os.path.join(base_figure_folder,
                                      "thermalization_analysis")
    check_folder(base_figure_folder, verbose=verbose)

    default_params = get_default_parameters(
        data_batch_folder="temp", include_euclidean_time_obs=False)

    ############ COLD START #############
    cold_beta60_params = copy.deepcopy(default_params)
    cold_beta60_params["batch_folder"] = batch_folder
    cold_beta60_params["batch_name"] = "B60_THERM_COLD"
    cold_beta60_params["load_binary_file"] = False
    cold_beta60_params["beta"] = 6.0
    cold_beta60_params["topc_y_limits"] = [-2, 2]
    cold_beta60_params["num_bins_per_int"] = 32
    cold_beta60_params["bin_range"] = [-2.5, 2.5]
    cold_beta60_params["hist_flow_times"] = [0, 250, 600]
    cold_beta60_params["NCfgs"] = get_num_observables(
        cold_beta60_params["batch_folder"],
        cold_beta60_params["batch_name"])
    cold_beta60_params["obs_file"] = "8_6.00"
    cold_beta60_params["N"] = 8
    cold_beta60_params["NT"] = 16
    cold_beta60_params["color"] = "#377eb8"

    ########## HOT RND START ############
    hot_rnd_beta60_params = copy.deepcopy(default_params)
    hot_rnd_beta60_params["batch_folder"] = batch_folder
    hot_rnd_beta60_params["batch_name"] = "B60_THERM_HOT_RND"

    ########## HOT RST START ############
    hot_rst_beta60_params = copy.deepcopy(default_params)
    hot_rst_beta60_params["batch_folder"] = batch_folder
    hot_rst_beta60_params["batch_name"] = "B60_THERM_HOT_RST"

    if run_pre_analysis:
        # Submitting distribution analysis
        cold_data = load_observable(cold_beta60_params)
        hot_rnd_data = load_observable(hot_rnd_beta60_params)
        hot_rst_data = load_observable(hot_rst_beta60_params)

    # # Loads post analysis data
    # cold_data = post_analysis.PostAnalysisDataReader(
    #     [cold_beta60_params],
    #     observables_to_load=cold_beta60_params["observables"],
    #     verbose=verbose)

    # hot_rnd_data = post_analysis.PostAnalysisDataReader(
    #     [hot_rnd_beta60_params],
    #     observables_to_load=hot_rnd_beta60_params["observables"],
    #     verbose=verbose)

    # hot_rst_data = post_analysis.PostAnalysisDataReader(
    #     [hot_rst_beta60_params],
    #     observables_to_load=hot_rst_beta60_params["observables"],
    #     verbose=verbose)

    # TODO: plot termaliations for the 3 different observables

    plot_types = ["default", "loglog", "logx", "logy"]

    y_labels = [
        [r"$P$", r"$Q$", r"$E$"],
        [r"$\frac{|P - \langle P \rangle|}{\langle P \rangle}$", 
            r"$\frac{|Q - \langle Q \rangle|}{\langle Q \rangle}$",
            r"$\frac{|E - \langle E \rangle|}{\langle E \rangle}$"],
        [r"$|P - \langle P \rangle|$", r"$|Q - \langle Q \rangle|$",
            r"$|E - \langle E \rangle|$"]]
    # y_labels[i_dr] = [r"$\langle P \rangle$", r"$\langle P \rangle$",
    #             r"$\langle P \rangle$"]

    subplot_rows = [1, 3]

    # Limits to be put on plot
    x_limits = [[] for i in range(3)]
    y_limits = [[], [], []]

    data_representations = ["default", "relerr", "abserr"]

    obs_list = cold_data["obs"].keys()

    x_label = r"$t_\mathrm{MC}$"

    for i_dr, dr in enumerate(data_representations):
        for pt in plot_types:
            for i_obs, obs in enumerate(obs_list):
                for plot_rows in subplot_rows:

                    # Sets up figure folder for observable
                    figure_folder = os.path.join(base_figure_folder, obs)
                    check_folder(figure_folder, verbose=verbose)

                    # Sets up plot type folder 
                    figure_folder = os.path.join(figure_folder, pt)
                    check_folder(figure_folder, verbose=verbose)

                    if obs == "energy":
                        correction_factor = - 1.0 / 64
                        cold_data["obs"][obs] *= correction_factor
                        hot_rnd_data["obs"][obs] *= correction_factor
                        hot_rst_data["obs"][obs] *= correction_factor

                    # Retrieves data and makes modifications
                    _cold_data = modify_data(
                        cold_data["obs"][obs][:mc_cutoff], dr)
                    _hot_rnd_data = modify_data(
                        hot_rnd_data["obs"][obs][:mc_cutoff], dr)
                    _hot_rst_data = modify_data(
                        hot_rst_data["obs"][obs][:mc_cutoff], dr)

                    # Creates figure name
                    figure_name = "{0:s}_{1:s}_{2:s}_{3:d}plotrows.pdf".format(
                        obs, pt, dr, plot_rows)

                    plot_data_array([np.arange(_cold_data.shape[0])
                                     for i in range(3)],
                                    [_cold_data, _hot_rnd_data,
                                        _hot_rst_data],
                                    ["Cold start", "Hot start",
                                        r"Hot start, $RST$"],
                                    x_label,
                                    y_labels[i_dr][i_obs],
                                    figure_name,
                                    figure_folder,
                                    plot_type=pt,
                                    x_limits=x_limits[i_obs],
                                    y_limits=y_limits[i_obs],
                                    mark_every=mark_every,
                                    subplot_rows=plot_rows)


def check_relative_path(p):
    """
    Function for ensuring are in the correct relative path.

    Args:
        p: str, path we are checking

    Return:
        relative path to p
    """
    if not os.path.isdir(p):
        p = os.path.join("..", p)
        check_relative_path(p)
    return p


def modify_data(data, data_rep):
    """
    Modifies data according to three possible methods,
        'default': no modification
        'relerr': takes |sample_avg - sample_i|
        'abserr': takes |sample_avg - sample_i|/|sample_avg|
    """
    if data_rep == "default":
        return data
    elif data_rep == "relerr":
        return data - np.mean(data)
    elif data_rep == "abserr":
        _mean = np.mean(data)
        return (data - _mean) / np.abs(_mean)
    else:
        raise NotImplementedError("Data modification {} not "
                                  "available.".format(data_rep))


def plot_data_array(data_x, data_y, data_labels, x_label, _y_label,
                    figname, figfolder, x_limits=[], y_limits=[],
                    plot_type="default", mark_every=1, subplot_rows=1):
    """
    Plots the termalization.
    """

    assert len(data_x) == len(data_y) == len(data_labels), (
        "data_x, data_y and data_labels is not of equal length: "
        "{} {} {}".format(len(data_x), len(data_y), len(data_labels)))

    marker = "-"

    fig, axes = plt.subplots(nrows=subplot_rows, sharex=True, sharey=True)

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

        # In order to keep track of axes
        _indices = list([0 for i in range(len(data_x))])
    else:
        _indices = list(range(len(data_x)))

    for i, x, y, labels in zip(_indices, data_x, data_y, data_labels):
        if plot_type == "default":
            axes[i].plot(x, y, marker, label=labels, markevery=mark_every)
        elif plot_type == "loglog":
            axes[i].loglog(x, np.abs(y), marker, label=labels, markevery=mark_every)
        elif plot_type == "logx":
            axes[i].semilogx(x, y, marker, label=labels, markevery=mark_every)
        elif plot_type == "logy":
            axes[i].semilogy(x, np.abs(y), marker, label=labels, markevery=mark_every)
        else:
            raise KeyError("plot_type key {} not "
                           "recognized".format(plot_type))

        if len(x_limits) > 0:
            axes[i].set_xlim(x_limits)
        if len(y_limits) > 0:
            axes[i].set_ylim(y_limits)

        axes[i].grid(True)
        axes[i].legend()

    fig.text(0.5, 0.04, x_label, ha='center')
    fig.text(0.04, 0.5, _y_label, va='center', rotation='vertical')

    figpath = os.path.join(figfolder, figname)
    plt.savefig(figpath)
    print "Figure saved at path {}".format(figpath)

    plt.close(fig)


if __name__ == '__main__':
    thermalization_analysis()
