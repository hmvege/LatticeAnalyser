#!/usr/bin/env python2

from default_analysis_params import get_default_parameters, load_pickle, \
    save_pickle

import copy
import os
import numpy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

try:
    import pre_analysis
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder
except ImportError:
    import sys
    sys.path.insert(0, "../")
    import pre_analysis
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder


def distribution_analysis():
    """Analysis for different SU3 epsilon matrix geeration values."""
    default_params = get_default_parameters(data_batch_folder="temp")

    verbose = True
    use_pickle = False
    pickle_name = "distribution_analysis_data_pickle.pkl"

    ########## Distribution analysis ##########
    dist_eps = [0.05, 0.10, 0.20, 0.24, 0.30, 0.40, 0.60]

    def create_dist_batch_set(default_parameters, eps):
        def clean_str(s): return str("%-2.2f" % s).replace(".", "")
        dist_data_beta60_analysis = copy.deepcopy(default_parameters)

        # Ensuring that the distribution runs folder exist
        dist_data_beta60_analysis["batch_folder"] = (
            "../data/distribution_tests/distribution_runs")
        if not os.path.isdir(dist_data_beta60_analysis["batch_folder"]):
            dist_data_beta60_analysis["batch_folder"] = \
                os.path.join("..", dist_data_beta60_analysis["batch_folder"])

        dist_data_beta60_analysis["batch_name"] = \
            "distribution_test_eps{0:s}".format(clean_str(eps))
        dist_data_beta60_analysis["beta"] = 6.0
        dist_data_beta60_analysis["num_bins_per_int"] = 16
        dist_data_beta60_analysis["bin_range"] = [-2.1, 2.1]
        dist_data_beta60_analysis["hist_flow_times"] = [0, 250, 600]
        dist_data_beta60_analysis["NCfgs"] = get_num_observables(
            dist_data_beta60_analysis["batch_folder"],
            dist_data_beta60_analysis["batch_name"])
        dist_data_beta60_analysis["obs_file"] = "6_6.00"  # 6^3x12, beta=6.0
        dist_data_beta60_analysis["N"] = 6
        dist_data_beta60_analysis["NT"] = 12
        dist_data_beta60_analysis["color"] = "#377eb8"
        return dist_data_beta60_analysis

    dist_param_list = [create_dist_batch_set(default_params, _eps)
                       for _eps in dist_eps]

    # dist_param_list = dist_param_list[:2]

    # exit("not performing the regular pre-analysis.")

    # # Submitting distribution analysis
    # for analysis_parameters in dist_param_list:
    #     pre_analysis.pre_analysis(analysis_parameters)

    # Use post_analysis data for further analysis.
    data = {}
    for eps, param in zip(dist_eps, dist_param_list):
        print "Loading data for eps={0:.2f}".format(eps)
        data[eps] = post_analysis.PostAnalysisDataReader(
            [param], observables_to_load=param["observables"])

    # Plot topc
    distribution_plotter(
        data, "topc", r"$\sqrt{8t_f}$", r"$<Q>$", verbose=verbose)

    # Plot topsus
    distribution_plotter(
        data, "topsus", r"$\sqrt{8t_f}$", r"$\chi(<Q^2>)$", verbose=verbose)

    # Plot plaq
    distribution_plotter(
        data, "plaq", r"$\sqrt{8t_f}$", r"$P_{\mu\nu}$", verbose=verbose)


def distribution_plotter(data, observable, xlabel, ylabel, mark_interval=10,
                         verbose=False):
    """
    Plots distributions to analyse how we deepend on the epsilon in data 
    generation.

    Plots autocorr together all in one figure, eps vs final autocorr, 
    eps vs obs at specific flow time
    """

    folder_path = "../figures"
    if not os.path.isdir(folder_path):
        folder_path = "../../figures"

    # Adds distribution runs folder
    folder_path = os.path.join(folder_path, "distribution_runs")
    check_folder(folder_path, verbose=verbose)

    # Adds post analysis folder
    folder_path = os.path.join(folder_path, "post_analysis")
    check_folder(folder_path, verbose=verbose)

    # Adds observable folder
    folder_path = os.path.join(folder_path, observable)
    check_folder(folder_path, verbose=verbose)

    # Retrieves relevant values
    eps_values = sorted(data.keys())
    autocorr = []
    obs_data = []
    data_types = ["unanalyzed", "bootstrap", "jackknife"]

    for eps in eps_values:
        # print data[eps][observable][6.0]["with_autocorr"].keys(), \
        #     data[eps][observable][6.0]["with_autocorr"]["autocorr"].keys(), \
        #     data[eps][observable][6.0]["with_autocorr"]["jackknife"].keys()

        autocorr.append(data[eps][observable][6.0]
                        ["with_autocorr"]["autocorr"])
        obs_data.append({t: data[eps][observable][6.0]["with_autocorr"][t]
                         for t in data_types})

    # Plots the different observables
    for t in data_types:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i, eps in enumerate(eps_values):
            ax.errorbar(obs_data[i][t]["x"], obs_data[i][t]["y"],
                        yerr=obs_data[i][t]["y_error"],
                        label=r"$\epsilon_{rnd}=%.2f$" % eps,
                        alpha=0.5, capsize=5, fmt="_",
                        markevery=mark_interval,
                        errorevery=mark_interval)

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        # Checks and creates relevant folder
        figname = os.path.join(folder_path, "{0:s}_{1:s}.pdf".format(
            t, observable))
        fig.savefig(figname)
        print "Created figure {}".format(figname)

        plt.close(fig)

    # Plots eps vs obs at final flow time
    for t in data_types:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        x = [obs_data[i][t]["x"][-1] for i in range(len(eps_values))]
        y = [obs_data[i][t]["y"][-1] for i in range(len(eps_values))]
        yerr = [obs_data[i][t]["y_error"][-1] for i in range(len(eps_values))]

        ax.errorbar(eps_values, y, yerr=yerr,
                    label=r"$t_f={0:.2f}$".format(x[0]),
                    alpha=0.5, capsize=5, fmt="_")

        ax.legend()
        ax.set_xlabel(r"$\epsilon_{rnd}$")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        # Checks and creates relevant folder
        figname = os.path.join(folder_path, "eps_vs_{0:s}_{1:s}.pdf".format(
            observable, t))
        fig.savefig(figname)
        print "Created figure {}".format(figname)

        plt.close(fig)

    # Plots eps vs autocorr at final flow time
    fig = plt.figure()
    ax = fig.add_subplot(111)

    y = [autocorr[i]["tau_int"][-1] for i in range(len(eps_values))]
    yerr = [autocorr[i]["tau_int_err"][-1] for i in range(len(eps_values))]

    ax.errorbar(eps_values, y, yerr=yerr,
                label=r"$t_f={0:.2f}$".format(x[0]),
                alpha=0.5, capsize=5, fmt="_")

    ax.legend()
    ax.set_xlabel(r"$\epsilon_{rnd}$")
    ax.set_ylabel(r"$\tau_\mathrm{int}$")
    ax.grid(True)

    # Checks and creates relevant folder
    figname = os.path.join(folder_path, "eps_vs_tau_int_{0:s}_{1:s}.pdf".format(
        observable, t))
    fig.savefig(figname)
    print "Created figure {}".format(figname)

    plt.close(fig)


if __name__ == '__main__':
    distribution_analysis()
