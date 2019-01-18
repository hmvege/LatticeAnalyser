#!/usr/bin/env python2

from default_analysis_params import get_default_parameters
import copy
import os
import sys
import re
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

try:
    from pre_analysis.pre_analyser import pre_analysis
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder, \
        load_observable
except ImportError:
    sys.path.insert(0, "../")
    from pre_analysis.pre_analyser import pre_analysis
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder, \
        load_observable


def thermalization_analysis():
    """Runs the thermalization analysis."""

    verbose = True
    run_pre_analysis = True
    batch_folder = "../data/thermalization_data"
    if not os.path.isdir(batch_folder):
        batch_folder = os.path.join("..", batch_folder)

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

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.


def plot_termalization(data_x, data_y, data_labels, x_label, y_label,
                       figname, figfolder, plot_log="default", plot_window="1"):
    """
    Plots the 
    """
    fortsett her! husk å legge til modes:
    - samme vindu
    - tre vinduer
    - log-log / logx / logy
    assert len(data_x)==len(data_y)==len(data_labels), (
        "data_x, data_y and data_labels is not of equal length: "
        "{} {} {}".format(len(data_x), len(data_y), len(data_labels)))

    pass


if __name__ == '__main__':
    thermalization_analysis()
