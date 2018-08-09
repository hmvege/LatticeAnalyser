#!/usr/bin/env python2

from pre_analysis.pre_analyser import pre_analysis
from post_analysis.post_analyser import post_analysis
from default_analysis_params import get_default_parameters
from tools.folderreadingtools import get_num_observables
import copy
import os


def thermalization_analysis():
    default_params = get_default_parameters(data_batch_folder="temp")

    ############ COLD START #############
    cold_start_data_beta60_analysis = copy.deepcopy(default_params)
    cold_start_data_beta60_analysis["batch_folder"] = "../data/"
    cold_start_data_beta60_analysis["batch_name"] = "beta60_8x16_run"
    cold_start_data_beta60_analysis["beta"] = 6.0
    cold_start_data_beta60_analysis["topc_y_limits"] = [-2, 2]
    cold_start_data_beta60_analysis["num_bins_per_int"] = 32
    cold_start_data_beta60_analysis["bin_range"] = [-2.5, 2.5]
    cold_start_data_beta60_analysis["hist_flow_times"] = [0, 250, 600]
    cold_start_data_beta60_analysis["NCfgs"] = get_num_observables(
        cold_start_data_beta60_analysis["batch_folder"], 
        cold_start_data_beta60_analysis["batch_name"])
    cold_start_data_beta60_analysis["obs_file"] = "8_6.00"
    cold_start_data_beta60_analysis["N"] = 8
    cold_start_data_beta60_analysis["NT"] = 16
    cold_start_data_beta60_analysis["color"] = "#377eb8"

    ########## HOT RND START ############
    hot_rst_start_data_beta60_analysis = copy.deepcopy(default_params)

    ########## HOT RST START ############
    hot_rnd_start_data_beta60_analysis = copy.deepcopy(default_params)


if __name__ == '__main__':
    thermalization_analysis()