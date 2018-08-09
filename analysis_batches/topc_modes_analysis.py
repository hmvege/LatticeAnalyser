#!/usr/bin/env python2

from pre_analysis.pre_analyser import pre_analysis
from post_analysis.post_analyser import post_analysis
from default_analysis_params import get_default_parameters
from tools.folderreadingtools import get_num_observables
import copy
import os

def topc_modes_analysis():
    """Analysis for different lattice sizes and their topological charges."""
    default_params = get_default_parameters(data_batch_folder="temp")

    ########## Smaug data 8x16 analysis ##########
    smaug8x16_data_beta60_analysis = copy.deepcopy(default_params)
    smaug8x16_data_beta60_analysis["batch_folder"] = "../data/"
    smaug8x16_data_beta60_analysis["batch_name"] = "beta60_8x16_run"
    smaug8x16_data_beta60_analysis["beta"] = 6.0
    smaug8x16_data_beta60_analysis["topc_y_limits"] = [-2, 2]
    smaug8x16_data_beta60_analysis["num_bins_per_int"] = 32
    smaug8x16_data_beta60_analysis["bin_range"] = [-2.5, 2.5]
    smaug8x16_data_beta60_analysis["hist_flow_times"] = [0, 250, 600]
    smaug8x16_data_beta60_analysis["NCfgs"] = get_num_observables(
        smaug8x16_data_beta60_analysis["batch_folder"], 
        smaug8x16_data_beta60_analysis["batch_name"])
    smaug8x16_data_beta60_analysis["obs_file"] = "8_6.00"
    smaug8x16_data_beta60_analysis["N"] = 8
    smaug8x16_data_beta60_analysis["NT"] = 16
    smaug8x16_data_beta60_analysis["color"] = "#377eb8"

    ########## Smaug data 12x24 analysis ##########
    smaug12x24_data_beta60_analysis = copy.deepcopy(default_params)
    smaug12x24_data_beta60_analysis["batch_folder"] = "../data/"
    smaug12x24_data_beta60_analysis["batch_name"] = "beta60_12x24_run"
    smaug12x24_data_beta60_analysis["beta"] = 6.0
    smaug12x24_data_beta60_analysis["topc_y_limits"] = [-4, 4]
    smaug12x24_data_beta60_analysis["num_bins_per_int"] = 16
    smaug12x24_data_beta60_analysis["bin_range"] = [-4.5, 4.5]
    smaug12x24_data_beta60_analysis["hist_flow_times"] = [0, 250, 600]
    smaug12x24_data_beta60_analysis["NCfgs"] = get_num_observables(
        smaug12x24_data_beta60_analysis["batch_folder"], 
        smaug12x24_data_beta60_analysis["batch_name"])
    smaug12x24_data_beta60_analysis["obs_file"] = "12_6.00"
    smaug12x24_data_beta60_analysis["N"] = 12
    smaug12x24_data_beta60_analysis["NT"] = 24
    smaug12x24_data_beta60_analysis["color"] = "#377eb8"

    ########## Smaug data 16x32 analysis ##########
    smaug16x32_data_beta61_analysis = copy.deepcopy(default_params)
    smaug16x32_data_beta61_analysis["batch_folder"] = "../data/"
    smaug16x32_data_beta61_analysis["batch_name"] = "beta61_16x32_run"
    smaug16x32_data_beta61_analysis["beta"] = 6.1
    smaug16x32_data_beta61_analysis["topc_y_limits"] = [-8, 8]
    smaug16x32_data_beta61_analysis["num_bins_per_int"] = 16
    smaug16x32_data_beta61_analysis["bin_range"] = [-7.5, 7.5]
    smaug16x32_data_beta61_analysis["hist_flow_times"] = [0, 250, 600]
    smaug16x32_data_beta61_analysis["NCfgs"] = get_num_observables(
        smaug16x32_data_beta61_analysis["batch_folder"], 
        smaug16x32_data_beta61_analysis["batch_name"])
    smaug16x32_data_beta61_analysis["obs_file"] = "16_6.10"
    smaug16x32_data_beta61_analysis["N"] = 16
    smaug16x32_data_beta61_analysis["NT"] = 32
    smaug16x32_data_beta61_analysis["color"] = "#377eb8"

    #### Submitting 8x16 analysis
    analysis_parameter_list = [smaug8x16_data_beta60_analysis]
    for analysis_parameters in analysis_parameter_list:
        pre_analysis(analysis_parameters)

    #### Submitting 12x24 analysis
    analysis_parameter_list = [smaug12x24_data_beta60_analysis]
    for analysis_parameters in analysis_parameter_list:
        pre_analysis(analysis_parameters)

    #### Submitting 16x32 analysis
    analysis_parameter_list = [smaug16x32_data_beta61_analysis]
    for analysis_parameters in analysis_parameter_list:
        pre_analysis(analysis_parameters)

if __name__ == '__main__':
    topc_modes_analysis()