#!/usr/bin/env python2

from pre_analysis.pre_analyser import pre_analysis
from post_analysis.post_analyser import post_analysis
from default_analysis_params import get_default_parameters
from tools.folderreadingtools import get_num_observables
import copy
import os

def distribution_analysis():
    """Analysis for different SU3 epsilon matrix geeration values."""
    default_params = get_default_parameters(data_batch_folder="temp")

    ########## Distribution analysis ##########
    dist_eps = [0.05, 0.10, 0.20, 0.24, 0.30, 0.40, 0.60]
    def create_dist_batch_set(default_parameters, eps):
        clean_str = lambda s: str("%-2.2f"%s).replace(".", "")
        dist_data_beta60_analysis = copy.deepcopy(default_parameters)
        dist_data_beta60_analysis["batch_folder"] = (
            "../data/distribution_tests/distribution_runs")
        dist_data_beta60_analysis["batch_name"] = \
            "distribution_test_eps{0:s}".format(clean_str(eps))
        dist_data_beta60_analysis["beta"] = 6.0
        dist_data_beta60_analysis["num_bins_per_int"] = 16
        dist_data_beta60_analysis["bin_range"] = [-2.1, 2.1]
        dist_data_beta60_analysis["hist_flow_times"] = [0, 250, 600]
        dist_data_beta60_analysis["NCfgs"] = get_num_observables(
            dist_data_beta60_analysis["batch_folder"],
            dist_data_beta60_analysis["batch_name"])
        dist_data_beta60_analysis["obs_file"] = "6_6.00" # 6^3x12, beta=6.0
        dist_data_beta60_analysis["N"] = 6
        dist_data_beta60_analysis["NT"] = 12
        dist_data_beta60_analysis["color"] = "#377eb8"
        return dist_data_beta60_analysis

    dist_param_list = [create_dist_batch_set(default_params, _eps)
        for _eps in dist_eps]

    #### Submitting distribution analysis
    analysis_parameter_list = dist_param_list
    for analysis_parameters in analysis_parameter_list:
        pre_analysis(analysis_parameters)

if __name__ == '__main__':
    distribution_analysis()