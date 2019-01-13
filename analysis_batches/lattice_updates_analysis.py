#!/usr/bin/env python2

from pre_analysis.pre_analyser import pre_analysis
from post_analysis.post_analyser import post_analysis
from default_analysis_params import get_default_parameters
from tools.folderreadingtools import get_num_observables
import copy
import os


def lattice_updates_analysis():
    default_params = get_default_parameters(data_batch_folder="temp")

    N_corr = [200, 400, 600]
    N_updates = [10, 20, 30]
    param_list = []
    ############ Sets up the different N_up/N_corr analysises ##########

    for i_N_corr in N_corr:
        for i_N_up in N_updates:
            _params = copy.deepcopy(default_params)
            _parmas["observables"] = ["plaq", "energy", "topc", "topct"]
            param_list.append(_params)

    raise NotImplementedError("Missing data to fully implement "
        "lattice_updates_analysis. Also, need to set up a comparison in some "
        "kind of post analysis.")

    # Submitting distribution analysis
    for analysis_parameters in param_list:
        pre_analysis(analysis_parameters)

    # TODO: find folder with .out files
    # TODO: run pre analysis
    # TODO: load pre analysis data
    # TODO: load times
    # TODO: plot final autocorr on grid
    # TODO: plot times on grid
    

if __name__ == '__main__':
    lattice_updates_analysis()