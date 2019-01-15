#!/usr/bin/env python2

# from pre_analysis.pre_analyser import pre_analysis
# from post_analysis.post_analyser import post_analysis
# from default_analysis_params import get_default_parameters
# from tools.folderreadingtools import get_num_observables
# import copy
# import os

from default_analysis_params import get_default_parameters, load_pickle, \
    save_pickle

import copy
import os
import sys
import re
import numpy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt

try:
    from pre_analysis.pre_analyser import pre_analysis
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder
except ImportError:
    sys.path.insert(0, "../")
    from pre_analysis.pre_analyser import pre_analysis
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder


def lattice_updates_analysis():
    default_params = get_default_parameters(data_batch_folder="temp")

    run_pre_analysis = True
    N_corr = [200, 400, 600]
    N_updates = [10, 20, 30]
    param_list = []
    ############ Sets up the different N_up/N_corr analysises ##########

    # Sets up Slurm output files
    output_folder = "../data/lattice_update_data"
    if not os.path.isdir(output_folder):
        output_folder = os.path.join("..", output_folder)
    output_file_path = os.path.join(output_folder, "output_files")

    # Sets up empty nested dictionary
    output_files = {
        icorr: {
            iup: {} for iup in N_updates
        } for icorr in N_corr}

    # Loops through different corr lengths and link update sizes
    for icorr in N_corr:
        for iup in N_updates:

            # Loops over files in directory
            for of in os.listdir(output_file_path):
                _tmp = re.findall(r"NUP([\d]+)_NCORR([\d]+)", of)[0]
                _NUp, _NCorr = list(map(int, _tmp))

                # If icorr and iup matches files, we add it to the dictionary
                # we created earlier.
                if icorr == _NCorr and iup == _NUp:
                    output_files[icorr][iup] = {
                        "NUp": iup,
                        "NCorr": icorr,
                        "output_path": os.path.join(output_file_path, of)
                    }
                    break

    # Sets up parameter list for analysis
    for icorr in N_corr:
        for iup in N_updates:
            _params = copy.deepcopy(default_params)
            _params["batch_folder"] = output_folder
            _params["batch_name"] = \
                "B60_NUP{0:d}_NCORR{1:d}".format(iup, icorr)
            _params["NCfgs"] = get_num_observables(_params["batch_folder"],
                                                   _params["batch_name"])
            _params["N"] = 16
            _params["NT"] = _params["N"]*2
            _params["observables"] = ["plaq", "energy", "topc"]
            _params.update(output_files[icorr][iup])
            _params["runtime"] = read_run_time(_params["output_path"])[1]
            param_list.append(_params)

    if run_pre_analysis:
        # Submitting distribution analysis
        for analysis_parameters in param_list:
            pre_analysis(analysis_parameters)

    print("Success: pre analysis done.")

    # TODO: run pre analysis
    # TODO: load pre analysis data
    # TODO: plot final autocorr on grid
    # TODO: plot times on grid


def read_run_time(filepath):
    """Function for retrieving the run time from Slurm output files.

    Args:
        filepath (str): filepath to Slurm output file.

    Returns:
        float, run time in [hours, seconds]."""

    found_time = []

    with open(filepath, "r") as f:

        for l in f:
            # String to search:
            # Program complete. Time used: 0.723097 hours (2603.150960 seconds)
            _res = re.findall(
                r"Program complete\D*(\d+\.\d+)\D*\((\d+\.\d+) seconds\)", l)
            if len(_res) > 0:
                found_time = list(map(float, _res[0]))
                break
        else:
            raise IOError("No times found for {}".format(filepath))

    return found_time


if __name__ == '__main__':
    lattice_updates_analysis()
