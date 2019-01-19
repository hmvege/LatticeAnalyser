#!/usr/bin/env python2

import json
import os
import sys
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
    from matplotlib import rc, rcParams

rc("text", usetex=True)
rcParams["font.family"] += ["serif"]

from default_analysis_params import get_default_parameters

try:
    from tools.folderreadingtools import get_num_observables, check_folder, \
        load_observable, check_relative_path
except ImportError:
    sys.path.insert(0, "../")
    from tools.folderreadingtools import get_num_observables, check_folder, \
        load_observable, check_relative_path


def weak_scaling_flow():
    """
    Weak scaling analysis.
    """

    # Basic parameters
    verbose = True
    run_pre_analysis = True
    # batch_folder = check_relative_path("data/scaling_output")
    base_figure_folder = check_relative_path("figures/")
    base_figure_folder = os.path.join(base_figure_folder,
                                      "weak_scaling")
    check_folder(base_figure_folder, verbose=verbose)
    default_params = get_default_parameters(
        data_batch_folder="temp", include_euclidean_time_obs=False)

    # Build correct files list
    # weak_scaling_files = filter(
    #     lambda _f: True if "weak_scaling" in _f else False,
    #     os.listdir(batch_folder))
    with open(datapath, "r") as f:
        string_scaling_times = json.load(f)["runs"]
    string_scaling_times = filter(
        lambda _f: True if "weak_scaling" in _f["runname"] else False,
        string_scaling_times)

    # Splits into gen, io, flow
    weakrong_scaling = filter(
        lambda _f: True if "gen" in _f["runname"] else False,
        string_scaling_times)
    weakong_scaling = filter(
        lambda _f: True if "io" in _f["runname"] else False,
        string_scaling_times)
    weaktrong_scaling = filter(
        lambda _f: True if "flow" in _f["runname"] else False,
        string_scaling_times)