#!/usr/bin/env python2

# from pre_analysis.pre_analyser import pre_analysis
# from post_analysis.post_analyser import post_analysis
# from default_analysis_params import get_default_parameters
# from tools.folderreadingtools import get_num_observables
import copy
import os
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
    from matplotlib import rc, rcParams

rc("text", usetex=True)
rcParams["font.family"] += ["serif"]


try:
    from default_analysis_params import get_default_parameters
    from pre_analysis.pre_analyser import pre_analysis, get_data_parameters
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder
except ImportError:
    import sys
    sys.path.insert(0, "../")
    from default_analysis_params import get_default_parameters
    from pre_analysis.pre_analyser import pre_analysis, get_data_parameters
    import post_analysis.post_analyser as post_analysis
    from tools.folderreadingtools import get_num_observables, check_folder


def topc_modes_analysis():
    """Analysis for different lattice sizes and their topological charges."""
    default_params = get_default_parameters(data_batch_folder="temp")

    run_pre_analysis = True
    verbose = True

    data_path = "../data/"
    if not os.path.isdir(data_path):
        data_path = "../" + data_path

    ########## Smaug data 8x16 analysis ##########
    smaug8x16_data_beta60_analysis = copy.deepcopy(default_params)
    smaug8x16_data_beta60_analysis["batch_folder"] = data_path
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
    smaug12x24_data_beta60_analysis["batch_folder"] = data_path
    smaug12x24_data_beta60_analysis["batch_name"] = "beta60_12x24_run"
    smaug12x24_data_beta60_analysis["beta"] = 6.0
    smaug12x24_data_beta60_analysis["topc_y_limits"] = [-4, 4]
    smaug12x24_data_beta60_analysis["num_bins_per_int"] = 16
    smaug12x24_data_beta60_analysis["bin_range"] = [-4.5, 4.5]
    smaug12x24_data_beta60_analysis["hist_flow_times"] = [0, 100, 600]
    smaug12x24_data_beta60_analysis["NCfgs"] = get_num_observables(
        smaug12x24_data_beta60_analysis["batch_folder"],
        smaug12x24_data_beta60_analysis["batch_name"])
    smaug12x24_data_beta60_analysis["obs_file"] = "12_6.00"
    smaug12x24_data_beta60_analysis["N"] = 12
    smaug12x24_data_beta60_analysis["NT"] = 24
    smaug12x24_data_beta60_analysis["color"] = "#377eb8"

    ########## Smaug data 16x32 analysis ##########
    smaug16x32_data_beta61_analysis = copy.deepcopy(default_params)
    smaug16x32_data_beta61_analysis["batch_folder"] = data_path
    smaug16x32_data_beta61_analysis["batch_name"] = "beta61_16x32_run"
    smaug16x32_data_beta61_analysis["beta"] = 6.1
    smaug16x32_data_beta61_analysis["topc_y_limits"] = [-8, 8]
    smaug16x32_data_beta61_analysis["num_bins_per_int"] = 16
    smaug16x32_data_beta61_analysis["bin_range"] = [-7.5, 7.5]
    smaug16x32_data_beta61_analysis["hist_flow_times"] = [0, 100, 400]
    smaug16x32_data_beta61_analysis["NCfgs"] = get_num_observables(
        smaug16x32_data_beta61_analysis["batch_folder"],
        smaug16x32_data_beta61_analysis["batch_name"])
    smaug16x32_data_beta61_analysis["obs_file"] = "16_6.10"
    smaug16x32_data_beta61_analysis["N"] = 16
    smaug16x32_data_beta61_analysis["NT"] = 32
    smaug16x32_data_beta61_analysis["color"] = "#377eb8"

    param_list = [
        smaug8x16_data_beta60_analysis,
        smaug12x24_data_beta60_analysis,
        smaug16x32_data_beta61_analysis]

    if run_pre_analysis:
        # Submitting analysis
        for analysis_parameters in param_list:
            pre_analysis(analysis_parameters)

    # Loads topc data
    data = []
    # N_val = [24, 24, 28]
    for i, param in enumerate(param_list):
        print "Loading data for: {}".format(param["batch_name"])
        # fpath = os.path.join(param["batch_folder"], param["batch_name"],
        #                      "{0:d}_{1:.2f}.npy".format(N_val[i],
        #                                                 param["beta"]))
        data_, p = get_data_parameters(param)
        data.append({"data": data_("topc")["obs"].T,
                     "beta": param["beta"],
                     "N": param["N"]})

        # print data_("topc")["obs"].shape

    # Flow time to plots
    flow_times = [0, 25, 50, 100, 150, 250, 450, 600]

    # Histogram plotting
    xlim = 7.5
    NBins = np.arange(-xlim, xlim, 0.05)
    for t_f in flow_times:
        # Adds unanalyzed data
        fig, axes = plt.subplots(len(param_list), 1,
                                 sharey=False, sharex=True)
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            lab = r"${0:d}^3\times{1:d}$, $\beta={2:.2f}$".format(
                data[i]["N"], data[i]["N"]*2, data[i]["beta"])

            weights = np.ones_like(data[i]["data"][t_f])
            weights /= len(data[i]["data"][t_f])
            ax.hist(data[i]["data"][t_f], bins=NBins,
                    label=lab, weights=weights)
            ax.legend(loc="upper right")
            ax.grid(True)
            ax.set_xlim(-xlim, xlim)

            if i == 1:
                ax.set_ylabel(r"Hits(normalized)")
            elif i == 2:
                ax.set_xlabel(r"$Q$")

        # Sets up figure
        figpath = "figures/topc_modes_analysis"
        if not os.path.isdir(figpath):
            figpath = "../" + figpath
        check_folder(figpath, verbose=verbose)
        figpath = os.path.join(figpath, "topc_modes_tf{}.pdf".format(t_f))
        fig.savefig(figpath)
        print "Figure saved at {0:s}".format(figpath)
        plt.close(fig)


if __name__ == '__main__':
    topc_modes_analysis()
