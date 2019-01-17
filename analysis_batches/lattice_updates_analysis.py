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
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
    from matplotlib import cm, rc, rcParams
    from mpl_toolkits.mplot3d import axes3d

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

    run_pre_analysis = False
    verbose = False
    N_corr = [200, 400, 600]
    N_updates = [10, 20, 30]
    param_list = []
    beta = 6.0
    figure_folder = "../figures/lattice_updates"
    output_folder = "../data/lattice_update_data"
    ############ Sets up the different N_up/N_corr analysises ##########

    # Sets up Slurm output files
    if not os.path.isdir(output_folder):
        output_folder = os.path.join("..", output_folder)
    output_file_path = os.path.join(output_folder, "output_files")

    # Sets up empty nested dictionary
    output_files = {
        icorr: {
            iup: {} for iup in N_updates
        } for icorr in N_corr}

    # Retrieves standard parameters
    default_params = get_default_parameters(
        data_batch_folder="temp", verbose=verbose)

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
            _params["beta"] = beta
            _params["color"] = "#377eb8"
            _params["N"] = 16
            _params["NT"] = _params["N"]*2
            _params["observables"] = ["plaq", "energy", "topc"]
            _params.update(output_files[icorr][iup])
            _times = read_run_time(_params["output_path"])
            _params["total_runtime"] = _times[-1][-1]
            _params["update_time"] = _times[0][0]
            param_list.append(_params)

    if run_pre_analysis:
        # Submitting distribution analysis
        for analysis_parameters in param_list:
            pre_analysis(analysis_parameters)

    print("Success: pre analysis done.")

    # Use post_analysis data for further analysis.
    data = {
        icorr: {
            iup: {} for iup in N_updates
        } for icorr in N_corr}
    for i, _params in enumerate(param_list):
        print "Loading data for NCorr={0:d} NUp={1:d}".format(_params["NCorr"],
                                                              _params["NUp"])
        data[_params["NCorr"]][_params["NUp"]] = {
            "data": post_analysis.PostAnalysisDataReader(
                [_params], observables_to_load=_params["observables"],
                verbose=verbose),
            "params": _params,
        }

    print("Success: post analysis data retrieved.")

    X_corr, Y_up = np.meshgrid(N_corr, N_updates)

    # Sets up time grid
    Z_total_runtimes = np.zeros((3, 3))
    Z_update_times = np.zeros((3, 3))
    X_flow = np.zeros((3, 3, 251))
    Z_autocorr = np.zeros((3, 3, 251))
    Z_autocorr_error = np.zeros((3, 3, 251))
    for i, icorr in enumerate(N_corr):
        for j, iup in enumerate(N_updates):

            Z_total_runtimes[i, j] = \
                data[icorr][iup]["params"]["total_runtime"]
            Z_update_times[i, j] = \
                data[icorr][iup]["params"]["update_time"]

            _tmp = data[icorr][iup]["data"]["topc"][beta]
            Z_autocorr[i, j] = \
                _tmp["with_autocorr"]["autocorr"]["tau_int"]
            Z_autocorr_error[i, j] = \
                _tmp["with_autocorr"]["autocorr"]["tau_int_err"]

            X_flow[i, j] = _tmp["with_autocorr"]["unanalyzed"]["x"]
            X_flow[i, j] *= data[icorr][iup]["data"].flow_epsilon[6.0]
            X_flow[i, j] = np.sqrt(8*X_flow[i, j])
            X_flow[i, j] *= data[icorr][iup]["data"].lattice_sizes[6.0][0]

    # Plots update and total run-times on grid
    heatmap_plotter(N_corr, N_updates, Z_total_runtimes,
                    "topc_total_runtime.pdf",
                    xlabel=r"$N_\mathrm{corr}$", ylabel=r"$N_\mathrm{up}$",
                    cbartitle=r"$t_\mathrm{total}$",
                    figure_folder=figure_folder)

    heatmap_plotter(N_corr, N_updates, Z_update_times,
                    "topc_update_runtime.pdf",
                    xlabel=r"$N_\mathrm{corr}$", ylabel=r"$N_\mathrm{up}$",
                    cbartitle=r"$t_\mathrm{update}$",
                    figure_folder=figure_folder)

    # Plots final autocorrelations on grid
    flow_times = [0, 100, 250]
    for tf in flow_times:
        heatmap_plotter(N_corr, N_updates, Z_autocorr[:, :, tf],
                        "topc_autocorr_tau{0:d}.pdf".format(tf),
                        xlabel=r"$N_\mathrm{corr}$", ylabel=r"$N_\mathrm{up}$",
                        cbartitle=r"$\tau_\mathrm{int}$",
                        figure_folder=figure_folder)

        heatmap_plotter(N_corr, N_updates, Z_autocorr_error[:, :, tf],
                        "topc_autocorr_err_tau{0:d}.pdf".format(tf),
                        xlabel=r"$N_\mathrm{corr}$", ylabel=r"$N_\mathrm{up}$",
                        cbartitle=r"$\tau_\mathrm{int}$",
                        figure_folder=figure_folder)

    # Plots all of the 9 autocorrs in single figure
    plot9_figures(X_flow, X_corr, Y_up, Z_autocorr, Z_autocorr_error,
                  "topc_autocorr.pdf",
                  xlabel=r"$\sqrt{8t_{f}}$", ylabel=r"$\tau_\mathrm{int}$",
                  figure_folder=figure_folder, mark_interval=10)


def heatmap_plotter(x, y, z, figure_name, tick_param_fs=None, label_fs=None,
                    vmin=None, vmax=None, xlabel=None, ylabel=None,
                    cbartitle=None, x_tick_mode="int", y_tick_mode="int",
                    figure_folder=""):
    """Plots a heatmap surface."""
    fig, ax = plt.subplots()

    if x_tick_mode == "exp":
        xheaders = ['%1.1e' % i for i in x]
    elif x_tick_mode == "int":
        xheaders = ['%d' % int(i) for i in x]
    elif x_tick_mode == "float":
        xheaders = ['%1.2f' % i for i in x]
    else:
        xheaders = ['%g' % i for i in x]

    if y_tick_mode == "exp":
        yheaders = ['%1.1e' % i for i in y]
    elif y_tick_mode == "int":
        yheaders = ['%d' % int(i) for i in y]
    elif y_tick_mode == "float":
        yheaders = ['%1.2f' % i for i in y]
    else:
        yheaders = ['%g' % i for i in y]

    heatmap = ax.pcolormesh(z, edgecolors="k", linewidth=2,
                            vmin=vmin, vmax=vmax, cmap="YlGnBu")
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.ax.tick_params(labelsize=tick_param_fs)
    cbar.ax.set_title(cbartitle, fontsize=label_fs)

    # # ax.set_title(method, fontsize=fs1)
    ax.set_xticks(np.arange(z.shape[1]) + .5, minor=False)
    ax.set_yticks(np.arange(z.shape[0]) + .5, minor=False)

    ax.set_xticklabels(xheaders, rotation=90, fontsize=tick_param_fs)
    ax.set_yticklabels(yheaders, fontsize=tick_param_fs)

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)

    check_folder(figure_folder)
    figure_path = os.path.join(figure_folder, figure_name)
    fig.savefig(figure_path)
    print("Figure saved at {}".format(figure_path))
    plt.close(fig)


def plot9_figures(t, x, y, z, z_error, figure_name,
                  xlabel=None, ylabel=None, figure_folder="",
                  mark_interval=1):

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            axes[i, j].errorbar(
                t[i, j], z[i, j],
                yerr=z_error[i, j],
                label=r"$N_\mathrm{corr}=%d, N_\mathrm{up}=%d$" % (
                    x[i, j], y[i, j]),
                alpha=0.5, capsize=5, fmt="_",
                markevery=mark_interval,
                errorevery=mark_interval)

            if i == 1 and j == 0:
                axes[i, j].set_ylabel(ylabel, fontsize=12)
            if i == 2 and j == 1:
                axes[i, j].set_xlabel(xlabel, fontsize=12)

            axes[i, j].legend(fontsize=6, loc="upper left")
            axes[i, j].grid(True)

    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.96,
                        top=0.96, wspace=0.09, hspace=0.1)
    check_folder(figure_folder)
    figure_path = os.path.join(figure_folder, figure_name)
    fig.savefig(figure_path)
    print("Figure saved at {}".format(figure_path))
    plt.close(fig)


def read_run_time(filepath):
    """Function for retrieving the run time from Slurm output files.

    Args:
        filepath (str): filepath to Slurm output file.

    Returns:
        float, run time in [hours, seconds]."""

    update_time = []
    total_time = []

    with open(filepath, "r") as f:

        for l in f:
            # String to search:
            # Program complete. Time used: 0.723097 hours (2603.150960 seconds)
            _total_time = re.findall(
                r"Program complete\D*(\d+\.\d+)\D*\((\d+\.\d+) seconds\)", l)
            if len(_total_time) > 0:
                total_time = list(map(float, _total_time[0]))

            # String to search:
            # Total update time for 120000 updates: 7367.337558 sec.
            _update_time = re.findall(
                r"Total update time for \d+ updates: (\d+\.\d+) sec\.", l)
            if len(_update_time) > 0:
                update_time = list(map(float, _update_time))

            if len(update_time) != 0 and len(total_time) != 0:
                break
        else:
            raise IOError("No times found for {}".format(filepath))

    return update_time, total_time


if __name__ == '__main__':
    lattice_updates_analysis()
