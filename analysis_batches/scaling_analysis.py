#!/usr/bin/env python2

from default_analysis_params import get_default_parameters
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


try:
    from tools.folderreadingtools import get_num_observables, check_folder, \
        load_observable, check_relative_path, SlurmDataReader
except ImportError:
    sys.path.insert(0, "../")
    from tools.folderreadingtools import get_num_observables, check_folder, \
        load_observable, check_relative_path, SlurmDataReader


def scaling_analysis():
    """
    Scaling analysis.
    """

    # TODO: complete load slurm output function
    # TODO: add line fitting procedure to plot_scaling function
    # TODO: double check what I am plotting makes sense(see pdf)

    # Basic parameters
    verbose = True
    run_pre_analysis = True
    json_file = "run_times_tmp.json"
    datapath = os.path.join(("/Users/hansmathiasmamenvege/Programming/LQCD/"
                             "data/scaling_output/"), json_file)
    datapath = os.path.join(("/Users/hansmathiasmamenvege/Programming/LQCD/"
                             "LatticeAnalyser"), json_file)

    slurm_output_folder = check_relative_path("../data/scaling_output")

    slurm_json_output_path = os.path.split(datapath)[0]
    slurm_json_output_path = os.path.join(slurm_json_output_path,
                                          "slurm_output_data.json")

    # Comment this out to use old file
    if os.path.isfile(slurm_json_output_path):
        datapath = slurm_json_output_path

    # Extract times from slurm files and put into json file
    if not os.path.isfile(datapath):
        print "No {} found. Loading slurm data.".format(json_file)
        load_slurm_folder(slurm_output_folder, slurm_json_output_path)
        datapath = slurm_json_output_path

    # Basic figure setup
    base_figure_folder = check_relative_path("figures")
    base_figure_folder = os.path.join(base_figure_folder, "scaling")
    check_folder(base_figure_folder, verbose=verbose)

    # Strong scaling folder setup
    strong_scaling_figure_folder = os.path.join(base_figure_folder, "strong")
    check_folder(strong_scaling_figure_folder, verbose=verbose)

    # Weak scaling folder setup
    weak_scaling_figure_folder = os.path.join(base_figure_folder, "weak")
    check_folder(weak_scaling_figure_folder, verbose=verbose)

    default_params = get_default_parameters(
        data_batch_folder="temp", include_euclidean_time_obs=False)

    # Loads scaling times and splits into weak and strong
    with open(datapath, "r") as f:
        scaling_times = json.load(f)["runs"]
    strong_scaling_times = filter_scalings(scaling_times, "strong_scaling_np")
    weak_scaling_times = filter_scalings(scaling_times, "weak_scaling_np")

    # Splits strong scaling into gen, io, flow
    gen_strong_scaling = filter_scalings(strong_scaling_times, "gen")
    io_strong_scaling = filter_scalings(strong_scaling_times, "io")
    flow_strong_scaling = filter_scalings(strong_scaling_times, "flow")

    # Splits weak scaling into gen, io, flow
    gen_weak_scaling = filter_scalings(weak_scaling_times, "gen")
    gen_weak_scaling = filter_duplicates(gen_weak_scaling)
    io_weak_scaling = filter_scalings(weak_scaling_times, "io")
    flow_weak_scaling = filter_scalings(weak_scaling_times, "flow")

    # Adds number of processors to strong scaling
    gen_strong_scaling = add_numprocs(gen_strong_scaling)
    io_strong_scaling = add_numprocs(io_strong_scaling)
    flow_strong_scaling = add_numprocs(flow_strong_scaling)

    # Adds number of processors to strong scaling
    gen_weak_scaling = add_numprocs(gen_weak_scaling)
    io_weak_scaling = add_numprocs(io_weak_scaling)
    flow_weak_scaling = add_numprocs(flow_weak_scaling)

    scalings = [gen_strong_scaling, io_strong_scaling, flow_strong_scaling,
                gen_weak_scaling, io_weak_scaling, flow_weak_scaling]

    times_to_scan = ["update_time", "time"]
    times_to_scan = ["time"]

    # For speedup and retrieving parallelizability fraction.
    min_procs = 8

    strong_scaling_list = []
    weak_scaling_list = []

    for time_type in times_to_scan:

        # Loops over scaling values in scalings
        for sv in scalings:
            x = [i["NP"] for i in sv]
            y = [i[time_type] for i in sv]

            # Sets up filename and folder name
            _scaling = list(set([i["runname"].split("_")[0] for i in sv]))
            _sc_part = list(set([i["runname"].split("_")[-1] for i in sv]))
            assert len(_scaling) == 1, \
                "incorrect sv type list: {}".format(_scaling)
            assert len(_sc_part) == 1, \
                "incorrect sv part list length: {}".format(_sc_part)
            _sc_part = _sc_part[0]
            _scaling = _scaling[0]
            figure_name = "{}_{}_{}.pdf".format(_scaling, _sc_part, time_type)

            if _sc_part != "gen" and time_type == "update_time":
                print "Skipping {}".format(figure_name)
                continue

            # Sets correct figure folder
            if _scaling == "strong":
                _loc = "upper right"
                figure_folder = strong_scaling_figure_folder
            elif _scaling == "weak":
                _loc = "upper left"
                figure_folder = weak_scaling_figure_folder
            else:
                raise ValueError("Scaling type not recognized for"
                                 " folder: {}".format(_scaling))

            if _sc_part == "io":
                _label = r"Input/Output"
            elif _sc_part == "gen":
                _label = r"Configuration generation"
            else:
                _label = _sc_part.capitalize()

            _xlabel = r"$N_p$"
            if time_type == "time":
                _time_type = _sc_part

            if _sc_part == "io":
                _ylabel = r"$t_\mathrm{IO}$[s]"
            else:
                _ylabel = r"$t_\mathrm{%s}$[s]" % _time_type.replace(
                    "_", r"\ ").capitalize()

            # Sets speedup labels
            if _sc_part == "io":
                _ylabel_speedup = (r"$t_{\mathrm{IO},p=%s}/t_{\mathrm{IO},p}$"
                                   "[s]" % min_procs)
            else:
                _tmp = _time_type.replace("_", r"\ ").capitalize()
                _ylabel_speedup = (r"$t_{\mathrm{%s},p=%s}/t_{\mathrm{%s},p}$"
                    "[s]" % (_tmp, min_procs, _tmp))

            _tmp_dict = {
                "sc": _sc_part,
                "x": np.asarray(x),
                "y": np.asarray(y),
                "label": _label,
                "xlabel": _xlabel,
                "ylabel": _ylabel,
                "ylabel_speedup": _ylabel_speedup,
                "figure_folder": figure_folder,
                "figure_name": figure_name,
                "loc": _loc,
            }

            if _scaling == "strong":
                strong_scaling_list.append(_tmp_dict)
            else:
                weak_scaling_list.append(_tmp_dict)

            # plot_scaling(x, y, _label, _xlabel, _ylabel,
            #              figure_folder, figure_name, loc=_loc)

    # plot_all_scalings(strong_scaling_list, "strong")
    # plot_all_scalings(weak_scaling_list, "weak")
    plot_speedup(strong_scaling_list, "strong")
    # amdahls_law(strong_scaling_list, "strong")


def filter_scalings(scaling_list, scaling_type):
    """
    Filter the scaling values.
    """
    return filter(
        lambda _f: True if scaling_type in _f["runname"] else False,
        scaling_list)


def filter_duplicates(old_list):
    """
    Filter out run duplicates.
    """
    new_list = []
    indices_to_remove = []

    # Store the one which was run the latest
    for i, _l1 in enumerate(old_list):

        for j, _l2 in enumerate(old_list):

            # If the elements is identical
            if i == j:
                continue

            # If runnames match, it means we have a duplicate
            if _l1["threads"] == _l2["threads"]:

                # Keeps the newest of the runs
                if _l1["start_time"] > _l2["start_time"]:
                    indices_to_remove.append(i)
                else:
                    indices_to_remove.append(j)

    for i, elem in enumerate(old_list):
        if not i in indices_to_remove:
            new_list.append(elem)

    return new_list


def add_numprocs(value):
    """
    Adds number of processors and returns a list sorted by this.
    """
    return_list = []
    for l in value:
        _tmp_dict = l
        _tmp_dict["NP"] = l["totsize"] / l["subdimsize"]
        return_list.append(_tmp_dict)
    return sorted(return_list, key=lambda i: i["NP"])


def plot_scaling(x, y, label, xlabel, ylabel, figfolder, figname, loc=None):
    """
    Plots the scaling.
    """
    fig, ax = plt.subplots()
    ax.semilogx(x, y, "o-", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc=loc)
    figpath = os.path.join(figfolder, figname)
    fig.savefig(figpath)
    print "Figure {} created.".format(figpath)


def plot_all_scalings(sc_list, sc_type):
    """
    Plots gen, flow and io scaling in one window.
    """

    fig, axes = plt.subplots(nrows=3, ncols=1)
    sc_list = [x for _, x in sorted(
        zip(["gen", "flow", "io"], sc_list), key=lambda k: k[1]["sc"])]
    sc_list = [sc_list[1], sc_list[0], sc_list[2]]  # Bad hard coding

    for data, ax in zip(sc_list, axes):
        ax.semilogx(data["x"], data["y"], "o-", label=data["label"])
        ax.set_xlabel(data["xlabel"])
        ax.set_ylabel(data["ylabel"])
        ax.grid(True)
        ax.legend(loc=data["loc"])

    figpath = os.path.join(data["figure_folder"], sc_type + "_all" + ".pdf")
    fig.savefig(figpath)
    print "Figure {} created.".format(figpath)


def load_slurm_folder(p, json_fpath, verbose=True):
    """
    Loads all slurm data in a folder and builds a json file which it stores.
    """
    def filter_function(f): return True if ".out" in f else False
    slurm_dict = {"runs": []}
    for f in filter(filter_function, os.listdir(p)):
        if verbose:
            print "Reading Slurm output from ", os.path.join(p, f)
        _tmp_slurm_data = SlurmDataReader(os.path.join(p, f))
        slurm_dict["runs"].append(_tmp_slurm_data.read(verbose=True))

    assert os.path.splitext(json_fpath)[-1] == ".json", (
        "please specify json output location")

    with file(json_fpath, "w+") as json_file:
        json.dump(slurm_dict, json_file, indent=4)

    print "json data dumped to {}".format(json_fpath)


def plot_speedup(sc_list, sc_type):
    """Plots the speedup of a program."""
    fig, axes = plt.subplots(nrows=3, ncols=1)
    sc_list = [x for _, x in sorted(
        zip(["gen", "flow", "io"], sc_list), key=lambda k: k[1]["sc"])]
    sc_list = [sc_list[1], sc_list[0], sc_list[2]]  # Bad hard coding

    for data in sc_list:
        data["y"] = data["y"][0] / data["y"]

    for data, ax in zip(sc_list, axes):
        ax.semilogx(data["x"], data["y"], "o-", label=data["label"])
        ax.set_xlabel(data["xlabel"])
        ax.set_ylabel(data["ylabel_speedup"])
        ax.grid(True)
        ax.legend(loc=data["loc"])

    figpath = os.path.join(
        data["figure_folder"], "speedup_" + sc_type + "_all" + ".pdf")
    fig.savefig(figpath)
    print "Figure {} created.".format(figpath)


def amdahls_law(sc_list, sc_type):
    """Uses Amdahl's law to get the fraction that can be parallelized."""


if __name__ == '__main__':
    scaling_analysis()
