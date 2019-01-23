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
    json_file = "run_times_updated.json"
    datapath = os.path.join(("/Users/hansmathiasmamenvege/Programming/LQCD/"
                             "data/scaling_output/"), json_file)
    datapath = os.path.join(("/Users/hansmathiasmamenvege/Programming/LQCD/"
                             "LatticeAnalyser"), json_file)

    slurm_output_folder = check_relative_path("../data/scaling_output")

    # Extract times from slurm files and put into json file
    if not os.path.isfile(datapath):
        print "No {} found. Loading slurm data.".format(json_file)
        load_slurm_folder(slurm_output_folder)

    exit("bad - read in data!!")

    # Basic figure setup
    base_figure_folder = check_relative_path("figures")
    base_figure_folder = os.path.join(base_figure_folder,
                                      "scaling")
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

    for time_type in ["update_time", "time"]:

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
                figure_folder = strong_scaling_figure_folder
            elif _scaling == "weak":
                figure_folder = weak_scaling_figure_folder
            else:
                raise ValueError("Scaling type not recognized for"
                                 " folder: {}".format(_scaling))

            plot_scaling(x, y, _sc_part.capitalize(), r"$N_p$",
                         r"$t_\mathrm{%s}$" % time_type.replace(
                             "_", r"\ ").capitalize(),
                         figure_folder, figure_name)


def filter_scalings(scaling_list, scaling_type):
    """
    Filter the scaling values.
    """
    return filter(
        lambda _f: True if scaling_type in _f["runname"] else False,
        scaling_list)


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


def plot_scaling(x, y, label, xlabel, ylabel, figfolder, figname):
    """
    Plots the scaling.
    """
    fig, ax = plt.subplots()
    ax.loglog(x, y, "o-", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    figpath = os.path.join(figfolder, figname)
    fig.savefig(figpath)
    print "Figure {} created.".format(figpath)


def load_slurm_folder(p):
    """
    Loads all slurm data in a folder and builds a json file which it stores.
    """
    filter_function = lambda f: True if ".out" in f else False
    slurm_dict = {"runs": []}
    for f in filter(filter_function, os.listdir(p)):
        _tmp_slurm_data = SlurmDataReader(os.path.join(p, f))
        slurm_dict["runs"].append(_tmp_slurm_data.read(verbose=True))
        # exit("Success! @ 182 in scaling_analysis")

    json_fpath = os.path.basename(p)
    json_fpath = json_fpath + ".json"
    with file(json_fpath, "w+") as json_file:
        json.dump(json_dict, json_file, indent=4)
    print "json data dumped to {}".format(json_fpath)

if __name__ == '__main__':
    scaling_analysis()
