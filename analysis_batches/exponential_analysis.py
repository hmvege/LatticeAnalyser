import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import os

rc("text", usetex=True)
rcParams["font.family"] += ["serif"]



def get_val(_f):
    """Retrieves single line value."""
    return float(_f.readline().split(" ")[-1].rstrip("\n"))


def analyse_precision(input_file, output_folder, figure_folder):
    """Create table + plot of precision."""
    assert os.path.isfile(input_file), (
        "Missing file: {}".format(input_file))

    data = {}

    with open(input_file, "r") as f:
        data["Luscher"] = get_val(f)
        data["Morningstar"] = get_val(f)
        data["Taylor2"] = get_val(f)
        data["Taylor4"] = get_val(f)
        data["header"] = [r"$N$",
                          r"$\mathrm{Taylor}(N)$",
                          r"abs$(\mathrm{Luscher}-\mathrm{Taylor}(N))$",
                          r"abs$(\mathrm{Morningstar}-\mathrm{Taylor}(N))$",
                          r"rel$(\mathrm{Morningstar}-\mathrm{Taylor}(N))$",
                          r"abs$(\mathrm{Taylor}(2)-\mathrm{Taylor}(N))$",
                          r"abs$(\mathrm{Taylor}(4)-\mathrm{Taylor}(N))$",
                          r"abs$(\mathrm{Taylor}(N)-\mathrm{Taylor}(16))$",
                          r"rel$(\mathrm{Taylor}(N)-\mathrm{Taylor}(16))$"]

        # Reads in data
        data["data"] = []
        f.readline()
        for l in f:
            data["data"].append(
                list(map(float, [i.rstrip("\n") for i in l.split(" ")])))
        data["data"] = np.asarray(data["data"]).T

    # Save to .dat file for pgf tables
    output_data_file = os.path.join(output_folder, "precision.dat")
    np.savetxt(output_data_file, data["data"],
               header=" ".join(data["header"]))

    # Create plot
    fig, ax = plt.subplots(1, 1, sharey=False, sharex=True)
    ax.semilogy(data["data"][0], data["data"][3], "->", color="#1f78b4", label=data["header"][3])
    ax.semilogy(data["data"][0], data["data"][2], "-x", color="#33a02c", label=data["header"][2])
    ax.semilogy(data["data"][0], data["data"][8], "-o", color="#e31a1c", label=data["header"][7])
    ax.grid(True)
    ax.set_xlabel(r"Taylor polynomial degree $N$")
    ax.set_ylabel(r"Absolute difference")
    ax.legend()

    figname = os.path.join(figure_folder, "precision.pdf")
    fig.savefig(figname)
    plt.show()
    plt.close(fig)



def analyse_times(input_file, output_folder, figure_folder):
    """Create table + plot of times."""
    assert os.path.isfile(input_file), (
        "Missing file: {}".format(input_file))

    data = {}

    with open(input_file, "r") as f:
        data["Luscher"] = get_val(f)
        data["Morningstar"] = get_val(f)
        data["Taylor2"] = get_val(f)
        data["Taylor4"] = get_val(f)
        data["header"] = [r"$N$",
                          r"$\mathrm{Taylor}(N)$",
                          r"$\mathrm{Taylor}(N)/\mathrm{Morningstar}$"]

        # Reads in data
        data["data"] = []
        f.readline()
        for l in f:
            data["data"].append(
                list(map(float, [i.rstrip("\n") for i in l.split(" ")])))
        data["data"] = np.asarray(data["data"]).T

    # Save to .dat file for pgf tables
    output_data_file = os.path.join(output_folder, "timing.dat")
    np.savetxt(output_data_file, data["data"],
               header=" ".join(data["header"]))

    # Create plot
    fig, axes = plt.subplots(2, 1, sharey=False, sharex=True)
    axes[0].plot(data["data"][0], data["data"][1], color="#1f78b4", label=data["header"][1])
    axes[1].axhline(1.0, linestyle="--", alpha=1.0, color="#a6cee3")
    axes[1].semilogy(data["data"][0], data["data"][2], color="#1f78b4", label=data["header"][2])
    for ax in axes:
        ax.grid(True)
        # ax.legend()
    axes[0].set_ylabel(r"Taylor$(N)$ [seconds]")
    axes[1].set_ylabel(r"Taylor$(N)$/Morningstar")
    axes[1].set_xlabel(r"Taylor polynomial degree $N$")

    figname = os.path.join(figure_folder, "timing.pdf")
    fig.savefig(figname)
    plt.show()
    plt.close(fig)



def main():
    # Read in data
    base_folder = "/Users/hansmathiasmamenvege/Programming/LQCD"
    input_folder = "GluonAction/output/defaultPerformanceRun/observables"
    precision_data_file = os.path.join(
        base_folder, input_folder, "exp_precision.dat")
    timing_data_file = os.path.join(
        base_folder, input_folder, "exp_timing.dat")

    # Sets up output folders
    analysis_name = "exponential_analysis"
    exp_analysis_output_folder = os.path.join(
        base_folder, "data", analysis_name)
    figure_folder = os.path.join(base_folder, "LatticeAnalyser",
                                 "figures", analysis_name)

    if not os.path.isdir(exp_analysis_output_folder):
        os.mkdir(exp_analysis_output_folder)
        print("Creating folder at {}".format(exp_analysis_output_folder))

    if not os.path.isdir(figure_folder):
        os.mkdir(figure_folder)
        print("Creating folder at {}".format(figure_folder))

    analyse_precision(precision_data_file,
                      exp_analysis_output_folder, figure_folder)

    analyse_times(timing_data_file, exp_analysis_output_folder, figure_folder)


if __name__ == '__main__':
    main()
