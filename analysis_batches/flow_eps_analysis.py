import numpy as np
import os
import copy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
    from matplotlib import rc, rcParams
    
    # For zooming in on particular part of plot
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes



rc("text", usetex=True)
rcParams["font.family"] += ["serif"]


def rm_hidden(l):
    """Removes all hidden files in list l."""
    return [f for f in l if not f.startswith(".")]


def get_max_x_limit(feps_dictionary, observable):
    """Gets the maximal x limit from smallest epsilon."""
    lowest_eps = np.min(feps_dictionary.keys())
    x_limit = np.max(feps_dictionary[lowest_eps]["data"][observable][:, 0])
    return x_limit, lowest_eps


def get_nearest(x0, x, y):
    """Gets the value closest to a given x0."""
    fit_index = np.argmin(np.abs(x - x0))
    return x[fit_index], y[fit_index]


def get_data(p):
    """
    Takes argument path, and retrieves all data found under that folder path.
    """
    obs_dict = {}
    for obs in rm_hidden(os.listdir(p)):
        obs_path = os.path.join(p, obs)

        # Retrieves observable files with full path
        obs_file_names = rm_hidden(os.listdir(obs_path))
        obs_files = [os.path.join(obs_path, f) for f in obs_file_names]
        assert len(obs_files) == 1, "More than one observable file found."

        obs_dict[obs] = np.loadtxt(obs_files[0], skiprows=3)
    return obs_dict


def observable_plotter(feps_dictionary, observable, x_limit=0,
                       x_label=r"$t_f$", y_label="", savefig_folder="",
                       exclude_eps=[], fig_format="pdf", zoom_factor=250):
    """Method for plotting an observable."""
    feps_dict = copy.deepcopy(feps_dictionary)
    if not os.path.isdir(savefig_folder):
        os.mkdir(savefig_folder)
        print "> mkdir {0:s}".format(savefig_folder)

    if x_limit == 0:
        x_limit, min_eps = get_max_x_limit(feps_dict, observable)

    fig_name = os.path.join(savefig_folder,
                            "{0:s}.{1:s}".format(observable, fig_format))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Sets up zoom box
    if observable == "energy":
        axins = zoomed_inset_axes(ax, zoom_factor, loc=2)

    # Plots epsilon values
    for feps in sorted(feps_dict):
        if observable == "energy":
            a = 0.09314041
            r0 = 0.5
            feps_dict[feps]["data"][observable][:, 1] *= (-1)
            feps_dict[feps]["data"][observable][:, 0] *= (a**2)/(r0**2)
            feps_dict[feps]["data"][observable][:, 1] *= \
                (feps_dict[feps]["data"][observable][:, 0]**2)
            # feps_dict[feps]["data"][observable][:,0] /= (a**2)

        xvals, yvals = feps_dict[feps]["data"][observable].T
        ls = feps_dict[feps]["ls"][-1]
        mk = feps_dict[feps]["marker"][-1]
        label = r"$\epsilon_f = {:<.{w}f}$".format(
            float(feps), w=len(str(feps)) - 2)

        xvals = filter(lambda x: x < x_limit, xvals)
        yvals = yvals[:len(xvals)]

        ax.plot(xvals, yvals, label=label, linestyle=ls)

        # Plots zoom box
        if observable == "energy":
            axins.plot(xvals, yvals, linestyle=ls)

    ax.legend(loc="lower right", ncol=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    if observable == "energy":
        ax.set_xlim([-0.02, 1.0])
        ax.set_ylim([-0.01, 0.35])
    else:
        ax.set_xlim([-0.05, x_limit*1.05])

    # Finalizes zoom box
    if observable == "energy":
        axins.set_xlim([-0.00010, 0.002])
        axins.set_ylim([-0.00005, 0.0006])
        # axins.grid(True)
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

    plt.savefig(fig_name, dpi=400)

    print "Figure saved at: {0:s}.\nPlotting done for {1:s}.".format(
        fig_name, observable)


def get_differences(feps_dictionary, observable, diff="relative", x_limit=0,
                    x_label=r"$t_f$", y_label="", savefig_folder="",
                    exclude_eps=[], fig_format="pdf"):
    """Method for looking at the relative differences between the epsilons."""
    feps_dict = copy.deepcopy(feps_dictionary)
    for _eps in exclude_eps:
        if _eps in feps_dict:
            del feps_dict[_eps]

    for feps in sorted(feps_dict):
        if observable == "energy":
            a = 0.09314041
            r0 = 0.5
            feps_dict[feps]["data"][observable][:, 1] *= (-1)
            feps_dict[feps]["data"][observable][:, 0] *= (a**2)/(r0**2)
            feps_dict[feps]["data"][observable][:, 1] *= \
                (feps_dict[feps]["data"][observable][:, 0]**2)
            # feps_dict[feps]["data"][observable][:,0] /= (a**2)

    min_eps = np.min(feps_dict.keys())
    max_eps = np.max(feps_dict.keys())
    x_min = feps_dict[min_eps]["data"][observable][:, 0]
    y_min = feps_dict[min_eps]["data"][observable][:, 1]
    x_max = feps_dict[max_eps]["data"][observable][:, 0]
    y_max = feps_dict[max_eps]["data"][observable][:, 1]

    # Dictionary to populate with epsilon values, x-axis values and obs values
    plot_dict = {}

    # Smalls epsilon value to add in order to include edge points when
    # selecting points at the edge
    sel_eps = 1e-4

    # Retrieves eps values close to eps-max points
    for feps in sorted(feps_dict)[:-1]:

        # Sets up empty lists for populating with value to compare.
        _xvals = []
        _yvals = []

        # Loops over values to select from in max-eps.
        # Ensures all values we can plot are less than x-max.
        for _x_index in np.where(x_max < (x_min[-1] + sel_eps))[0][1:]:

            # Selects values closest to each other for max-eps-value and 
            # eps-values.
            _close_values = \
                feps_dict[feps]["data"][observable][:, 0] - x_max[_x_index]
            _closest_index = int(np.argmin(np.abs(_close_values)))

            # Retrieves values that can be compared from other epsilon value
            _xvals.append(
                feps_dict[feps]["data"][observable][:, 0][_closest_index])
            _yvals.append(
                feps_dict[feps]["data"][observable][:, 1][_closest_index])

        plot_dict[feps] = [np.array(_xvals), np.array(_yvals)]

    # Retrieves values for max epsilon
    _xmax_indices = np.where(x_max < (x_min[-1] + sel_eps))[0][1:]
    plot_dict[sorted(feps_dict.keys())[-1]] = [
        x_max[_xmax_indices], y_max[_xmax_indices]]

    # Gets values to plot
    for feps in sorted(feps_dict)[1:]:
        xvals, yvals = plot_dict[feps]
        _x, _y = [], []
        for imin, _xmin in enumerate(x_min):
            for iv, _xv in enumerate(xvals):
                eps = min_eps
                if observable == "energy":
                    # The energy uses a different xaxis scale
                    eps = 1e-7

                # Only picks values which are close to each other on x-axis
                if np.abs(_xv - _xmin) < eps:
                    _x.append(_xv)
                    if diff == "absolute":
                        _y.append(np.abs(y_min[imin] - yvals[iv]))
                    elif diff == "relative":
                        _y.append(
                            np.abs((y_min[imin] - yvals[iv])/y_min[imin]))
                    else:
                        raise KeyError(
                            ("%s is not a recognized method "
                             "of difference comparison." % diff.capitalize()))
                    continue

        # Skips first element, as that tends to be zero, and may obfuscate
        # the plot.
        plot_dict[feps] = [np.asarray(_x), np.asarray(_y)]

    fig_name = os.path.join(savefig_folder,
                            "{0:s}_diff_{1:s}.{2:s}".format(observable,
                                                            diff, fig_format))
    if diff == "absolute":
        title = r"Absolute difference, $|${%s$_{\epsilon_{min}} - $" % y_label
        title += r"%s$_{\epsilon_{i}}|$" % y_label
    else:
        title = r"Relative difference, $|${%s$_{\epsilon_{min}} - $" % y_label
        title += r"%s$_{\epsilon_{i}}|/$%s$_{\epsilon_{min}}$" % (
            y_label, y_label)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plots the values found
    feps_keys = sorted(feps_dict.keys())[1:]
    for feps in feps_keys:
        x_plot_values = plot_dict[feps][0]
        y_plot_values = plot_dict[feps][1]

        if np.abs(np.min(y_plot_values)) <= 1e-15:
            x_plot_values = filter(lambda x: np.abs(x) > 1e-15, x_plot_values)
            y_plot_values = filter(lambda x: np.abs(x) > 1e-15, y_plot_values)

        ls = feps_dict[feps]["ls"][-1]
        marker = feps_dict[feps]["marker"]
        label = r"$\epsilon_f = {:<.{w}f}$".format(
            float(feps), w=len(str(feps)) - 2)
        # print plot_dict[feps][0][:5], plot_dict[feps][1][:5]
        ax.semilogy(x_plot_values, y_plot_values, label=label,
                    linestyle=ls, marker=marker)

    ax.legend(loc="upper center", ncol=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    # ax.set_title(title)
    if observable != "energy":
        ax.set_xlim([-0.05, x_limit*1.05])

    plt.savefig(fig_name, dpi=400)
    plt.close(fig)
    print "Figure saved at: {0:s}.\nPlotting done for {1:s}.".format(
        fig_name, observable)


def main():
    # Flow epsilon main folder location
    feps_main_folder = "../../data/flow_eps/flow_eps_data"

    # Distribution sub folder locations
    feps_folders = rm_hidden(sorted(os.listdir(feps_main_folder)))
    feps_data_paths = [os.path.join(feps_main_folder, f, "flow_observables")
                       for f in feps_folders]

    feps_values = [float(".".join(f.split("_")[-2:])) for f in feps_folders]
    feps_values = sorted(feps_values)
    feps_dict = {feps: {"data": get_data(feps_data_paths[i])}
                 for i, feps in enumerate(feps_values)}

    observables = {
        "plaq": {
            "x": r"$t_f/a^2$",
            "y": r"$\langle P \rangle$",
            "y_abs": r"$\mathrm{err}\big(\langle P \rangle\big)_\mathrm{abs}$",
            "y_rel":
                r"$\mathrm{err}\big(\langle P \rangle\big)_\mathrm{rel}$"},
        "energy": {
            "x": r"$t_f/a^2$",
            "y": r"$\langle E \rangle$",
            "y_abs": r"$\mathrm{err}\big(\langle E \rangle\big)_\mathrm{abs}$",
            "y_rel":
                r"$\mathrm{err}\big(\langle E \rangle\big)_\mathrm{rel}$"},
        "topc": {
            "x": r"$t_f/a^2$",
            "y": r"$\langle P \rangle$",
            "y_abs": r"$\mathrm{err}\big(\langle Q \rangle\big)_\mathrm{abs}$",
            "y_rel":
                r"$\mathrm{err}\big(\langle Q \rangle\big)_\mathrm{rel}$"},
        # "topct": {"x": r"$t_f/a^2$", "y": r"$\langle Q(t_e) \rangle$"},
    }

    exclude_eps = []

    # Sets up linestyles
    linestyles = [
        ('solid',               (0, ())),
        ('loosely dotted',      (0, (1, 10))),
        ('dotted',              (0, (1, 5))),
        ('densely dotted',      (0, (1, 1))),

        ('loosely dashed',      (0, (5, 10))),
        ('dashed',              (0, (5, 5))),
        ('densely dashed',      (0, (5, 1))),

        ('loosely dashdotted',  (0, (3, 10, 1, 10))),
        ('dashdotted',          (0, (3, 5, 1, 5))),
        ('densely dashdotted',  (0, (3, 1, 1, 1))),

        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]

    markers = [".", "2", "o", "v", "^", "D", "h", "x", "8", "s", "p", "P", "*"]

    for feps, ls, marker in zip(feps_values, linestyles, markers):
        feps_dict[feps]["ls"] = ls
        feps_dict[feps]["marker"] = marker

    for obs in observables:

        observable_plotter(feps_dict, obs, x_limit=1.0,
                           x_label=observables[obs]["x"],
                           y_label=observables[obs]["y"],
                           savefig_folder="../figures/flow_epsilon_test_figures",
                           exclude_eps=exclude_eps, fig_format="pdf")

        get_differences(feps_dict, obs, x_limit=1.0, diff="absolute",
                        x_label=observables[obs]["x"],
                        y_label=observables[obs]["y_abs"],
                        savefig_folder="../figures/flow_epsilon_test_figures",
                        exclude_eps=exclude_eps)

        get_differences(feps_dict, obs, x_limit=0, diff="relative",
                        x_label=observables[obs]["x"],
                        y_label=observables[obs]["y_rel"],
                        savefig_folder="../figures/flow_epsilon_test_figures",
                        exclude_eps=exclude_eps)


if __name__ == '__main__':
    main()
