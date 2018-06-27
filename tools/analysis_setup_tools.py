from collections import OrderedDict
from table_printer import TablePrinter
import tools.sciprint as sciprint
import numpy as np
import types


def _check_splits(N, numsplits):
    """Checks if the temporal dimension has been split into good .intervals"""
    if N % numsplits != 0:
        print ("Number of splits not even: N %% "
            "numsplits = %d %% %d = %d." % (N, numsplits, N % numsplits))


def _check_intervals(intervals, numsplits):
    """Sets up the intervals"""
    # if (intervals == numsplits == None) or \
    #     (intervals != None and numsplits != None):
    if (intervals == numsplits == None):

        raise ValueError(("Either provide MC intervals to plot for or the "
            "number of MC intervals to split into."))


def interval_setup(beta_param_list, int_type):
    """
    Sets up the intervals to analyse for in Monte Carlo or Euclidean time.

    Args:
        beta_param_list: list of beta dictionary parameters. List assumed to 
            be ordered for the different beta values.
        int_type: str, either "MC" or "Eucl"

    Returns:
        list of dictionaries for the different beta values containing list 
            of interval tuples.
    """

    _create_int = lambda l: "-".join(["%03d" % i for i in l])
    N_betas = len(beta_param_list)

    if int_type == "Eucl":
        NTot_arg = "NT"
        numsplits_arg = "numsplits_eucl"
        intervals_arg = "intervals_eucl"
    else:
        NTot_arg = "NCfgs"
        numsplits_arg = "MC_time_splits"
        intervals_arg = "MCInt"

    # If we have provided exact intervals, will create a dictionary
    if not isinstance(beta_param_list[0][intervals_arg], types.NoneType):
        interval_list = [_create_int(beta_param_list[ib][intervals_arg])
            for ib in range(N_betas)]
        interval_list = [interval_list]
    else:
        _temp = []
        for plist in beta_param_list:
            _temp.append([_create_int(i) for i in get_intervals(
                            plist[NTot_arg], 
                            numsplits=plist[numsplits_arg], 
                            intervals=plist[intervals_arg])[0]])

        _num_splits = beta_param_list[0][numsplits_arg]
        interval_list = np.asarray(_temp).T

    return interval_list


def get_intervals(N, numsplits=None, intervals=None):
    """
    Method for retrieving the monte carlo time intervals.

    Args:
        N: int, number of points.
        numsplits: int, optional, number of splits to make in N.
        intervals: int, optional, excact intervals to make in N.

    Returns:
        List of intervals, list of tuples
        Size of interval, int

    Raises:
        ValueError: if no intervals or numsplits is provided, or if both is
            provided.
    """
    _check_intervals(intervals, numsplits)

    if isinstance(intervals, types.NoneType):
        split_interval = N/numsplits
        intervals = zip(
            range(0, N+1, split_interval), 
            range(split_interval, N+1, split_interval)
        )
        _check_splits(N, numsplits)

    if isinstance(intervals[0], (list, tuple)):
        # If we have a list of interval tuples/lists
        return intervals, intervals[0][1] - intervals[0][0]
    else:
        return [tuple(intervals)], intervals[1] - intervals[0]


def append_fit_params(fplist, obs_name, analysis_name, fparams):
    """Function for appending fit parameters."""
    chi_squared, fit_params, topsus, topsus_err, N_F, N_F_err, \
        fit_target, interval, descr, extrap_method, obs_name_latex = fparams
    fplist.append({
        "observable_type": obs_name,
        "descr": descr,
        "extrap_method": extrap_method,
        "analysis_type": analysis_name,
        "fit_target": fit_target,
        "chi_squared": chi_squared,
        "a": fit_params[2],
        "a_err": fit_params[3],
        "b": fit_params[0],
        "b_err": fit_params[1],
        "topsus": topsus,
        "topsus_err": topsus_err,
        "N_F": N_F,
        "N_F_err": N_F_err,
        "interval": interval,
        "obs_name_latex": obs_name_latex,
    })
    return fplist


def write_fit_parameters_to_file(fparams, fname, skip_values=None, 
    verbose=False, verbose_latex=False):
    """Function for writing fit parameters to file."""
    with open(fname, "w") as f:
        sorted_parameter_list = sorted(fparams, key=lambda d: \
            (d["fit_target"], d["analysis_type"]))

        # Default float width
        fw = 14
        dict_keys = OrderedDict([
            ("observable_type", {"name": "obs", "w": 14, "type": "s"}),
            ("descr", {"name": "description", "w": 35, "type": "s"}),
            ("fit_target", {"name": "sqrt(8t_0)", "w": 11, "type": ".2f"}),
            ("extrap_method", {"name": "extrap.-method", "w": 15, "type": "s"}),
            ("interval", {"name": "interval/slice", "w": 60, "type": "s"}),
            ("analysis_type", {"name": "atype", "w": 12, "type": "s"}),
            ("chi_squared", {"name": "Chi^2", "w": 25, "type": ".8f"}),
            ("a", {"name": "a", "w": fw, "type": ".8f"}),
            ("a_err", {"name": "aerr", "w": fw, "type": ".8f"}),
            ("b", {"name": "b", "w": fw, "type": ".8f"}),
            ("b_err", {"name": "berr", "w": fw, "type": ".8f"}),
            ("topsus", {"name": "topsus", "w": fw, "type": ".8f"}),
            ("topsus_err", {"name": "topsuserr", "w": fw, "type": ".8f"}),
            ("N_F", {"name": "N_F", "w": fw, "type": ".8f"}),
            ("N_F_err", {"name": "N_F_err", "w": fw, "type": ".8f"}),
        ])

        # Sets header in text file
        header_string = ""
        create_str = lambda _val, _width, _fcode: "{0:<{w}{t}}".format(
            _val, w=_width, t=_fcode)
        for k in dict_keys.items():
            if not k[0] in skip_values:
                header_string += create_str(k[-1]["name"], k[-1]["w"], "s")
        if verbose: 
            print header_string
        f.write(header_string + "\n")

        # Writes out analysis values to text file
        for fp in sorted_parameter_list:
            line_values = ""
            for k in dict_keys.items():
                if not k[0] in skip_values:
                    line_values += create_str(fp[k[0]], k[-1]["w"],
                        k[-1]["type"])
            if verbose:
                print line_values
            f.write(line_values + "\n")

        # Obs  sqrt(8t)  extrap.method  int/slice  chi^2  topsus  Nf
        table_header = [r"$\mathcal{O}$", r"$\sqrt{8t_{f,0,\text{extrap}}}$", 
            "Extrap. method", "Interval/slice", r"$\chi^2$", 
            r"$\chi^{\frac{1}{4}}$", r"$N_F$"]
        table_body = [
            [fp["obs_name_latex"] for fp in sorted_parameter_list],
            [fp["fit_target"] for fp in sorted_parameter_list],
            [fp["extrap_method"] for fp in sorted_parameter_list],
            [r"{:s}".format(fp["interval"]) for fp in sorted_parameter_list],
            [r"{:.2g}".format(fp["chi_squared"]) for fp in sorted_parameter_list],
            [sciprint.sciprint(fp["topsus"], fp["topsus_err"], prec=4) for fp in sorted_parameter_list],
            [sciprint.sciprint(fp["N_F"], fp["N_F_err"], prec=4) for fp in sorted_parameter_list],
        ]

        width_list = [len(tab)+2 for tab in table_header]
        width_list[0] = 45
        width_list[3] = 30
        topsus_table = TablePrinter(table_header, table_body)
        topsus_table.print_table(width=width_list, ignore_latex_cols=[2])


def main():
    exit("%s Not intended as standalone module." % __name__)


if __name__ == '__main__':
    main()