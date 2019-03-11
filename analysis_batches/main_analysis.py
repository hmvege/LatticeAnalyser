#!/usr/bin/env python2


def main_analysis(run_pre_analysis=True, run_post_analysis=True):
    from pre_analysis.pre_analyser import pre_analysis
    from post_analysis.post_analyser import post_analysis
    from default_analysis_params import get_default_parameters
    from tools.folderreadingtools import get_num_observables
    import copy
    import os

    # Different batches
    # data_batch_folder = "../GluonAction/data8"
    data_batch_folder = "../GluonAction/data11"
    # data_batch_folder = "../GluonAction/DataGiovanni"
    # data_batch_folder = "../data/topc_modes_8x16"
    data_batch_folder = "../data/data11"

    obs_exlusions = ["w_t_energy", "energy", "topcMC", "topsusMC", "qtq0effMC"]
    default_params = get_default_parameters(
        data_batch_folder=data_batch_folder,
        obs_exlusions=obs_exlusions)

    # observables = observables_euclidean_time
    # , "topsust", "topsuste", "topsusMC", "topsusqtq0"]
    observables = ["topsus"]
    # observables = ["energy"]
    # observables = ["w_t_energy"]
    observables = ["plaq", "energy", "topc", "topsus", "topc2", "topc4", "qtq0eff"]

    default_params["observables"] = observables

    # Post analysis parameters
    line_fit_interval_points = 20
    # topsus_fit_targets = [0.3,0.4,0.5,0.58]
    # topsus_fit_targets = [0.3, 0.4, 0.5, 0.6] # tf = sqrt(8*t0)
    topsus_fit_targets = [0.6, "t0", "w0", "t0cont", "w0cont"]
    energy_fit_target = 0.3

    # Will try to use pickled reference scale instead
    use_pickled_reference_scale = True

    # Method of continuum extrapolation.
    # Options: plateau, plateau_mean, nearest, interpolate, bootstrap
    extrapolation_methods = ["plateau", "plateau_mean", "nearest",
                             "interpolate", "bootstrap"]
    # extrapolation_methods = ["plateau"]
    extrapolation_methods = ["bootstrap"]
    plot_continuum_fit = False

    # Number of different sectors we will analyse in euclidean time
    default_params["numsplits_eucl"] = 4
    intervals_eucl = [None, None, None, None]

    # Number of different sectors we will analyse in monte carlo time
    default_params["MC_time_splits"] = 4
    # MC_intervals = [[0, 1000], [500, 1000], [500, 1000], [175, 250]]
    # MC_intervals = [[0, 1000], [0, 1000], [0, 2000], [0, 125]]
    MC_intervals = [None, None, None, None]

    # Extraction point in flow time a*t_f for q0 in qtq0
    q0_flow_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # [fermi]

    # Flow time indexes in percent to plot qtq0 in euclidean time at
    euclidean_time_percents = [0, 0.25, 0.50, 0.75, 1.00]
    # euclidean_time_percents = [0]

    # Data types to be looked at in the post-analysis.
    post_analysis_data_type = [
        "bootstrap", "jackknife", "unanalyzed", "blocked", "blocked_bs"]
    post_analysis_data_type = ["bootstrap"]

    # Blocking
    default_params["blocking_analysis"] = True

    # Check to only generate data for post-analysis
    default_params["only_generate_data"] = True

    ########## Main analysis ##########
    databeta60 = copy.deepcopy(default_params)
    databeta60["batch_name"] = "beta60"
    databeta60["beta"] = 6.0
    databeta60["block_size"] = 10  # None
    databeta60["topc_y_limits"] = [-9, 9]
    databeta60["topc2_y_limits"] = [-81, 81]
    databeta60["NCfgs"] = get_num_observables(
        databeta60["batch_folder"],
        databeta60["batch_name"])
    databeta60["obs_file"] = "24_6.00"
    databeta60["MCInt"] = MC_intervals[0]
    databeta60["N"] = 24
    databeta60["NT"] = 2*databeta60["N"]
    databeta60["color"] = "#e41a1c"

    databeta61 = copy.deepcopy(default_params)
    databeta61["batch_name"] = "beta61"
    databeta61["beta"] = 6.1
    databeta61["block_size"] = 10  # None
    databeta61["topc_y_limits"] = [-12, 12]
    databeta61["topc2_y_limits"] = [-144, 144]
    databeta61["NCfgs"] = get_num_observables(
        databeta61["batch_folder"],
        databeta61["batch_name"])
    databeta61["obs_file"] = "28_6.10"
    databeta61["MCInt"] = MC_intervals[1]
    databeta61["N"] = 28
    databeta61["NT"] = 2*databeta61["N"]
    databeta61["color"] = "#377eb8"

    databeta62 = copy.deepcopy(default_params)
    databeta62["batch_name"] = "beta62"
    databeta62["beta"] = 6.2
    databeta62["block_size"] = 10  # None
    databeta62["topc_y_limits"] = [-12, 12]
    databeta62["topc2_y_limits"] = [-196, 196]
    databeta62["NCfgs"] = get_num_observables(
        databeta62["batch_folder"],
        databeta62["batch_name"])
    databeta62["obs_file"] = "32_6.20"
    databeta62["MCInt"] = MC_intervals[2]
    databeta62["N"] = 32
    databeta62["NT"] = 2*databeta62["N"]
    databeta62["color"] = "#4daf4a"

    databeta645 = copy.deepcopy(default_params)
    databeta645["flow_epsilon"] = 0.02
    databeta645["batch_name"] = "beta645"
    databeta645["beta"] = 6.45
    databeta645["block_size"] = 25  # None
    databeta645["topc_y_limits"] = [-15, 15]
    databeta645["topc2_y_limits"] = [-300, 300]
    databeta645["NCfgs"] = get_num_observables(
        databeta645["batch_folder"],
        databeta645["batch_name"])
    databeta645["obs_file"] = "48_6.45"
    databeta645["MCInt"] = MC_intervals[3]
    databeta645["N"] = 48
    databeta645["NT"] = 2*databeta645["N"]
    databeta645["color"] = "#984ea3"

    # Adding relevant batches to args
    analysis_parameter_list = [databeta60, databeta61, databeta62, databeta645]
    # analysis_parameter_list = [databeta645]

    section_seperator = "="*160
    print section_seperator
    print "Observables to be analysed: %s" % ", ".join(
        default_params["observables"])
    print section_seperator + "\n"

    # Submitting main analysis
    if run_pre_analysis:
        for analysis_parameters in analysis_parameter_list:
            pre_analysis(analysis_parameters)

    if not analysis_parameter_list[0]["MCInt"] is None:
        assert sum(
            [len(plist["MCInt"]) - len(analysis_parameter_list[0]["MCInt"])
             for plist in analysis_parameter_list]) == 0, \
            "unequal amount of MC intervals"

    # Submitting post-analysis data
    if run_post_analysis:
        if len(analysis_parameter_list) >= 3:
            post_analysis(analysis_parameter_list,
                          default_params["observables"],
                          topsus_fit_targets, line_fit_interval_points,
                          energy_fit_target,
                          q0_flow_times, euclidean_time_percents,
                          extrapolation_methods=extrapolation_methods,
                          plot_continuum_fit=plot_continuum_fit,
                          post_analysis_data_type=post_analysis_data_type,
                          figures_folder=default_params["figures_folder"],
                          gif_params=default_params["gif"],
                          verbose=default_params["verbose"])
        else:
            msg = "Need at least 3 different beta values to run post analysis"
            msg += "(%d given)." % len(analysis_parameter_list)
            print msg


if __name__ == '__main__':
    main_analysis()
