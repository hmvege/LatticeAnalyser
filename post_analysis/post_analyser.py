from observable_analysis import *
from tools.postanalysisdatareader import PostAnalysisDataReader
from tools.analysis_setup_tools import append_fit_params, \
    write_fit_parameters_to_file, get_intervals, interval_setup, \
    save_pickle, load_pickle
import types
import numpy as np
import os
from tqdm import tqdm


def default_post_analysis(PostAnalysis, data, figures_folder, analysis_type,
                          verbose=False):
    analysis = PostAnalysis(data, figures_folder=figures_folder,
                            verbose=verbose)
    analysis.set_analysis_data_type(analysis_type)
    print analysis
    analysis.plot()


def default_interval_pa(*args, **kwargs):
    pass


def default_slice_pa(*args, **kwargs):
    pass


def plaq_post_analysis(*args, **kwargs):
    default_post_analysis(PlaqPostAnalysis, *args, **kwargs)


def energy_post_analysis(*args, **kwargs):
    default_post_analysis(PlaqPostAnalysis, *args, **kwargs)


def post_analysis(batch_parameter_list, observables, topsus_fit_targets,
                  line_fit_interval_points, energy_fit_target, q0_flow_times,
                  euclidean_time_percents, extrapolation_methods="nearest",
                  plot_continuum_fit=False, figures_folder="figures",
                  post_analysis_data_type=[
                      "bootstrap", "jackknife", "unanalyzed"],
                  topcr_tf="t0beta", gif_params=None, verbose=False):
    """
    Post analysis of the flow observables.

    Args: 
            batch_parameter_list: list of dicts, batch parameters.
            observables: list of str, which observables to plot for
            topsus_fit_targets: list of x-axis points to line fit at.
            line_fit_interval_points: int, number of points which we will use in 
                    line fit.
            energy_fit_target: point of which we will perform a line fit at.
            q0_flow_times: points where we perform qtq0 at.
            euclidean_time_percents: points where we perform qtq0e at.
            extrapolation_methods: list of str, optional, extrapolation methods 
                    to use for selecting topsus_fit_target point. Default is "nearest".
            plot_continuum_fit: bool, optional. If we are to plot the point of 
                    topsus_fit_target extraction. Default is False.
            figures_folder: str, optional. Where to place figures folder. Default 
                    is "figures".
            post_analysis_data_type: list of str, what type of data to use in the 
                    post analysis. Default is ["bootstrap", "jackknife", "unanalyzed"].
            gif_params: dict, parameters to use in gif creation. Default is None.
                    dict = { 
                            "gif_observables": [list_of_observables],
                            "gif_euclidean_time": euclidean_time_to_plot_at, 
                            "gif_flow_range": [range of floats to plot for],
                            "betas_to_plot": "all",
                            "plot_together": False,
                            "error_shape": "band"}
            verbose: bool, a more verbose run. Default is False.
    """

    section_seperator = "="*160
    print section_seperator
    print "Post-analysis: retrieving data from folders: %s" % (
        ", ".join([os.path.join(b["batch_folder"], b["batch_name"])
                   for b in batch_parameter_list]))

    # Topcr requires a few more observables to be fully utilized.
    old_obs = observables
    if "topcr" in observables:
        observables += ["topc2", "topc4"]
    if "topcrMC" in observables:
        observables += ["topc2MC", "topc4MC"]

    data = PostAnalysisDataReader(batch_parameter_list,
                                  observables_to_load=observables)

    # Resets to the old observables, as not to analyze topc2 and topc4.
    observables = old_obs

    fit_parameters = []
    reference_scale = {
        extrap_method: {atype: {} for atype in post_analysis_data_type}
        for extrap_method in extrapolation_methods
    }

    if "energy" in observables:

        energy_analysis = EnergyPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        for extrapolation_method in extrapolation_methods:

            for analysis_type in post_analysis_data_type:
                energy_analysis.set_analysis_data_type(analysis_type)

                print energy_analysis
                if verbose:
                    print "Energy extrapolation method: ", extrapolation_method
                    print "Energy analysis type: ", analysis_type

                energy_analysis.plot_autocorrelation()

                # t_0 analysis
                energy_analysis.plot()
                energy_analysis.plot(x_limits=[-0.015, 0.15],
                                     y_limits=[-0.025, 0.4], plot_hline_at=0.3,
                                     figure_name_appendix="_zoomed",
                                     zoom_box={"xlim": [0.1104, 0.1115],
                                               "ylim": [0.298, 0.302],
                                               "zoom_factor": 50})

                t0_dict = energy_analysis.get_t0_scale(
                    extrapolation_method=extrapolation_method,
                    E0=energy_fit_target, plot_fit=False)


                # w_0 analysis
                box_settings = {
                    "xlim": [0.111, 0.117],
                    "ylim": [0.295, 0.305],
                    "zoom_factor": 10}

                energy_analysis.plot_w(figure_name_appendix="_w_scale")
                energy_analysis.plot_w(x_limits=[-0.015, 0.15],
                                       y_limits=[-0.025, 0.4],
                                       plot_hline_at=0.3,
                                       figure_name_appendix="_w_scale_zoomed",
                                       zoom_box=box_settings)

                w0_dict = energy_analysis.get_w0_scale(
                    extrapolation_method=extrapolation_method,
                    W0=energy_fit_target, plot_fit=False)

                # Updates reference dictionaries
                reference_scale[extrapolation_method][analysis_type] = \
                    t0_dict

                for k in w0_dict:
                	if isinstance(w0_dict[k], dict):
                		reference_scale[extrapolation_method][
                			analysis_type][k].update(w0_dict[k])
                	else:
                		reference_scale[extrapolation_method][
                			analysis_type][k] = w0_dict[k]

                # # Retrofits the energy for continiuum limit
                # energy_analysis.plot_continuum(0.3, 0.015,
                # 	extrapolation_method=extrapolation_method)

                # # Plot running coupling
                # energy_analysis.coupling_fit()

        for t_flow in q0_flow_times:
            energy_analysis.plot_autocorrelation_at(target_flow=t_flow)
            energy_analysis.plot_mc_history_at(target_flow=t_flow)

    else:
        reference_scale = None

    if "w_t_energy" in observables:

        w_t_energy_analysis = WtPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        for extrapolation_method in extrapolation_methods:

            for analysis_type in post_analysis_data_type:
                w_t_energy_analysis.set_analysis_data_type(analysis_type)

                print w_t_energy_analysis

                w_t_energy_analysis.plot_autocorrelation()

                if verbose:
                    print "Wt extrapolation method: ", extrapolation_method
                    print "Wt analysis type: ", analysis_type

                box_settings = {
                    "xlim": [0.111, 0.117],
                    "ylim": [0.295, 0.305],
                    "zoom_factor": 10}

                w_t_energy_analysis.plot(figure_name_appendix="_w_scale")
                w_t_energy_analysis.plot(
                    x_limits=[-0.015, 0.15], y_limits=[-0.025, 0.4],
                    plot_hline_at=0.3, figure_name_appendix="_w_scale_zoomed",
                    zoom_box=box_settings)

                w0_dict = w_t_energy_analysis.get_w0_scale(
                    extrapolation_method=extrapolation_method,
                    W0=energy_fit_target, plot_fit=False)

                # w0_reference_scale[extrapolation_method][analysis_type] = \
                # 	w0_dict

        for t_flow in q0_flow_times:
            w_t_energy_analysis.plot_autocorrelation_at(target_flow=t_flow)
            w_t_energy_analysis.plot_mc_history_at(target_flow=t_flow)


    # Loads/saves pickle file if it exists
    reference_scale_file = (
        "reference_scale_%s.pkl" % "-".join(data.batch_names))
    if os.path.isfile(reference_scale_file) and \
    	isinstance(reference_scale, type(None)) or \
    	not "energy" in observables:

        reference_scale = load_pickle(reference_scale_file)
    else:
        save_pickle(reference_scale_file, reference_scale)

    data.set_reference_values(reference_scale)

    if "plaq" in observables:
        plaq_analysis = PlaqPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)
        for analysis_type in post_analysis_data_type:
            plaq_analysis.set_analysis_data_type(analysis_type)
            print plaq_analysis
            plaq_analysis.plot()
            plaq_analysis.plot_autocorrelation()
            for t_flow in q0_flow_times:
                plaq_analysis.plot_autocorrelation_at(target_flow=t_flow)
                plaq_analysis.plot_mc_history_at(target_flow=t_flow)

    if "topc" in observables:
        topc_analysis = TopcPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)
        for analysis_type in post_analysis_data_type:
            topc_analysis.set_analysis_data_type(analysis_type)
            print topc_analysis
            topc_analysis.plot(y_limits=[-5, 5])
            topc_analysis.plot_autocorrelation()
            for t_flow in q0_flow_times:
                topc_analysis.plot_autocorrelation_at(target_flow=t_flow)
                topc_analysis.plot_mc_history_at(target_flow=t_flow)

    if "topc2" in observables:
        topc2_analysis = Topc2PostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)
        for analysis_type in post_analysis_data_type:
            topc2_analysis.set_analysis_data_type(analysis_type)
            print topc2_analysis

            if "topcr" in observables:
                continue

            topc2_analysis.plot()
            topc2_analysis.plot_autocorrelation()
            for t_flow in q0_flow_times:
                topc2_analysis.plot_autocorrelation_at(target_flow=t_flow)
                topc2_analysis.plot_mc_history_at(target_flow=t_flow)

    if "topc4" in observables:
        topc4_analysis = Topc4PostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)
        for analysis_type in post_analysis_data_type:
            topc4_analysis.set_analysis_data_type(analysis_type)
            print topc4_analysis

            if "topcr" in observables:
                continue

            topc4_analysis.plot()
            topc4_analysis.plot_autocorrelation()
            for t_flow in q0_flow_times:
                topc4_analysis.plot_autocorrelation_at(target_flow=t_flow)
                topc4_analysis.plot_mc_history_at(target_flow=t_flow)

    if "topcr" in observables:
        topcr_analysis = TopcRPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)
        for analysis_type in post_analysis_data_type:
            topcr_analysis.set_analysis_data_type(analysis_type)
            print topcr_analysis
            topcr_analysis.plot()
            topcr_analysis.print_batch_values()

        topcr_analysis.compare_lattice_values(tf=topcr_tf)

    if "topcrMC" in observables:
        topcrmc_analysis = TopcRMCIntervalPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        print interval_setup(batch_parameter_list, "MC")

        interval_dict_list = topcrmc_analysis.setup_intervals(
            intervals=interval_setup(batch_parameter_list, "MC"))

        for analysis_type in post_analysis_data_type:
            topcrmc_analysis.set_analysis_data_type(analysis_type)
            print topcrmc_analysis

            for int_keys in interval_dict_list:
                print section_seperator
                print "Interval: %s" % int_keys
                topcrmc_analysis.plot_interval(int_keys)
                topcrmc_analysis.compare_lattice_values(int_keys, tf=topcr_tf)
                topcrmc_analysis.print_batch_values(int_keys)

            topcrmc_analysis.plot_series([0, 1, 2, 3])

    if "topct" in observables:
        topct_analysis = TopctPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        interval_dict_list = topct_analysis.setup_intervals()

        for analysis_type in post_analysis_data_type:
            topct_analysis.set_analysis_data_type(analysis_type)
            print topct_analysis
            for int_keys in interval_dict_list:
                topct_analysis.plot_interval(int_keys)

            topct_analysis.plot_series([0, 1, 2, 3])
            topct_analysis.plot_autocorrelation()

    if "topcte" in observables:
        topcte_analysis = TopcteIntervalPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        interval_dict_list = topcte_analysis.setup_intervals(
            intervals=interval_setup(batch_parameter_list, "Eucl"))

        for analysis_type in post_analysis_data_type:
            topcte_analysis.set_analysis_data_type(analysis_type)

            for int_keys in interval_dict_list:
                topcte_analysis.plot_interval(int_keys)

            topcte_analysis.plot_series([0, 1, 2, 3])
            topcte_analysis.plot_autocorrelation()

    if "topcMC" in observables:
        topcmc_analysis = TopcMCIntervalPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        interval_dict_list = topcmc_analysis.setup_intervals(
            intervals=interval_setup(batch_parameter_list, "MC"))

        for analysis_type in post_analysis_data_type:
            topcmc_analysis.set_analysis_data_type(analysis_type)
            print topcmc_analysis

            for int_keys in interval_dict_list:
                topcmc_analysis.plot_interval(int_keys)

            topcmc_analysis.plot_series([0, 1, 2, 3])
            topcmc_analysis.plot_autocorrelation()

    # Loops over different extrapolation methods
    if "topsus" in observables:

        topsus_analysis = TopsusPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        for analysis_type in post_analysis_data_type:
            topsus_analysis.set_analysis_data_type(analysis_type)
            topsus_analysis.plot()
            topsus_analysis.plot_autocorrelation()
            for cont_target in topsus_fit_targets:
                for extrapolation_method in extrapolation_methods:
                    topsus_analysis.plot_continuum(
                        cont_target, extrapolation_method=extrapolation_method,
                        plot_continuum_fit=plot_continuum_fit)

                    if topsus_analysis.check_continuum_extrapolation():
                        fit_parameters = append_fit_params(
                            fit_parameters,
                            topsus_analysis.observable_name_compact,
                            analysis_type,
                            topsus_analysis.get_linefit_parameters())

            for t_flow in q0_flow_times:
                topsus_analysis.plot_autocorrelation_at(target_flow=t_flow)
                topsus_analysis.plot_mc_history_at(target_flow=t_flow)

    if "topsusqtq0" in observables:
        topsusqtq0_analysis = TopsusQtQ0PostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        interval_dict_list = topsusqtq0_analysis.setup_intervals()

        for analysis_type in post_analysis_data_type:
            topsusqtq0_analysis.set_analysis_data_type(analysis_type)
            print topsusqtq0_analysis

            for extrapolation_method in extrapolation_methods:
                for int_keys in interval_dict_list:

                    if (list(set(int_keys))[0] == "0.00" and
                            extrapolation_method == "bootstrap"):
                        print("Skipping intervals 0.00, as they may contain "
                              "negative numbers from bootstrapped data.")
                        continue

                    topsusqtq0_analysis.plot_interval(int_keys)
                    for cont_target in topsus_fit_targets:
                        topsusqtq0_analysis.plot_continuum(
                            cont_target, int_keys,
                            extrapolation_method=extrapolation_method)

                        if topsusqtq0_analysis.check_continuum_extrapolation():
                            fit_parameters = append_fit_params(
                                fit_parameters,
                                topsusqtq0_analysis.observable_name_compact,
                                analysis_type,
                                topsusqtq0_analysis.get_linefit_parameters())

            topsusqtq0_analysis.plot_series([0, 1, 2, 3])
            topsusqtq0_analysis.plot_series([3, 4, 5, 6])
            topsusqtq0_analysis.plot_series([0, 2, 4, 6])
            topsusqtq0_analysis.plot_autocorrelation()

    if "topsust" in observables:
        topsust_analysis = TopsustPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        interval_dict_list = topsust_analysis.setup_intervals()

        for analysis_type in post_analysis_data_type:
            topsust_analysis.set_analysis_data_type(analysis_type)
            print topsust_analysis

            for int_keys in interval_dict_list:
                topsust_analysis.plot_interval(int_keys)
                for cont_target in topsus_fit_targets:
                    for extrapolation_method in extrapolation_methods:
                        topsust_analysis.plot_continuum(
                            cont_target, int_keys,
                            extrapolation_method=extrapolation_method)

                        if topsust_analysis.check_continuum_extrapolation():
                            fit_parameters = append_fit_params(
                                fit_parameters,
                                topsust_analysis.observable_name_compact,
                                analysis_type,
                                topsust_analysis.get_linefit_parameters())

            topsust_analysis.plot_series([0, 1, 2, 3])
            topsust_analysis.plot_autocorrelation()

    if "topsuste" in observables:
        topsuste_analysis = TopsusteIntervalPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        interval_dict_list = topsuste_analysis.setup_intervals(
            intervals=interval_setup(batch_parameter_list, "Eucl"))

        for analysis_type in post_analysis_data_type:
            topsuste_analysis.set_analysis_data_type(analysis_type)
            print topsuste_analysis
            for int_keys in interval_dict_list:
                topsuste_analysis.plot_interval(int_keys)
                for cont_target in topsus_fit_targets:
                    for extrapolation_method in extrapolation_methods:
                        topsuste_analysis.plot_continuum(
                            cont_target, int_keys,
                            extrapolation_method=extrapolation_method)

                        if topsuste_analysis.check_continuum_extrapolation():
                            fit_parameters = append_fit_params(
                                fit_parameters,
                                topsuste_analysis.observable_name_compact,
                                analysis_type,
                                topsuste_analysis.get_linefit_parameters())

            topsuste_analysis.plot_series([0, 1, 2, 3])
            topsuste_analysis.plot_autocorrelation()

    if "topsusMC" in observables:
        topsusmc_analysis = TopsusMCIntervalPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        interval_dict_list = topsusmc_analysis.setup_intervals(
            intervals=interval_setup(batch_parameter_list, "MC"))

        for analysis_type in post_analysis_data_type:
            topsusmc_analysis.set_analysis_data_type(analysis_type)
            print topsusmc_analysis

            for int_keys in interval_dict_list:
                topsusmc_analysis.plot_interval(int_keys)
                for cont_target in topsus_fit_targets:
                    for extrapolation_method in extrapolation_methods:
                        topsusmc_analysis.plot_continuum(
                            cont_target, int_keys,
                            extrapolation_method=extrapolation_method)

                        if topsusmc_analysis.check_continuum_extrapolation():
                            fit_parameters = append_fit_params(
                                fit_parameters,
                                topsusmc_analysis.observable_name_compact,
                                analysis_type,
                                topsusmc_analysis.get_linefit_parameters())

            topsusmc_analysis.plot_series([0, 1, 2, 3])
            topsusmc_analysis.plot_autocorrelation()

    if "qtq0e" in observables:
        qtq0e_analysis = QtQ0EuclideanPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        # Retrieves flow times
        flow_times = qtq0e_analysis.setup_intervals()

        y_limits = [-0.05,0.5]

        # Checks that we have similar flow times.
        # +1 in order to ensure the zeroth flow time does not count as false.
        def clean_string(s): return float(s[-4:])
        assert np.all([np.all([clean_string(i)+1 for i in ft])
                       for ft in flow_times]), "q0 times differ."

        for te in euclidean_time_percents:
            for analysis_type in post_analysis_data_type:
                qtq0e_analysis.set_analysis_data_type(te, analysis_type)
                print qtq0e_analysis
                for tf in q0_flow_times:  # Flow times
                    print "Plotting te: %f and tf: %f" % (te, tf)
                    qtq0e_analysis.plot_interval(tf, te)

                qtq0e_analysis.plot_series(te, [0, 1, 3, 6], y_limits=y_limits)

    if "qtq0eff" in observables:
        # if analysis_type != "unanalyzed": continue
        plateau_limits = [0.4, 0.6]
        qtq0e_analysis = QtQ0EffectiveMassPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)
        for analysis_type in post_analysis_data_type:
            qtq0e_analysis.set_analysis_data_type(analysis_type)
            print qtq0e_analysis

            for tf in q0_flow_times:  # Flow times
                if tf != 0.6:
                    continue
                # qtq0e_analysis.plot_interval(tf)
                qtq0e_analysis.plot_plateau(tf, plateau_limits)

            # y_limits = [-1, 1]
            # error_shape = "bars"
            # qtq0e_analysis.plot_series([0, 1, 2, 3],
            #                            error_shape=error_shape,
            #                            y_limits=y_limits)
            # qtq0e_analysis.plot_series([0, 2, 3, 5],
            #                            error_shape=error_shape,
            #                            y_limits=y_limits)

    if "qtq0effMC" in observables:
        # if analysis_type != "unanalyzed": continue
        effmass_mc_analysis = QtQ0EffectiveMassMCIntervalsPostAnalysis(
            data, figures_folder=figures_folder, verbose=verbose)

        mc_interval_dict_list = effmass_mc_analysis.setup_intervals(
            intervals=interval_setup(batch_parameter_list, "MC"))

        for analysis_type in post_analysis_data_type:
            for tf0 in q0_flow_times:  # Flow times
                if tf0 != 0.6:
                    continue
                effmass_mc_analysis.set_analysis_data_type(tf0, analysis_type)
                print effmass_mc_analysis

                for int_keys in mc_interval_dict_list:
                    effmass_mc_analysis.plot_interval(tf0, int_keys)

                y_limits = [-1, 1]
                error_shape = "bars"
                effmass_mc_analysis.plot_series([0, 1, 2, 3], tf0,
                                                error_shape=error_shape,
                                                y_limits=y_limits)
                effmass_mc_analysis.plot_series([0, 2, 3, 4], tf0,
                                                error_shape=error_shape,
                                                y_limits=y_limits)

    # Prints and writes fit parameters to file.
    for obs in observables:
        if "topsus" in obs:
            skip_values = ["a", "a_err", "b", "b_err"]

            tab_filename = "topsus_extrap_values_" 
            tab_filename += "-".join(data.batch_names) + ".txt"

            write_fit_parameters_to_file(fit_parameters,
                                         os.path.join("param_file.txt"),
                                         skip_values=skip_values,
                                         verbose=verbose,
                                         tab_filename=tab_filename)
            break

    if len(gif_params["gif_observables"]) != 0:

        if "qtq0e" in gif_params["gif_observables"]:
            qtq0e_gif = QtQ0EPostGif(data, figures_folder=figures_folder,
                                     verbose=verbose)
            qtq0e_gif.image_creator(gif_params["gif_euclidean_time"],
                                    gif_betas=gif_params["betas_to_plot"],
                                    plot_together=gif_params["plot_together"],
                                    error_shape=gif_params["error_shape"])

        if "qtq0eff" in gif_params["gif_observables"]:
            qtq0eff_gif = QtQ0EffPostGif(data, figures_folder=figures_folder,
                                         verbose=verbose)
            qtq0eff_gif.data_setup()
            qtq0eff_gif.image_creator(
                gif_betas=gif_params["betas_to_plot"],
                plot_together=gif_params["plot_together"], error_shape="bars")


def main():
    exit("No default run for post_analyser.py is currently set up.")


if __name__ == '__main__':
    main()
