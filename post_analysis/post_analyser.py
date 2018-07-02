from observable_analysis import *
from tools.postanalysisdatareader import PostAnalysisDataReader
from tools.analysis_setup_tools import append_fit_params, \
	write_fit_parameters_to_file, get_intervals, interval_setup
from tools.value_comparer import ValueCMP
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


def post_analysis(beta_parameter_list, observables,
	topsus_fit_targets, line_fit_interval_points, energy_fit_target,
	q0_flow_times, euclidean_time_percents, extrapolation_methods="nearest",
	plot_continuum_fit=False, figures_folder="figures", 
	post_analysis_data_type=["bootstrap", "jackknife", "unanalyzed"], 
	bval_to_plot="all", topcr_tf="t0beta", gif_params=None, t0_value_extraction=None, 
	verbose=False):
	"""
	Post analysis of the flow observables.

	Args: 
		beta_parameter_list: list of dicts, beta batch parameters.
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
		bval_to_plot: str or list of beta floats. Which beta values to plot
			together. Default is "all".
		gif_params: dict, parameters to use in gif creation. Default is None.
			dict = { 
				"gif_observables": [list_of_observables],
				"gif_euclidean_time": euclidean_time_to_plot_at, 
				"gif_flow_range": [range of floats to plot for],
				"betas_to_plot": "all",
				"plot_together": False,
				"error_shape": "band"}
		t0_value_extraction: bool, will write values at given t0 to file 
			if True.
		verbose: bool, a more verbose run. Default is False.
	"""

	section_seperator = "="*160
	print section_seperator
	print "Post-analysis: retrieving data from folders: %s" % (
			", ".join([os.path.join(b["batch_folder"], b["batch_name"]) \
		for b in beta_parameter_list]))

	# Topcr requires a few more observables to be fully utilized.
	old_obs = observables
	if "topcr" in observables:
		observables += ["topc2", "topc4"]
	if "topcrMC" in observables:
		observables += ["topc2MC", "topc4MC"]

	data = PostAnalysisDataReader(beta_parameter_list,
		observables_to_load=observables)

	# Resets to the old observables, as not to analyze topc2 and topc4.
	observables = old_obs 

	fit_parameters = []
	t0_reference_scale = {
		extrap_method: {atype: {} for atype in post_analysis_data_type}
		for extrap_method in extrapolation_methods
	}

	cmp_values = ValueCMP(observables, post_analysis_data_type, 
		verbose=verbose)

	# comparison_values = {obs: {extrap_method: {atype: {} 
	# 			for atype in post_analysis_data_type}
	# 		for extrap_method in extrapolation_methods} 
	# 	for obs in observables
	# }

	# # Dictionary to store values we are to write out to file in.
	# comparison_values = {obs: {atype: {} 
	# 		for atype in post_analysis_data_type}
	# 	for obs in observables
	# }

	if "energy" in observables:
		for extrapolation_method in extrapolation_methods:
			
			energy_analysis = EnergyPostAnalysis(data, 
				figures_folder=figures_folder, verbose=verbose)

			for analysis_type in post_analysis_data_type:
				energy_analysis.set_analysis_data_type(analysis_type)
				
				print energy_analysis
				if verbose:
					print "Energy extrapolation method: ", extrapolation_method
					print "Energy analysis type: ", analysis_type


				energy_analysis.plot()
				energy_analysis.plot(x_limits=[-0.01,0.15], 
					y_limits=[-0.025, 0.4], plot_hline_at=0.3, 
					figure_name_appendix="_zoomed")

				t0_dict = energy_analysis.get_scale(
					extrapolation_method=extrapolation_method, 
					E0=energy_fit_target, plot_fit=False)

				t0_reference_scale[extrapolation_method][analysis_type] = \
					t0_dict

				# # Retrofits the energy for continiuum limit
				# energy_analysis.plot_continuum(0.3, 0.015, 
				# 	extrapolation_method=extrapolation_method)

				# # Plot running coupling
				# energy_analysis.coupling_fit()
	else:
		t0_reference_scale = None

	data.set_reference_values(t0_reference_scale)

	if "plaq" in observables:
		plaq_analysis = PlaqPostAnalysis(data,
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			plaq_analysis.set_analysis_data_type(analysis_type)
			print plaq_analysis
			plaq_analysis.plot()


	if "topc" in observables:
		topc_analysis = TopcPostAnalysis(data, 
			figures_folder=figures_folder,verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topc_analysis.set_analysis_data_type(analysis_type)
			print topc_analysis
			topc_analysis.plot(y_limits=[-5,5])
			# comparison_values["topc"][analysis_type] = \
			# 	topc_analysis.get_values("t0", analysis_type)

			# cmp_values.append(topc_analysis.get_values("t0", analysis_type), 
			# 	analysis_type)


	if "topc2" in observables:
		topc2_analysis = Topc2PostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topc2_analysis.set_analysis_data_type(analysis_type)
			print topc2_analysis
			topc2_analysis.plot()

			# comparison_values["topc2"][analysis_type] = \
			# 	topc2_analysis.get_values("t0", analysis_type)

	if "topc4" in observables:
		topc4_analysis = Topc4PostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topc4_analysis.set_analysis_data_type(analysis_type)
			print topc4_analysis
			topc4_analysis.plot()

	if "topcr" in observables:
		topcr_analysis = TopcRPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topcr_analysis.set_analysis_data_type(analysis_type)
			print topcr_analysis
			topcr_analysis.plot()

		topcr_analysis.compare_lattice_values(tf=topcr_tf)

	if "topcrMC" in observables:
		topcrmc_analysis = TopcRMCIntervalPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)

		print interval_setup(beta_parameter_list, "MC")

		interval_dict_list = topcrmc_analysis.setup_intervals(
			intervals=interval_setup(beta_parameter_list, "MC"))

		for analysis_type in post_analysis_data_type:
			topcrmc_analysis.set_analysis_data_type(analysis_type)
			print topcrmc_analysis

			for int_keys in interval_dict_list:
				print section_seperator
				print "Interval: %s" % int_keys
				topcrmc_analysis.plot_interval(int_keys)
				topcrmc_analysis.compare_lattice_values(int_keys, tf=topcr_tf)

			topcrmc_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "topct" in observables:
		topct_analysis = TopctPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)

		interval_dict_list = topct_analysis.setup_intervals()

		for analysis_type in post_analysis_data_type:
			topct_analysis.set_analysis_data_type(analysis_type)
			print topct_analysis
			for int_keys in interval_dict_list:
				topct_analysis.plot_interval(int_keys)
				# cmp_values.append(topct_analysis.get_values("t0", analysis_type), 
				# 	analysis_type)

			topct_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "topcte" in observables:
		topcte_analysis = TopcteIntervalPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)

		interval_dict_list = topcte_analysis.setup_intervals(
			intervals=interval_setup(beta_parameter_list, "Eucl"))

		for analysis_type in post_analysis_data_type:
			topcte_analysis.set_analysis_data_type(analysis_type)
			print topcte_analysis
			for int_keys in interval_dict_list:
				topcte_analysis.plot_interval(int_keys)

			topcte_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "topcMC" in observables:
		topcmc_analysis = TopcMCIntervalPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)

		interval_dict_list = topcmc_analysis.setup_intervals(
			intervals=interval_setup(beta_parameter_list, "MC"))

		for analysis_type in post_analysis_data_type:
			topcmc_analysis.set_analysis_data_type(analysis_type)
			print topcmc_analysis

			for int_keys in interval_dict_list:
				topcmc_analysis.plot_interval(int_keys)

			topcmc_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	# Loops over different extrapolation methods
	for extrapolation_method in extrapolation_methods:
		if "topsus" in observables:
			topsus_analysis = TopsusPostAnalysis(data, 
				figures_folder=figures_folder, verbose=verbose)
			for analysis_type in post_analysis_data_type:
				topsus_analysis.set_analysis_data_type(analysis_type)
				topsus_analysis.plot()
				for cont_target in topsus_fit_targets:
					topsus_analysis.plot_continuum(cont_target, 
						extrapolation_method=extrapolation_method,
						plot_continuum_fit=plot_continuum_fit)

					fit_parameters = append_fit_params(fit_parameters, 
						topsus_analysis.observable_name_compact, analysis_type,
						topsus_analysis.get_linefit_parameters())


		if "topsusqtq0" in observables:
			topsusqtq0_analysis = TopsusQtQ0PostAnalysis(data,
				figures_folder=figures_folder, verbose=verbose)

			interval_dict_list = topsusqtq0_analysis.setup_intervals()

			for analysis_type in post_analysis_data_type:
				topsusqtq0_analysis.set_analysis_data_type(analysis_type)
				print topsusqtq0_analysis

				for int_keys in interval_dict_list:

					if (list(set(int_keys))[0] == "0.00" and 
						extrapolation_method == "bootstrap"):
						print ("Skipping intervals 0.00, as they may contain "
							"negative numbers from bootstrapped data.")
						continue

					topsusqtq0_analysis.plot_interval(int_keys)
					for cont_target in topsus_fit_targets:
						topsusqtq0_analysis.plot_continuum(cont_target, 
							int_keys,
							extrapolation_method=extrapolation_method)

						fit_parameters = append_fit_params(fit_parameters, 
							topsusqtq0_analysis.observable_name_compact, 
							analysis_type,
							topsusqtq0_analysis.get_linefit_parameters())

				topsusqtq0_analysis.plot_series([0,1,2,3], beta=bval_to_plot)
				topsusqtq0_analysis.plot_series([3,4,5,6], beta=bval_to_plot)

		if "topsust" in observables:
			topsust_analysis = TopsustPostAnalysis(data,
				figures_folder=figures_folder, verbose=verbose)

			interval_dict_list = topsust_analysis.setup_intervals()

			for analysis_type in post_analysis_data_type:
				topsust_analysis.set_analysis_data_type(analysis_type)
				print topsust_analysis

				for int_keys in interval_dict_list:
					topsust_analysis.plot_interval(int_keys)
					for cont_target in topsus_fit_targets:
						topsust_analysis.plot_continuum(cont_target, int_keys,
							extrapolation_method=extrapolation_method)

						fit_parameters = append_fit_params(fit_parameters, 
							topsust_analysis.observable_name_compact, 
							analysis_type,
							topsust_analysis.get_linefit_parameters())

				topsust_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

		if "topsuste" in observables:
			topsuste_analysis = TopsusteIntervalPostAnalysis(data, 
				figures_folder=figures_folder, verbose=verbose)

			print interval_setup(beta_parameter_list, "Eucl")

			interval_dict_list = topsuste_analysis.setup_intervals(
				intervals=interval_setup(beta_parameter_list, "Eucl"))

			for analysis_type in post_analysis_data_type:
				topsuste_analysis.set_analysis_data_type(analysis_type)
				print topsuste_analysis
				for int_keys in interval_dict_list:
					topsuste_analysis.plot_interval(int_keys)
					for cont_target in topsus_fit_targets:
						topsuste_analysis.plot_continuum(cont_target, int_keys,
							extrapolation_method=extrapolation_method)

						fit_parameters = append_fit_params(fit_parameters, 
							topsuste_analysis.observable_name_compact, 
							analysis_type,
							topsuste_analysis.get_linefit_parameters())

				topsuste_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

		if "topsusMC" in observables:
			topsusmc_analysis = TopsusMCIntervalPostAnalysis(data,
				figures_folder=figures_folder, verbose=verbose)

			interval_dict_list = topsusmc_analysis.setup_intervals(
				intervals=interval_setup(beta_parameter_list, "MC"))

			for analysis_type in post_analysis_data_type:
				topsusmc_analysis.set_analysis_data_type(analysis_type)
				print topsusmc_analysis
			
				for int_keys in interval_dict_list:
					topsusmc_analysis.plot_interval(int_keys)
					for cont_target in topsus_fit_targets:
						topsusmc_analysis.plot_continuum(cont_target, int_keys,
							extrapolation_method=extrapolation_method)

						fit_parameters = append_fit_params(fit_parameters, 
							topsusmc_analysis.observable_name_compact, 
							analysis_type,
							topsusmc_analysis.get_linefit_parameters())

				topsusmc_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "qtq0e" in observables:
		qtq0e_analysis = QtQ0EuclideanPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)

		# Retrieves flow times
		flow_times = qtq0e_analysis.setup_intervals()

		# Checks that we have similar flow times.
		# +1 in order to ensure the zeroth flow time does not count as false.
		clean_string = lambda s: float(s[-4:])
		assert np.all([np.all([clean_string(i)+1 for i in ft]) \
			for ft in flow_times]), "q0 times differ."

		for te in euclidean_time_percents[:1]:
			for analysis_type in post_analysis_data_type:
				qtq0e_analysis.set_analysis_data_type(te, analysis_type)
				print qtq0e_analysis
				for tf in q0_flow_times: # Flow times
					print "Plotting te: %g and tf: %g" % (te, tf)
					qtq0e_analysis.plot_interval(tf, te)
				
				qtq0e_analysis.plot_series(te, [0,1,2,3], beta=bval_to_plot)
				qtq0e_analysis.plot_series(te, [0,2,3,4], beta=bval_to_plot)

	if "qtq0eff" in observables:
		# if analysis_type != "unanalyzed": continue
		qtq0e_analysis = QtQ0EffectiveMassPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			qtq0e_analysis.set_analysis_data_type(analysis_type)
			print qtq0e_analysis

			for tf in q0_flow_times: # Flow times
				if tf != 0.6: 
					continue
				qtq0e_analysis.plot_interval(tf)

			y_limits = [-1, 1]
			error_shape = "bars"
			qtq0e_analysis.plot_series([0,1,2,3], beta=bval_to_plot,
				error_shape=error_shape, y_limits=y_limits)
			qtq0e_analysis.plot_series([0,2,3,4], beta=bval_to_plot,
				error_shape=error_shape, y_limits=y_limits)


	if "qtq0effMC" in observables:
		# if analysis_type != "unanalyzed": continue
		effmass_mc_analysis = QtQ0EffectiveMassMCIntervalsPostAnalysis(data,
			figures_folder=figures_folder, verbose=verbose)

		for analysis_type in post_analysis_data_type:
			effmass_mc_analysis.set_analysis_data_type(analysis_type)
			print effmass_mc_analysis

			for tf in q0_flow_times: # Flow times
				if tf != 0.6: 
					continue
				effmass_mc_analysis.plot_interval(tf)

			y_limits = [-1, 1]
			error_shape = "bars"
			effmass_mc_analysis.plot_series([0,1,2,3], beta=bval_to_plot,
				error_shape=error_shape, y_limits=y_limits)
			effmass_mc_analysis.plot_series([0,2,3,4], beta=bval_to_plot,
				error_shape=error_shape, y_limits=y_limits)


	# Prints and writes fit parameters to file.
	for obs in observables:
		if "topsus" in obs:
			skip_values = ["a", "a_err", "b", "b_err"]
			write_fit_parameters_to_file(fit_parameters, 
				os.path.join("param_file.txt"), skip_values=skip_values, 
				verbose=verbose)
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
			qtq0eff_gif.image_creator(gif_betas=gif_params["betas_to_plot"],
				plot_together=gif_params["plot_together"],
				error_shape="bars")
			


def main():
	exit("No default run for post_analyser.py is currently set up.")

if __name__ == '__main__':
	main()