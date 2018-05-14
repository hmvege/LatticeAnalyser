from observable_analysis import *
from tools.postanalysisdatareader import PostAnalysisDataReader
from collections import OrderedDict
import numpy as np
import os
from tqdm import tqdm

def append_fit_params(fplist, obs_name, analysis_name, fparams):
	"""Function for appending fit parameters."""
	chi_squared, fit_params, topsus, topsus_err, N_F, N_F_err, \
		fit_target, interval = fparams
	fplist.append({
		"observable_type": obs_name,
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
	})
	return fplist

def write_fit_parameters_to_file(fparams, fname, verbose=False):
	"""Function for writing fit parameters to file."""
	with open(fname, "w") as f:
		sorted_parameter_list = sorted(fparams, key=lambda d: \
			(d["fit_target"], d["analysis_type"]))

		# Default float width
		fw = 14
		dict_keys = OrderedDict([
			("observable_type", {"name": "obs", "w": 14, "type": "s"}),
			("fit_target", {"name": "f_t", "w": 5, "type": ".2f"}),
			("interval", {"name": "int", "w": 12, "type": "s"}),
			("analysis_type", {"name": "atype", "w": 14, "type": "s"}),
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
			header_string += create_str(k[-1]["name"], k[-1]["w"], "s")
		if verbose: 
			print header_string
		f.write(header_string + "\n")

		# Writes out analysis values to text file
		for fp in sorted_parameter_list:
			line_values = ""
			for k in dict_keys.items():
				line_values += create_str(fp[k[0]], k[-1]["w"],
					k[-1]["type"])
			if verbose:
				print line_values
			f.write(line_values + "\n")

def default_post_analysis(PostAnalysis, data, figures_folder, analysis_type,
	verbose=False):
	analysis = PostAnalysis(data, figures_folder=figures_folder, 
		verbose=verbose)
	analysis.set_analysis_data_type(analysis_type)
	print analysis
	analysis.plot()

def plaq_post_analysis(*args, **kwargs):
	default_post_analysis(PlaqPostAnalysis, *args, **kwargs)

def post_analysis(beta_parameter_list, observables,
	topsus_fit_targets, line_fit_interval_points, energy_fit_target,
	q0_flow_times, euclidean_time_percents, figures_folder="figures", 
	post_analysis_data_type=None, bval_to_plot="all", gif_params=None, 
	verbose=False):
	"""
	Post analysis of the flow observables.

	Args: 
		beta_parameter_list: list of the beta batch parameters.
		topsus_fit_targets: list of x-axis points to line fit at.
		line_fit_interval_points: int, number of points which we will use in 
			line fit.
		energy_fit_target: point of which we will perform a line fit at.
		q0_flow_times: points where we perform qtq0 at.
		euclidean_time_percents: points where we perform qtq0e at.
	"""

	batch_folders = [os.path.join(b["batch_folder"], b["batch_name"]) \
		for b in beta_parameter_list]

	print "="*100 
	print "Post-analysis: retrieving data from folders: %s" % (
		", ".join(batch_folders))

	if post_analysis_data_type == None:
		post_analysis_data_type = ["bootstrap", "jackknife", "unanalyzed"]

	# Loads data from post analysis folder
	data = PostAnalysisDataReader(batch_folders)

	continuum_targets = topsus_fit_targets

	fit_parameters = []

	if "plaq" in observables:
		plaq_analysis = PlaqPostAnalysis(data,
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			plaq_analysis.set_analysis_data_type(analysis_type)
			print plaq_analysis
			plaq_analysis.plot()

	if "energy" in observables:
		# Plots energy
		energy_analysis = EnergyPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			energy_analysis.set_analysis_data_type(analysis_type)
			print energy_analysis
			energy_analysis.plot()

			# # Retrofits the energy for continiuum limit
			# energy_analysis.plot_continuum(0.3, 0.015, "bootstrap_fit")

			# # Plot running coupling
			# energy_analysis.coupling_fit()

	if "topc" in observables:
		topc_analysis = TopcPostAnalysis(data, 
			figures_folder=figures_folder,verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topc_analysis.set_analysis_data_type(analysis_type)
			print topc_analysis
			topc_analysis.plot(y_limits=[-5,5])

	if "topc2" in observables:
		topc2_analysis = Topc2PostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topc2_analysis.set_analysis_data_type(analysis_type)
			print topc2_analysis
			topc2_analysis.plot()

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
		topcr_analysis.compare_lattice_values()

	if "topct" in observables:
		topct_analysis = TopctPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topct_analysis.set_analysis_data_type(analysis_type)
			print topct_analysis
			N_int, intervals = topct_analysis.get_N_intervals()
			for i in range(N_int):
				topct_analysis.plot_interval(i)
			topct_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "topcte" in observables:
		topcte_analysis = TopcteIntervalPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topcte_analysis.set_analysis_data_type(analysis_type)
			print topcte_analysis
			N_int, intervals = topcte_analysis.get_N_intervals()
			for i in range(N_int):
				topcte_analysis.plot_interval(i)
			topcte_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "topcMC" in observables:
		topcmc_analysis = TopcMCIntervalPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topcmc_analysis.set_analysis_data_type(analysis_type)
			print topcmc_analysis
			N_int, intervals = topcmc_analysis.get_N_intervals()
			for i in range(N_int):
				topcmc_analysis.plot_interval(i)
			topcmc_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "topsus" in observables:
		print "FIX EXTRAPOLATION SELECTION @ topsus @ post_analyser.py"
		# Plots topsusprint analysis
		topsus_analysis = TopsusPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topsus_analysis.set_analysis_data_type(analysis_type)
			print topsus_analysis
			topsus_analysis.plot()
			for cont_target in continuum_targets:
				topsus_analysis.plot_continuum(cont_target)

				fit_parameters = append_fit_params(fit_parameters, 
					topsus_analysis.observable_name_compact, analysis_type,
					topsus_analysis.get_linefit_parameters())

	if "topsus4" in observables:
		topsus4_analysis = Topsus4PostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topsus4_analysis.set_analysis_data_type(analysis_type)
			print topsus4_analysis
			topsus4_analysis.plot()

	if "topsusqtq0" in observables:
		print "FIX SELECTION & EXTRAPOLATION @ topsusqtq0 @ post_analyser.py"
		topsusqtq0_analysis = TopsusQtQ0PostAnalysis(data,
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topsusqtq0_analysis.set_analysis_data_type(analysis_type)
			print topsusqtq0_analysis
			N_int, intervals = topsusqtq0_analysis.get_N_intervals()
			for i in range(N_int):
				topsusqtq0_analysis.plot_interval(i)
				for cont_target in continuum_targets:
					topsusqtq0_analysis.plot_continuum(cont_target, i)

					fit_parameters = append_fit_params(fit_parameters, 
						topsusqtq0_analysis.observable_name_compact, 
						analysis_type,
						topsusqtq0_analysis.get_linefit_parameters())

			topsusqtq0_analysis.plot_series([0,1,2,3], beta=bval_to_plot)
			topsusqtq0_analysis.plot_series([3,4,5,6], beta=bval_to_plot)

	if "topsust" in observables:
		print "FIX EXTRAPOLATION SELECTION @ topsust @ post_analyser.py"
		topsust_analysis = TopsustPostAnalysis(data,
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topsust_analysis.set_analysis_data_type(analysis_type)
			print topsust_analysis
			N_int, intervals = topsust_analysis.get_N_intervals()
			for i in range(N_int):
				topsust_analysis.plot_interval(i)
				for cont_target in continuum_targets:
					topsust_analysis.plot_continuum(cont_target, i)

					fit_parameters = append_fit_params(fit_parameters, 
						topsust_analysis.observable_name_compact, 
						analysis_type,
						topsust_analysis.get_linefit_parameters())

			topsust_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "topsuste" in observables:
		print "FIX EXTRAPOLATION SELECTION @ topsuste @ post_analyser.py"
		topsuste_analysis = TopsusteIntervalPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topsuste_analysis.set_analysis_data_type(analysis_type)
			print topsuste_analysis
			N_int, intervals = topsuste_analysis.get_N_intervals()
			for i in range(N_int):
				topsuste_analysis.plot_interval(i)
				for cont_target in continuum_targets:
					topsuste_analysis.plot_continuum(cont_target, i)

					fit_parameters = append_fit_params(fit_parameters, 
						topsuste_analysis.observable_name_compact, 
						analysis_type,
						topsuste_analysis.get_linefit_parameters())

			topsuste_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "topsusMC" in observables:
		print "FIX EXTRAPOLATION SELECTION @ topsusMC @ post_analyser.py"
		topsusmc_analysis = TopsusMCIntervalPostAnalysis(data,
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			topsusmc_analysis.set_analysis_data_type(analysis_type)
			print topsusmc_analysis
			N_int, intervals = topsusmc_analysis.get_N_intervals()
			for i in range(N_int):
				topsusmc_analysis.plot_interval(i)
				for cont_target in continuum_targets:
					topsusmc_analysis.plot_continuum(cont_target, i)

					fit_parameters = append_fit_params(fit_parameters, 
						topsusmc_analysis.observable_name_compact, 
						analysis_type,
						topsusmc_analysis.get_linefit_parameters())

			topsusmc_analysis.plot_series([0,1,2,3], beta=bval_to_plot)

	if "qtq0e" in observables:
		print "FIX SELECTION @ qtq0e @ post_analyser.py"
		qtq0e_analysis = QtQ0EuclideanPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)

		# Checks that we have similar flow times
		N_tf, flow_intervals = qtq0e_analysis.get_N_intervals()
		clean_string = lambda s: float(s[-4:])

		# Retrieves flow times for each beta value.
		flow_times = np.asarray([b[1].keys() \
			for b in flow_intervals.items()]).T

		# +1 in order to ensure the zeroth flow time does not count as false.
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
		print "FIX SELECTION @ qtq0eff @ post_analyser.py"
		qtq0e_analysis = QtQ0EffectiveMassPostAnalysis(data, 
			figures_folder=figures_folder, verbose=verbose)
		for analysis_type in post_analysis_data_type:
			qtq0e_analysis.set_analysis_data_type(analysis_type)
			print qtq0e_analysis

			for tf in q0_flow_times: # Flow times
				print tf
				qtq0e_analysis.plot_interval(tf)
			y_limits = [-1, 1]
			error_shape = "bars"
			qtq0e_analysis.plot_series([0,1,2,3], beta=bval_to_plot,
				error_shape=error_shape, y_limits=y_limits)
			qtq0e_analysis.plot_series([0,2,3,4], beta=bval_to_plot,
				error_shape=error_shape, y_limits=y_limits)

	for obs in observables:
		if "topsus" in obs:
			write_fit_parameters_to_file(fit_parameters, 
				os.path.join("param_file.txt"), verbose=verbose)
			break

	if len(gif_params["gif_observables"]) != 0:

		if "qtq0e" in gif_params["gif_observables"]:
			qtq0e_gif = QtQ0EPostGif(data, figures_folder=figures_folder, 
				verbose=verbose)
			qtq0e_gif.image_creator(gif_params["gif_euclidean_time"],
				gif_betas=gif_params["betas_to_plot"], 
				plot_together=gif_params["plot_together"],
				error_shape=gif_params["error_shape"])

		# if "qtq0eff" in gif_params["gif_observables"]:
		# 	qtq0eff_gif = QtQ0EffPostGif(data, figures_folder=figures_folder, 
		# 		verbose=verbose)
		# 	qtq0eff_gif.data_setup()
		# 	qtq0eff_gif.image_creator(gif_betas=gif_params["betas_to_plot"],
		# 		plot_together=gif_params["plot_together"],
		# 		error_shape="bars")
			




def main():
	exit("No default run for post_analyser.py is currently set up.")

if __name__ == '__main__':
	main()