from observable_analysis import *
from tools.folderreadingtools import DataReader
from tools.analysis_setup_tools import get_intervals
import time
import numpy as np
import types
from tqdm import tqdm


def analyse_default(analysis_object, N_bs, NBins=None, skip_histogram=False,
	bs_index_lists=None):
	"""Default analysis method for pre-analysis."""
	print analysis_object
	analysis_object.boot(N_bs, index_lists=bs_index_lists)
	analysis_object.jackknife()
	analysis_object.save_post_analysis_data()
	analysis_object.plot_original()
	analysis_object.plot_boot()
	analysis_object.plot_jackknife()
	analysis_object.autocorrelation()
	analysis_object.plot_autocorrelation(0)
	analysis_object.plot_autocorrelation(-1)
	analysis_object.plot_mc_history(0)
	analysis_object.plot_mc_history(int(analysis_object.NFlows * 0.25))
	analysis_object.plot_mc_history(int(analysis_object.NFlows * 0.50))
	analysis_object.plot_mc_history(int(analysis_object.NFlows * 0.75))
	analysis_object.plot_mc_history(-1)
	analysis_object.plot_original()
	analysis_object.plot_boot()
	analysis_object.plot_jackknife()
	if not skip_histogram:
		# Plots histogram at the beginning, during and end.
		hist_pts = [0, 
			int(analysis_object.NFlows * 0.25), 
			int(analysis_object.NFlows * 0.50), 
			int(analysis_object.NFlows * 0.75), -1
		]
		for iHist in hist_pts:
			analysis_object.plot_histogram(iHist, NBins=NBins)
		analysis_object.plot_multihist([hist_pts[0], hist_pts[2], 
			hist_pts[-1]], NBins=NBins)
	analysis_object.plot_integrated_correlation_time()
	analysis_object.save_post_analysis_data() # save_as_txt=False

def gif_analysis(gif_analysis_obj, gif_flow_range, N_bs, 
	gif_euclidean_time=None):
	"""Function for creating gifs as effectively as possible."""

	# Sets up random boot strap lists so they are equal for all flow times
	NCfgs = gif_analysis_obj.N_configurations
	bs_index_lists = np.random.randint(NCfgs, size=(N_bs, NCfgs))

	# Runs basic data analysis
	gif_descr = "Data creation for %s gif" \
		% gif_analysis_obj.observable_name_compact
	for iFlow in tqdm(gif_flow_range, desc=gif_descr):
		if isinstance(gif_euclidean_time, types.NoneType):
			gif_analysis_obj.set_time(iFlow)
		else:
			gif_analysis_obj.set_time(iFlow, gif_euclidean_time)
		gif_analysis_obj.boot(N_bs, index_lists=bs_index_lists)
		gif_analysis_obj.autocorrelation()
		gif_analysis_obj.save_post_analysis_data()

def analyse_plaq(params):
	"""Analysis of the plaquette."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	plaq_analysis = PlaquetteAnalyser(obs_data("plaq"), dryrun=dryrun, 
		parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(plaq_analysis, N_bs)

def analyse_energy(params):
	"""Analysis of the energy."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	energy_analysis = EnergyAnalyser(obs_data("energy"), dryrun=dryrun, 
		parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(energy_analysis, N_bs)

def analyse_topsus(params):
	"""Analysis of topsus."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	topsus_analysis = TopsusAnalyser(obs_data("topc"), dryrun=dryrun, 
		parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(topsus_analysis, N_bs)

def analyse_topc(params):
	"""Analysis of Q."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	topc_analysis = TopcAnalyser(obs_data("topc"), dryrun=dryrun, 
		parallel=parallel, numprocs=numprocs, verbose=verbose)

	if topc_analysis.beta == 6.0:
		topc_analysis.y_limits = [-9, 9]
	elif topc_analysis.beta == 6.1:
		topc_analysis.y_limits = [-12, 12]
	elif topc_analysis.beta == 6.2:
		topc_analysis.y_limits = [-12, 12]
	else:
		topc_analysis.y_limits = [None, None]

	analyse_default(topc_analysis, N_bs)

def analyse_topc2(params):
	"""Analysis of Q^2."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	topc2_analysis = Topc2Analyser(obs_data("topc"), dryrun=dryrun, 
		parallel=parallel, numprocs=numprocs, verbose=verbose)

	if topc2_analysis.beta == 6.0:
		topc2_analysis.y_limits = [-81, 81]
	elif topc2_analysis.beta == 6.1:
		topc2_analysis.y_limits = [-144, 144]
	elif topc2_analysis.beta == 6.2:
		topc2_analysis.y_limits = [-196, 196]
	else:
		topc2_analysis.y_limits = [None, None]

	analyse_default(topc2_analysis, N_bs, NBins=150)

def analyse_topc4(params):
	"""Analysis the topological chage with q^4."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	topc4_analysis = Topc4Analyser(obs_data("topc"), dryrun=dryrun, 
		parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(topc4_analysis, N_bs)

def analyse_topcr(params):
	"""
	Analysis of the ratio with R=q4c/q2 of the topological charge. Performs an
	analysis on Q^2 and Q^4 with the same bootstrap samples, such that an post
	analysis can be performed on these explisitly.
	"""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

	topc2_analysis = Topc2Analyser(obs_data("topc"), dryrun=dryrun, 
		parallel=parallel, numprocs=numprocs, verbose=verbose)
	topc4_analysis = Topc4Analyser(obs_data("topc"), dryrun=dryrun, 
		parallel=parallel, numprocs=numprocs, verbose=verbose)

	N_cfgs_topc2 = topc2_analysis.N_configurations
	N_cfgs_topc4 = topc4_analysis.N_configurations
	assert N_cfgs_topc2 == N_cfgs_topc4, "NCfgs differ in topc2 and topc4."
	bs_index_lists = np.random.randint(N_cfgs_topc2,
		size=(N_bs, N_cfgs_topc2))

	analyse_default(topc2_analysis, N_bs, NBins=150, 
		bs_index_lists=bs_index_lists)
	analyse_default(topc4_analysis, N_bs, bs_index_lists=bs_index_lists)

def analyse_topsus4(params):
	"""Analysis topological susceptiblity with q^4."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	topsus4_analysis = Topsus4Analyser(obs_data("topc"), dryrun=dryrun, 
		parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(topsus4_analysis, N_bs)

def analyse_topsus_qtq0(params, q0_flow_times):
	"""
	Analysis the topological susceptiblity with one charge q0 set a given 
	flow time.
	"""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

	topsus_qtq0_analysis = TopsusQtQ0Analyser(obs_data("topc"), dryrun=dryrun,
		parallel=parallel, numprocs=numprocs, verbose=verbose)

	for q0_flow_time in q0_flow_times:
		topsus_qtq0_analysis.setQ0(q0_flow_time) 
		analyse_default(topsus_qtq0_analysis, N_bs, skip_histogram=True)

def analyse_qtq0e(params, q0_flow_times, euclidean_time_percents):
	"""Analysis for the effective mass qtq0 with q0 at a fixed flow time."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	if not obs_data.has_observable("topct"): return

	qtq0_analysis = QtQ0EuclideanAnalyser(obs_data("topct"), dryrun=dryrun,
		parallel=parallel, numprocs=numprocs, verbose=verbose)

	for q0_flow_time in q0_flow_times:
		for euclidean_percent in euclidean_time_percents:
			qtq0_analysis.set_time(q0_flow_time, euclidean_percent)
			analyse_default(qtq0_analysis, N_bs)

def analyse_qtq0_effective_mass(params, q0_flow_times):
	"""
	Pre-analyser for the effective mass qtq0 with q0 at a fixed flow time.
	"""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	if not obs_data.has_observable("topct"): return

	qtq0eff_analysis = QtQ0EffectiveMassAnalyser(obs_data("topct"), 
		dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)
	for q0_flow_time in q0_flow_times:
		if q0_flow_time != 0.6: # Only zeroth flow 
			continue
		qtq0eff_analysis.set_time(q0_flow_time)
		analyse_default(qtq0eff_analysis, N_bs)

def analyse_topct(params, numsplits):
	"""Analyses topological charge at a specific euclidean time."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	if not obs_data.has_observable("topct"): return

	topct_analysis = TopctAnalyser(obs_data("topct"),
		dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)

	indexes = np.linspace(0, topct_analysis.NT, numsplits, dtype=int) - 1
	indexes[0] += 1
	for ie in indexes:
		topct_analysis.setEQ0(ie)
		analyse_default(topct_analysis, N_bs)

def analyse_topsust(params, numsplits):
	"""Analyses topological susceptibility at a specific euclidean time."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	if not obs_data.has_observable("topct"): return

	topct_analysis = TopsustAnalyser(obs_data("topct"),
		dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)

	indexes = np.linspace(0, topct_analysis.NT, numsplits, dtype=int) - 1
	indexes[0] += 1
	for ie in indexes:
		topct_analysis.setEQ0(ie)
		analyse_default(topct_analysis, N_bs, skip_histogram=True)

def analyse_topcte_intervals(params, numsplits=None, intervals=None):
	"""
	Analysis function for the topological charge in euclidean time intervals. 
	Requires either numsplits or intervals.

	Args:
		params: list of default parameters containing obs_data, dryrun, 
			parallel, numprocs, verbose, N_bs.
		numsplits: number of splits to make in the dataset. Default is None.
		intervals: intervals to plot in. Default is none.
	"""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	if not obs_data.has_observable("topct"): return

	t_interval, _ = get_intervals(obs_data.NT, numsplits=numsplits, 
		intervals=intervals)

	analyse_topcte = TopcteIntervalAnalyser(obs_data("topct"),
		dryrun=dryrun, parallel=parallel, 
		numprocs=numprocs, verbose=verbose)

	for t_int in t_interval:
		analyse_topcte.set_t_interval(t_int)
		analyse_default(analyse_topcte, N_bs)

def analyse_topsuste_intervals(params, numsplits=None, intervals=None):
	"""
	Analysis function for the topological susceptibility in euclidean time. 
	Requires either numsplits or intervals.

	Args:
		params: list of default parameters containing obs_data, dryrun, 
			parallel, numprocs, verbose, N_bs.
		numsplits: number of splits to make in the dataset. Default is None.
		intervals: intervals to plot in. Default is none.
	"""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	if not obs_data.has_observable("topct"): return

	t_interval, _ = get_intervals(obs_data.NT, numsplits=numsplits, 
		intervals=intervals)

	analyse_topsuste = TopsusteIntervalAnalyser(obs_data("topct"),
		dryrun=dryrun, parallel=parallel, 
		numprocs=numprocs, verbose=verbose)

	for t_int in t_interval:
		analyse_topsuste.set_t_interval(t_int)
		analyse_default(analyse_topsuste, N_bs)

def analyse_topcMCTime(params, numsplits=None, intervals=None):
	"""Analysis the topological charge in monte carlo time slices."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

	MC_interval, interval_size = get_intervals(obs_data.NCfgs, 
		numsplits=numsplits, intervals=intervals)

	bs_index_lists = np.random.randint(interval_size,
		size=(N_bs, interval_size))

	for MC_int in MC_interval:
		analyse_topcMC = TopcMCIntervalAnalyser(obs_data("topc"),
			mc_interval=MC_int, dryrun=dryrun, parallel=parallel,
			numprocs=numprocs, verbose=verbose)
		# analyse_topcMC.set_mc_interval(MC_int)
		analyse_default(analyse_topcMC, N_bs, bs_index_lists=bs_index_lists)

def analyse_topsusMCTime(params, numsplits=None, intervals=None):
	"""Analysis the topological susceptibility in monte carlo time slices."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

	MC_interval, interval_size = get_intervals(obs_data.NCfgs,
		numsplits=numsplits, intervals=intervals)

	bs_index_lists = np.random.randint(interval_size,
		size=(N_bs, interval_size))

	for MC_int in MC_interval:
		analyse_topcMC = TopsusMCIntervalAnalyser(obs_data("topc"),
			mc_interval=MC_int, dryrun=dryrun, parallel=parallel,
			numprocs=numprocs, verbose=verbose)
		# analyse_topcMC.set_mc_interval(MC_int)
		analyse_default(analyse_topcMC, N_bs, bs_index_lists=bs_index_lists)

def pre_analysis(parameters):
	"""
	Function for starting flow analyses.

	Args:
		parameters: dictionary containing following elements: batch_name, 
			batch_folder, observables, NCfgs, obs_file, load_file, 
			save_to_binary, base_parameters, flow_epsilon, NFlows,
			create_perflow_data, correct_energy
	"""

	# Analysis timers
	pre_time = time.clock()
	observable_strings = []

	# Retrieves analysis parameters
	batch_name = parameters["batch_name"]
	batch_folder = parameters["batch_folder"]
	figures_folder = parameters["figures_folder"]

	_bp = parameters["base_parameters"]
	parameters["lattice_size"] = parameters["N"]**3*parameters["NT"]

	# Retrieves data
	obs_data = DataReader(batch_name, batch_folder, figures_folder, 
		load_binary_file=parameters["load_file"],
		save_to_binary=parameters["save_to_binary"],
		flow_epsilon=parameters["flow_epsilon"], NCfgs=parameters["NCfgs"],
		create_perflow_data=parameters["create_perflow_data"],
		correct_energy=parameters["correct_energy"],
		lattice_size=parameters["lattice_size"], verbose=_bp["verbose"],
		dryrun=_bp["dryrun"])

	# Writes a parameters file for the post analysis
	obs_data.write_parameter_file()

	print "="*100

	# Builds parameters list to be passed to analyser
	params = [obs_data, _bp["dryrun"], _bp["parallel"], _bp["numprocs"], 
		_bp["verbose"], _bp["N_bs"]]

	# Runs through the different observables and analyses each one
	if "plaq" in parameters["observables"]:
		analyse_plaq(params)
	if "energy" in parameters["observables"]:
		analyse_energy(params)

	# Topological charge definitions
	if "topc" in parameters["observables"]:
		analyse_topc(params)
	if "topc2" in parameters["observables"]:
		analyse_topc2(params)
	if "topc4" in parameters["observables"]:
		analyse_topc4(params)
	if "topcr" in parameters["observables"]:
		analyse_topcr(params)
	if "topct" in parameters["observables"]:
		analyse_topct(params, parameters["num_t_euclidean_indexes"])
	if "topcte" in parameters["observables"]:
		analyse_topcte_intervals(params, parameters["numsplits_eucl"], 
			parameters["intervals_eucl"])
	if "topcMC" in parameters["observables"]:
		analyse_topcMCTime(params, numsplits=parameters["MC_time_splits"],
			intervals=parameters["MCInt"])

	# Topological susceptibility definitions
	if "topsus" in parameters["observables"]:
		analyse_topsus(params)
	# if "topsus4" in parameters["observables"]:
	# 	analyse_topsus4(params)
	if "topsust" in parameters["observables"]:
		analyse_topsust(params, parameters["num_t_euclidean_indexes"])
	if "topsuste" in parameters["observables"]:
		analyse_topsuste_intervals(params, parameters["numsplits_eucl"], 
			parameters["intervals_eucl"])
	if "topsusMC" in parameters["observables"]:
		analyse_topsusMCTime(params, numsplits=parameters["MC_time_splits"],
			intervals=parameters["MCInt"])
	if "topsusqtq0" in parameters["observables"]:
		analyse_topsus_qtq0(params, parameters["q0_flow_times"])

	# Other definitions
	if "qtq0e" in parameters["observables"]:
		analyse_qtq0e(params, parameters["q0_flow_times"],
			parameters["euclidean_time_percents"])
	if "qtq0eff" in parameters["observables"]:
		analyse_qtq0_effective_mass(params, parameters["q0_flow_times"])

	gif_params = parameters["gif"]
	gif_observables = gif_params["gif_observables"]
	gif_euclidean_time = gif_params["gif_euclidean_time"]
	gif_flow_range = gif_params["gif_flow_range"]
	if len(gif_observables) != 0:
		obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

		if "qtq0e" in gif_observables:
			qtq0e_gif_analysis = QtQ0EGif(obs_data("topct"), 
				dryrun=dryrun, parallel=parallel, numprocs=numprocs, 
				verbose=False)
			gif_analysis(qtq0e_gif_analysis, gif_flow_range, N_bs, 
				gif_euclidean_time=gif_euclidean_time)
		
		if "qtq0eff" in gif_observables:
			qtq0eff_gif_analysis = QtQ0EffGif(obs_data("topct"),
				dryrun=dryrun, parallel=parallel, numprocs=numprocs, 
				verbose=False)
			gif_analysis(qtq0eff_gif_analysis, gif_flow_range, N_bs,
				gif_euclidean_time=None)


	post_time = time.clock()
	print "="*100
	print "Analysis of batch %s observables %s in %.2f seconds" % (batch_name,
		", ".join([i.lower() for i in parameters["observables"]]),
		(post_time-pre_time))
	print "="*100

def main():
	exit("No default run for pre_analyser.py is currently set up.")

if __name__ == '__main__':
	main()