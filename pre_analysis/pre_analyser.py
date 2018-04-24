from observable_analysis import *

def analyse_default(analysis_object, N_bs, NBins=30, skip_histogram=False,
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
		analysis_object.plot_multihist([hist_pts[0], hist_pts[2], hist_pts[-1]],
			NBins=NBins)
	analysis_object.plot_integrated_correlation_time()
	analysis_object.plot_integrated_correlation_time()
	analysis_object.save_post_analysis_data() # save_as_txt=False

def analyse_plaq(params):
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	plaq_analysis = PlaquetteAnalyser(obs_data("plaq"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(plaq_analysis, N_bs)

def analyse_energy(params):
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	energy_analysis = EnergyAnalyser(obs_data("energy"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(energy_analysis, N_bs)

def analyse_topsus(params):
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	topsus_analysis = TopsusAnalyser(obs_data("topc"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(topsus_analysis, N_bs)

def analyse_topc(params):
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	topc_analysis = TopcAnalyser(obs_data("topc"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)

	if topc_analysis.beta == 6.0:
		topc_analysis.y_limits = [-9, 9]
	elif topc_analysis.beta == 6.1:
		topc_analysis.y_limits = [-12, 12]
	elif topc_analysis.beta == 6.2:
		topc_analysis.y_limits = [-12, 12]
	else:
		topc_analysis.y_limits = [None, None]

	analyse_default(topc_analysis, N_bs, NBins=150)

def analyse_topc2(params):
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params 
	topc2_analysis = Topc2Analyser(obs_data("topc"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)

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
	topc4_analysis = Topc4Analyser(obs_data("topc"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(topc4_analysis, N_bs)

def analyse_topcr(params):
	"""
	Analysis of the ratio with R=q4c/q2 of the topological charge. Performs an
	analysis on Q^2 and Q^4 with the same bootstrap samples, such that an post
	analysis can be performed on these explisitly.
	"""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	# topcr_analysis = TopcrAnalyser(obs_data("topc"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)

	topc2_analysis = Topc2Analyser(obs_data("topc"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)
	topc4_analysis = Topc4Analyser(obs_data("topc"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)

	N_cfgs_topc2 = topc2_analysis.N_configurations
	N_cfgs_topc4 = topc4_analysis.N_configurations
	assert N_cfgs_topc2 == N_cfgs_topc4, "NCfgs differ in topc2 and topc4."
	bs_index_lists = np.random.randint(N_cfgs_topc2,
		size=(N_bs, N_cfgs_topc2))

	analyse_default(topc2_analysis, N_bs, NBins=150, bs_index_lists=bs_index_lists)
	analyse_default(topc4_analysis, N_bs, bs_index_lists=bs_index_lists)

def analyse_topsus4(params):
	"""Analysis topological susceptiblity with q^4."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params
	topsus4_analysis = Topsus4Analyser(obs_data("topc"), dryrun=dryrun, parallel=parallel, numprocs=numprocs, verbose=verbose)
	analyse_default(topsus4_analysis, N_bs)

def analyse_topsus_qtq0(params, q0_flow_times):
	"""
	Analysis the topological susceptiblity with one charge q0 set a given 
	flow time.
	"""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

	qtq0_analysis = TopsusQtQ0Analyser(obs_data("topc"), dryrun=dryrun,
		parallel=parallel, numprocs=numprocs, verbose=verbose)

	for q0_flow_time_percent in q0_flow_times:
		qtq0_analysis.setQ0(q0_flow_time_percent) 
		analyse_default(qtq0_analysis, N_bs, skip_histogram=True)

def analyse_qtq0e(params, flow_time_indexes, euclidean_time_percents):
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

	qtq0_analysis = QtQ0EuclideanAnalyser(obs_data("topct"), dryrun=dryrun,
		parallel=parallel, numprocs=numprocs, verbose=verbose)

	for flow_time_index in flow_time_indexes:
		for euclidean_percent in euclidean_time_percents:
			qtq0_analysis.set_time(flow_time_index, euclidean_percent)
			analyse_default(qtq0_analysis, N_bs)

def analyse_qtq0_effective_mass(params, flow_time_indexes):
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

	qtq0eff_analysis = QtQ0EffectiveMassAnalyser(obs_data("topct"), dryrun=dryrun,
		parallel=parallel, numprocs=numprocs, verbose=verbose)

	for flow_time_index in flow_time_indexes:
		qtq0eff_analysis.set_time(flow_time_index)
		analyse_default(qtq0eff_analysis, N_bs)

def analyse_topct(params, numsplits):
	"""Analyses topological charge at a specific euclidean time."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

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

	analyse_topcte = TopcteIntervalAnalyser(obs_data("topct"),
		dryrun=dryrun, parallel=parallel, 
		numprocs=numprocs, verbose=verbose)

	# Sets up the intervals
	if (intervals == numsplits == None) or (intervals != None and numsplits != None):
		raise KeyError("Either provide intervals to plot for or the number of intervals to split into.")

	NT = analyse_topcte.NT
	if intervals == None:
		split_interval = NT/numsplits
		intervals = zip(
			range(0, NT+1, split_interval), 
			range(split_interval, NT+1, split_interval)
		)
		assert NT % numsplits == 0, "Bad number of splits: NT % numplits = %d " % (NT % numsplits)

	t_interval = iter(intervals)

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

	analyse_topsuste = TopsusteIntervalAnalyser(obs_data("topct"),
		dryrun=dryrun, parallel=parallel, 
		numprocs=numprocs, verbose=verbose)

	# Sets up the intervals
	if (intervals == numsplits == None) or (intervals != None and numsplits != None):
		raise KeyError("Either provide intervals to plot for or the number of intervals to split into.")

	NT = analyse_topsuste.NT
	if intervals == None:
		split_interval = NT/numsplits
		intervals = zip(
			range(0, NT+1, split_interval), 
			range(split_interval, NT+1, split_interval)
		)
		assert NT % numsplits == 0, "Bad number of splits: NT % numplits = %d " % (NT % numsplits)

	t_interval = iter(intervals)

	for t_int in t_interval:
		analyse_topsuste.set_t_interval(t_int)
		analyse_default(analyse_topsuste, N_bs)


def analyse_topcMCTime(params, numsplits=None, intervals=None):
	"""Analysis the topological charge in monte carlo time slices."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

	analyse_topcMC = TopcMCIntervalAnalyser(obs_data("topc"),
		dryrun=dryrun, parallel=parallel,
		numprocs=numprocs, verbose=verbose)

	# Sets up the intervals
	if (intervals == numsplits == None) or (intervals != None and numsplits != None):
		raise KeyError("Either provide MC intervals to plot for or the number of MC intervals to split into.")

	NCfgs = analyse_topcMC.N_configurations
	if intervals == None:
		split_interval = NCfgs/numsplits
		intervals = zip(
			range(0, NCfgs+1, split_interval), 
			range(split_interval, NCfgs+1, split_interval)
		)
		assert NCfgs % numsplits == 0, "Bad number of splits: NCfgs % numplits = %d " % (NCfgs % numsplits)

	MC_interval = iter(intervals)

	for MC_int in MC_interval:
		analyse_topcMC.set_MC_interval(MC_int)
		analyse_default(analyse_topcMC, N_bs)

def analyse_topsusMCTime(params, numsplits=None, intervals=None):
	"""Analysis the topological susceptibility in monte carlo time slices."""
	obs_data, dryrun, parallel, numprocs, verbose, N_bs = params

	analyse_topcMC = TopsusMCIntervalAnalyser(obs_data("topc"),
		dryrun=dryrun, parallel=parallel,
		numprocs=numprocs, verbose=verbose)

	# Sets up the intervals
	if (intervals == numsplits == None) or \
		(intervals != None and numsplits != None):

		raise KeyError(("Either provide MC intervals to plot for or the number"
			" of MC intervals to split into."))

	NCfgs = analyse_topcMC.N_configurations
	if intervals == None:
		split_interval = NCfgs/numsplits
		intervals = zip(
			range(0, NCfgs+1, split_interval), 
			range(split_interval, NCfgs+1, split_interval)
		)
		assert NCfgs % numsplits == 0, ("Bad number of splits: "
			"NCfgs % numplits = %d " % (NCfgs % numsplits))

	MC_interval = iter(intervals)

	for MC_int in MC_interval:
		analyse_topcMC.set_MC_interval(MC_int)
		analyse_default(analyse_topcMC, N_bs)

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
	_latsize = parameters["lattice_size"].items()[0]
	parameters["lattice_sizes"][_latsize[0]] = _latsize[1]

	# Retrieves data
	obs_data = DataReader(batch_name, batch_folder, figures_folder, 
		load_file=parameters["load_file"],
		flow_epsilon=parameters["flow_epsilon"], NCfgs=parameters["NCfgs"],
		create_perflow_data=parameters["create_perflow_data"],
		verbose=_bp["verbose"], dryrun=_bp["dryrun"],
		correct_energy=parameters["correct_energy"],
		lattice_sizes=parameters["lattice_sizes"])

	# Writes a parameters file for the post analysis
	obs_data.write_parameter_file()

	# Writes raw observable data to a single binary file
	if parameters["save_to_binary"] and not parameters["load_file"]:
		obs_data.write_single_file()
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
		analyse_topcMCTime(params, parameters["MC_time_splits"])

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
		analyse_topsusMCTime(params, parameters["MC_time_splits"])
	if "topsusqtq0" in parameters["observables"]:
		analyse_topsus_qtq0(params, parameters["q0_flow_times"])

	# Other definitions
	if "qtq0e" in parameters["observables"]:
		analyse_qtq0e(params, parameters["flow_time_indexes"],
			parameters["euclidean_time_percents"])
	if "qtq0eff" in parameters["observables"]:
		analyse_qtq0_effective_mass(params, parameters["eff_mass_flow_times"])
	

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