from pre_analysis import *

__all__ = ['analyse_energy', 'analyse_plaq', 'analyse_qtq0_effective_mass',
	'analyse_qtq0e', 'analyse_topc', 'analyse_topc2', 'analyse_topc4',
	'analyse_topcMCTime', 'analyse_topct', 'analyse_topcte_intervals',
	'analyse_topsus', 'analyse_topsus4', 'analyse_topsusMCTime', 
	'analyse_topsus_qtq0', 'analyse_topsust', 'analyse_topsuste_intervals']

def analyse_default(analysis_object, N_bs, NBins=30, skip_histogram=False):
	print analysis_object
	analysis_object.boot(N_bs)
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

def main():
	exit("No default run for pre_analyser.py is currently set up.")

if __name__ == '__main__':
	main()