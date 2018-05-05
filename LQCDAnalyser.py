#!/usr/bin/env python2

from pre_analysis.pre_analyser import pre_analysis
from post_analysis.post_analyser import post_analysis
import copy
import os
import numpy as np

def get_num_observables(batch_folder, beta_folder):
	"""Gets the number of observable in a folder."""
	flow_path = os.path.join(batch_folder, beta_folder, "flow_observables")
	num_obs = []

	# If flow path do not exist, then we return
	if not os.path.isdir(flow_path):
		return 0

	# Loops over flow obs folders
	for flow_obs in os.listdir(flow_path):
		# Skips all hidden files, e.g. .DS_Store
		if flow_obs.startswith("."):
			continue

		# Gets flow observable files, should be equal to number of observables
		flow_obs_path = os.path.join(flow_path, flow_obs)
		flow_obs_files = os.listdir(flow_obs_path)

		# In case observable folder is empty
		if len(flow_obs_files) == 0:
			continue

		# Removes all hidden files, e.g. .DS_Store
		flow_obs_files = [f for f in flow_obs_files if not f.startswith(".")]

		num_obs.append(len(flow_obs_files))

	assert not sum([i - num_obs[0] for i in num_obs]), \
		"number of flow observables in each flow observable differ"

	return num_obs[0]

def main():
	#### Available observables
	observables = [
		"plaq", "energy", 
		# Topological charge definitions
		"topc", "topc2", "topc4", "topcr", "topct", "topcte", "topcMC",
		# Topological susceptibility definitions
		"topsus", "topsust", "topsuste", "topsusMC", "topsusqtq0",
		# Other quantities 
		"topcr",
		"qtq0e",
		"qtq0eff",
		# "qtq0_gif",
	]

	# observables = ["topsus", "topsust", "topsuste", "topsusMC", "topsusqtq0"]
	# observables = ["topc", "plaq", "energy", "topsus", "topcr"]
	# observables = ["topcr", "qtq0eff"]
	# observables = ["qtq0eff"]
	# observables = ["topcr"]
	# observables = ["topsust", "topsuste", "topsusqtq0"]
	observables = ["qtq0e", "qtq0eff", "topsusqtq0"]
	observables = ["topsus"]

	print 100*"=" + "\nObservables to be analysed: %s" % ", ".join(observables)
	print 100*"=" + "\n"

	#### Base parameters
	N_bs = 500
	dryrun = False
	verbose = True
	parallel = True
	numprocs = 8
	base_parameters = {"N_bs": N_bs, "dryrun": dryrun, "verbose": verbose, 
		"parallel": parallel, "numprocs": numprocs}

	#### Try to load binary file(much much faster)
	load_file = True

	# If we are to create per-flow datasets as opposite to per-cfg datasets
	create_perflow_data = False

	#### Save binary file
	save_to_binary = False

	#### Load specific parameters
	NFlows = 1000
	flow_epsilon = 0.01

	#### Post analysis parameters
	run_post_analysis = True
	line_fit_interval_points = 20
	# topsus_fit_targets = [0.3,0.4,0.5,0.58]
	topsus_fit_targets = [0.5, 0.6]
	energy_fit_target = 0.3

	# Smearing gif parameters for qtq0e
	gif_euclidean_time = 0.5
	gif_flow_range = np.linspace(0, 0.6, 100)

	#### Different batches
	# data_batch_folder = "data2"
	# data_batch_folder = "data4"
	# data_batch_folder = "../GluonAction/data5"
	# data_batch_folder = "../GluonAction/data6"
	data_batch_folder = "../GluonAction/data8"
	# data_batch_folder = "../topc_modes_8x16"
	figures_folder = "figures"
	# data_batch_folder = "../GluonAction/DataGiovanni"
	# data_batch_folder = "smaug_data_beta61"

	#### If we need to multiply
	if data_batch_folder == "DataGiovanni":
		if "topct" in observables:
			observables.remove("topct")
		correct_energy = False
		load_file = True
		save_to_binary = False
	else:
		correct_energy = True

	#### Different beta values folders:
	beta_folders = ["beta60", "beta61", "beta62", "beta645"]
	# beta_folders = ["beta6_0", "beta6_1", "beta6_2"]
	# beta_folders = ["beta61"]
	# beta_folders = ["beta60"]

	# Indexes to look at for topct.
	num_t_euclidean_indexes = 5

	# Number of different sectors we will analyse in euclidean time
	numsplits_eucl = 4
	intervals_eucl = None

	# Number of different sectors we will analyse in monte carlo time
	MC_time_splits = 4
 
	# Extraction point in sqrt(8*t) for q0 in qtq0
	q0_flow_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

	# Flow time indexes to plot qtq0 in euclidean time at
	euclidean_time_percents = [0, 0.25, 0.50, 0.75, 1.00]
	# euclidean_time_percents = [0]
	
	# Data types to be looked at in the post-analysis.
	post_analysis_data_type = ["bootstrap"]

	# eff_mass_flow_times = [0, 100, 400, 700, 999]

	#### Analysis batch setups
	default_params = {
		"batch_folder": data_batch_folder,
		"figures_folder": figures_folder,
		"observables": observables,
		"load_file": load_file,
		"save_to_binary": save_to_binary, 
		"base_parameters": base_parameters,
		"flow_epsilon": flow_epsilon, 
		"NFlows": NFlows,
		"create_perflow_data": create_perflow_data,
		"correct_energy": correct_energy,
		"num_t_euclidean_indexes": num_t_euclidean_indexes,
		"q0_flow_times": q0_flow_times,
		"euclidean_time_percents": euclidean_time_percents,
		"numsplits_eucl": numsplits_eucl,
		"intervals_eucl": intervals_eucl,
		"MC_time_splits": MC_time_splits,
		# Gif smearing parameters in the qtq0e observable
		"gif_euclidean_time": gif_euclidean_time,
		"gif_flow_range": gif_flow_range,
		# Passing on lattice sizes
		"lattice_sizes": {
			6.0: 24**3*48,
			6.1: 28**3*56,
			6.2: 32**3*64,
			6.45: 48**3*96,
		},
	}

	databeta60 = copy.deepcopy(default_params)
	databeta60["batch_name"] = beta_folders[0]
	databeta60["NCfgs"] = get_num_observables(data_batch_folder,
		beta_folders[0])
	databeta60["obs_file"] = "24_6.00"
	databeta60["lattice_size"] = {6.0: 24**3*48}

	databeta61 = copy.deepcopy(default_params)
	databeta61["batch_name"] = beta_folders[1]
	databeta61["NCfgs"] = get_num_observables(data_batch_folder,
		beta_folders[1])
	databeta61["obs_file"] = "28_6.10"
	databeta61["lattice_size"] = {6.1: 28**3*56}

	databeta62 = copy.deepcopy(default_params)
	databeta62["batch_name"] = beta_folders[2]
	databeta62["NCfgs"] = get_num_observables(data_batch_folder, 
		beta_folders[2])
	databeta62["obs_file"] = "32_6.20"
	databeta62["lattice_size"] = {6.2: 32**3*64}

	databeta645 = copy.deepcopy(default_params)
	databeta645["batch_name"] = beta_folders[3]
	databeta645["NCfgs"] = get_num_observables(data_batch_folder,
		beta_folders[3])
	databeta645["obs_file"] = "48_6.45"
	databeta645["lattice_size"] = {6.45: 48**3*96}

	# smaug_data_beta60_analysis = copy.deepcopy(default_params)
	# smaug_data_beta60_analysis["batch_name"] = beta_folders[0]
	# smaug_data_beta60_analysis["NCfgs"] = get_num_observables(data_batch_folder,
	# 	beta_folders[0])
	# smaug_data_beta60_analysis["obs_file"] = "8_6.00"
	# smaug_data_beta60_analysis["lattice_size"] = {6.0: 8**3*16}

	#### Adding relevant batches to args
	# analysis_parameter_list = [databeta60, databeta61, databeta62, databeta645]
	analysis_parameter_list = [databeta60, databeta61, databeta62]
	# analysis_parameter_list = [databeta645]
	# analysis_parameter_list = [databeta61, databeta62]
	# analysis_parameter_list = [smaug_data_beta61_analysis]


	# #### Submitting observable-batches
	# for analysis_parameters in analysis_parameter_list:
	# 	pre_analysis(analysis_parameters)

	#### Submitting post-analysis data
	if len(analysis_parameter_list) >= 3:
		post_analysis(data_batch_folder, beta_folders, observables, 
			topsus_fit_targets, line_fit_interval_points, energy_fit_target,
			q0_flow_times, euclidean_time_percents,
			post_analysis_data_type=post_analysis_data_type,
			figures_folder=figures_folder, gif_flow_range=gif_flow_range,
			gif_euclidean_time=gif_euclidean_time, verbose=verbose)

if __name__ == '__main__':
	main()