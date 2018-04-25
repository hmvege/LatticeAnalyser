from flowanalyser import FlowAnalyser
from tools.folderreadingtools import check_folder
import copy
import numpy as np
import os

class MCIntervalAnalyser(FlowAnalyser):
	"""Base class for analysing different parts of the Monte Carlo history."""
	def __init__(self, *args, **kwargs):
		super(MCIntervalAnalyser, self).__init__(*args, **kwargs)
		self.y_original = copy.deepcopy(self.y)
		self.N_configurations_old = self.N_configurations
		self.observable_output_folder_path_old = \
			self.observable_output_folder_path
		self.post_analysis_folder_old = self.post_analysis_folder

	def set_MC_interval(self, MC_interval):
		self.y = copy.deepcopy(self.y_original)
		self.y = self.y[MC_interval[0]:MC_interval[1],:]

		self.N_configurations = self.y.shape[0]

		# Sets up variables deependent on the number of configurations again
		self.unanalyzed_y_data = np.zeros(
			(self.NFlows, self.N_configurations))
		self.autocorrelations = np.zeros(
			(self.NFlows, self.N_configurations/2))
		self.autocorrelations_errors = np.zeros(
			(self.NFlows, self.N_configurations/2))

		self.MC_interval = MC_interval

		# Creates a new folder to store t0 results in
		self.observable_output_folder_path = os.path.join(
			self.observable_output_folder_path_old,
			"MCint%03d-%03d" % MC_interval)
		check_folder(self.observable_output_folder_path, 
			self.dryrun, self.verbose)

		# Checks that {post_analysis_folder}/{observable_name}/{time interval}
		# exists.
		self.post_analysis_folder = os.path.join(self.post_analysis_folder_old,
			"%03d-%03d" % self.MC_interval)
		check_folder(self.post_analysis_folder, self.dryrun, self.verbose)

		# Resets some of the ac, jk and bs variable
		self.bootstrap_performed = False
		self.jackknife_performed = False
		self.autocorrelation_performed = False

	def __str__(self):
		info_string = lambda s1, s2: "\n{0:<20s}: {1:<20s}".format(s1, s2)
		return_string = ""
		return_string += "\n" + "="*100
		return_string += info_string("Data batch folder",
			self.batch_data_folder)
		return_string += info_string("Batch name", self.batch_name)
		return_string += info_string("Observable",
			self.observable_name_compact)
		return_string += info_string("Beta", "%.2f" % self.beta)
		return_string += info_string("Time interval",
			"[%d,%d)" % self.MC_interval)
		return_string += "\n" + "="*100
		return return_string
