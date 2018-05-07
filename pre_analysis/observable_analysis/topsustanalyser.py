from pre_analysis.core.topsusanalysercore import TopsusAnalyserCore
from tools.folderreadingtools import check_folder
import copy
import os
import numpy as np

class TopsustAnalyser(TopsusAnalyserCore):
	"""Analysis of the topological susceptibility in Euclidean Time."""
	observable_name = "Topological Susceptibility in Euclidean Time"
	observable_name_compact = "topsust"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi^{1/4}(\langle Q_t Q_{t_{euclidean}} \rangle) [GeV]$"

	def __init__(self, *args, **kwargs):
		super(TopsustAnalyser, self).__init__(*args, **kwargs)
		self.y_original = copy.deepcopy(self.y)
		self.observable_output_folder_path_old = \
			self.observable_output_folder_path
		self.NT = self.y_original.shape[-1]

	def setEQ0(self, t_euclidean_index):
		"""
		Sets the Euclidean time we are to analyse for. E.g. if it is 0.9, it 
		will be the Q that is closest to 90% of the total flowed time.

		Args:
			t_euclidean_index: integer of what time point we will look at
		"""

		# Finds the euclidean time zero index
		self.t_euclidean_index = t_euclidean_index

		self.V = self.lattice_sizes[self.beta] / float(self.NT)
		self.const = self.hbarc/self.a/self.V**(1./4)
		self.function_derivative_parameters = {"const": self.const}

		# Sets file name
		self.observable_name = r"$\chi(\langle Q_t Q_{t_{euclidean}}$"
		self.observable_name += r"$\rangle)^{1/4}$ at $i_{euclidean}=%d$" \
			% self.t_euclidean_index

		# Manual method for multiplying the matrices
		y_qe0 = copy.deepcopy(self.y_original[:,:,self.t_euclidean_index])
		self.y = copy.deepcopy(self.y_original)

		# Sums the euclidean time
		self.y = np.sum(self.y, axis=2)

		self.y *= y_qe0

		# self.y = np.abs(self.y) # If not absolute value, will get error!

		# Creates a new folder to store t0 results in
		self.observable_output_folder_path = os.path.join(
			self.observable_output_folder_path_old, 
			"%04d" % self.t_euclidean_index)
		check_folder(self.observable_output_folder_path, 
			self.dryrun, self.verbose)

		# Checks that {post_analysis_folder}/{observable_name}/{time interval}
		# exist.
		self.post_analysis_folder = os.path.join(
			self.post_analysis_folder_old, "%04d" % self.t_euclidean_index)
		check_folder(self.post_analysis_folder, self.dryrun, self.verbose)

		# Resets some of the ac, jk and bs variable
		self.bootstrap_performed = False
		self.jackknife_performed = False
		self.autocorrelation_performed = False

	def __str__(self):
		info_string = lambda s1, s2: "\n{0:<20s}: {1:<20s}".format(s1, s2)
		return_string = ""
		return_string += "\n" + "="*100
		return_string += info_string("Data batch folder", self.batch_data_folder)
		return_string += info_string("Batch name", self.batch_name)
		return_string += info_string("Observable", self.observable_name_compact)
		return_string += info_string("Beta", "%.2f" % self.beta)
		return_string += info_string("Euclidean time", "%d" % self.t_euclidean_index)
		return_string += "\n" + "="*100
		return return_string
