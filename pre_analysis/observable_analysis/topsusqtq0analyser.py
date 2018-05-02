from pre_analysis.core.topsusanalysercore import TopsusAnalyserCore
import copy
import numpy as np
import os
from tools.folderreadingtools import check_folder

class TopsusQtQ0Analyser(TopsusAnalyserCore):
	"""Topological susceptibility QtQ0 analysis class."""
	observable_name = r"$\chi(\langle Q_t Q_{t_0} \rangle)^{1/4}$"
	observable_name_compact = "topsusqtq0"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi(\langle Q_{t} Q_{t_0} \rangle)^{1/4} [GeV]$"

	def __init__(self, *args, **kwargs):
		super(TopsusQtQ0Analyser, self).__init__(*args, **kwargs)
		self.observable_output_folder_path_old = \
			self.observable_output_folder_path
		self.post_analysis_folder_old = self.post_analysis_folder
		self.y_original = copy.deepcopy(self.y)

	def __set_size(self):
		"""Function that sets the lattice size deepending on the beta value."""
		# Sets up constants used in the chi function for topsus
		self.function_derivative = ptools._chi_derivative
		self.V = self.lattice_sizes[self.beta] / float(self.NTemporal[self.beta])
		self.const = self.hbarc/self.a/self.V**(1./4)
		self.function_derivative_parameters = {"const": self.const}

	def setQ0(self, q0_flow_time, y_label=None):
		"""
		Sets the flow time we are to analyse for. E.g. if it is 0.9, it will 
		be the Q that is closest to 90% of the total flowed time.

		Args:
			q0_flow_time: float between 0.0 and 1.0, in which we 
			choose what percentage point of the data we set as q0.
		
		"""

		self._set_q0_time_and_index(q0_flow_time)

		self.plot_vline_at = self.q0_flow_time

		# Sets file name
		self.observable_name = (r"$\chi(\langle Q_t Q_{t_0} \rangle)^{1/4}$"
			" at $t=%.2f$" % (self.q0_flow_time))

		# Manual method for multiplying the matrices
		y_q0 = copy.deepcopy(self.y_original[:,self.q0_flow_time_index])
		self.y = copy.deepcopy(self.y_original)

		# Multiplying QtQ0
		for iFlow in xrange(self.y.shape[1]):
			self.y[:,iFlow] *= y_q0

		if y_label != None:
			self.y_label = y_label

		# Creates a new folder to store t0 results in
		self.observable_output_folder_path = os.path.join(
			self.observable_output_folder_path_old,
			"%04.2f" % self.q0_flow_time)
		check_folder(self.observable_output_folder_path, self.dryrun, 
			self.verbose)

		# Checks if {post_analysis_folder}/{observable_name}/{time interval}
		# exists.
		self.post_analysis_folder = os.path.join(self.post_analysis_folder_old,
			"%04.2f" % self.q0_flow_time)
		check_folder(self.post_analysis_folder, self.dryrun, self.verbose)

		# Resets some of the ac, jk and bs variable
		self.bootstrap_performed = False
		self.jackknife_performed = False
		self.autocorrelation_performed = False

	def jackknife(self, F=None, F_error=None, store_raw_jk_values=True):
		"""Overriding the jackknife class by adding the Correaltor function"""
		super(TopsusQtQ0Analyser, self).jackknife(F=self.chi,
			F_error=self.chi_std, store_raw_jk_values=store_raw_jk_values)

	def boot(self, N_bs, F=None, F_error=None, store_raw_bs_values=True, 
		index_lists=None):
		"""Overriding the bootstrap class by adding the Correaltor function"""
		super(TopsusQtQ0Analyser, self).boot(N_bs, F=self.chi,
			F_error=self.chi_std, store_raw_bs_values=store_raw_bs_values, 
			index_lists=index_lists)

	def __str__(self):
		info_string = lambda s1, s2: "\n{0:<20s}: {1:<20s}".format(s1, s2)
		return_string = ""
		return_string += "\n" + "="*100
		return_string += info_string("Data batch folder", self.batch_data_folder)
		return_string += info_string("Batch name", self.batch_name)
		return_string += info_string("Observable", self.observable_name_compact)
		return_string += info_string("Beta", "%.2f" % self.beta)
		return_string += info_string("Flow time t0", "%.2f" % self.q0_flow_time)
		return_string += "\n" + "="*100
		return return_string
