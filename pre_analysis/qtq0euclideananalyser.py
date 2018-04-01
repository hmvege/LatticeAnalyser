from core.flowanalyser import FlowAnalyser
import copy
import numpy as np
import os
from tools.folderreadingtools import check_folder
import statistics.parallel_tools as ptools

class QtQ0EuclideanAnalyser(FlowAnalyser):
	"""Correlator of <QtQ0> in euclidean time analysis class."""
	observable_name = r"$\langle Q_{t_e} Q_{t_{e,0}} \rangle$"
	observable_name_compact = "qtq0e"
	x_label = r"$t_e[fm]$"
	y_label = r"$\langle Q_{t_e} Q_{t_{e,0}} \rangle [GeV]$"
	mark_interval = 1
	error_mark_interval = 1

	def __init__(self, *args, **kwargs):
		super(QtQ0EuclideanAnalyser, self).__init__(*args, **kwargs)
		self.y_original = copy.deepcopy(self.y)
		self.NT = self.y_original.shape[-1]

		# Stores old variables for resetting at each new flow time
		self.N_configurations_old = self.N_configurations
		self.observable_output_folder_path_old = self.observable_output_folder_path
		self.post_analysis_folder_old = self.post_analysis_folder
		self.NFlows_old = self.NFlows
		self.x_old = self.x

	def set_flow_time(self, flow_time_index, euclidean_percent):
		"""Function for setting the flow time we will plot in euclidean time."""

		# Finds the q flow time zero value
		self.flow_time_index = flow_time_index
		self.flow_time = self.x_old[flow_time_index]
		self.euclidean_time = int((self.NT - 1) * euclidean_percent)

		# Restores y from original data
		self.y = copy.deepcopy(self.y_original)
		self.y = self.y[:,flow_time_index,:]

		# Sets the number of flows as the number of euclidean time slices,
		# as that is now what we are plotting in.
		assert self.y.shape[1] == self.NT, "the first row does not NT."
		self.NFlows = self.NT

		self.V = self.lattice_sizes[self.beta] / float(self.NT)
		self.const = (self.hbarc**4)/(self.a**4)/self.V
		self.const = 1.0 # Correlator contains no normalization
		self.function_derivative_parameters = {"const": self.const}

		self.function_derivative = ptools._C_derivative

		# Sets file name
		self.observable_name = r"$\langle Q_{t_e} Q_{t_{e,0}} \rangle$ at $t_e=%.2f$, $t_{flow}=%.2f$" % (self.euclidean_time, self.flow_time)
		# self.observable_name = r"$\chi(\langle Q_{t_e} Q_{t_{e,0}} \rangle)^{1/4}$ at $t_{flow}=%.2f$" % (self.flow_time)

		# Selects the configurations in euclidean time in flow time to multiply
		y_e0 = copy.deepcopy(self.y_original[:, self.flow_time_index, self.euclidean_time])

		# Multiplying QtQ0
		for iteuclidean in xrange(self.NFlows):
			self.y[:,iteuclidean] *= y_e0

		# Sets a new x-axis
		self.x = np.linspace(0, self.NFlows - 1, self.NFlows)

		# Sets up variables deependent on the number of configurations again
		self.unanalyzed_y = np.zeros(self.NFlows)
		self.unanalyzed_y_std = np.zeros(self.NFlows)
		self.unanalyzed_y_data = np.zeros((self.NFlows, self.N_configurations))

		# Resets bootstrap arrays
		self.bs_y = np.zeros(self.NFlows)
		self.bs_y_std = np.zeros(self.NFlows)

		# Resets jackknifed arrays
		self.jk_y = np.zeros(self.NFlows)
		self.jk_y_std = np.zeros(self.NFlows)

		# Resets autocorrelation arrays
		self.autocorrelations = np.zeros((self.NFlows, self.N_configurations/2))
		self.autocorrelations_errors = np.zeros((self.NFlows, self.N_configurations/2))
		self.integrated_autocorrelation_time = np.ones(self.NFlows)
		self.integrated_autocorrelation_time_error = np.zeros(self.NFlows)
		self.autocorrelation_error_correction = np.ones(self.NFlows)

		# Creates a new folder to store results in {beta}/{observable_name}/{flow time} exist
		self.observable_output_folder_path = os.path.join(self.observable_output_folder_path_old, "tflow%04d" % self.flow_time_index)
		check_folder(self.observable_output_folder_path, self.dryrun, self.verbose)

		# Creates a new folder to store results in {beta}/{observable_name}/{flow time}/{euclidean time} exist
		self.observable_output_folder_path = os.path.join(self.observable_output_folder_path, "te%04d" % self.euclidean_time)
		check_folder(self.observable_output_folder_path, self.dryrun, self.verbose)

		# Checks that {post_analysis_folder}/{observable_name}/{flow time} exist
		self.post_analysis_folder = os.path.join(self.post_analysis_folder_old, "tflow%04d" % self.flow_time_index)
		check_folder(self.post_analysis_folder, self.dryrun, self.verbose)

		# Checks that {post_analysis_folder}/{observable_name}/{flow time}/{euclidean time} exist
		self.post_analysis_folder = os.path.join(self.post_analysis_folder, "te%04d" % self.euclidean_time)
		check_folder(self.post_analysis_folder, self.dryrun, self.verbose)

		# Resets some of the ac, jk and bs variable
		self.bootstrap_performed = False
		self.jackknife_performed = False
		self.autocorrelation_performed = False

	def C(self, qtq0):
		"""Correlator for qtq0."""
		return self.const*qtq0

	def C_std(self, qtq0, qtq0_std):
		"""Correlator for qtq0 with error propagation."""
		return self.const*qtq0_std

	def jackknife(self, F=None, F_error=None, store_raw_jk_values=True):
		"""Overriding the jackknife class by adding the Correaltor function"""
		super(QtQ0EuclideanAnalyser, self).jackknife(F=self.C,
			F_error=self.C_std, store_raw_jk_values=store_raw_jk_values)

	def boot(self, N_bs, F=None, F_error=None, store_raw_bs_values=True):
		"""Overriding the bootstrap class by adding the Correaltor function"""
		super(QtQ0EuclideanAnalyser, self).boot(N_bs, F=self.C,
			F_error=self.C_std, store_raw_bs_values=store_raw_bs_values)

	def plot_jackknife(self, *args, **kwargs):
		"""Making sure we are plotting with in euclidean time."""
		kwargs["x"] = self.x
		super(QtQ0EuclideanAnalyser, self).plot_jackknife(*args, **kwargs)

	def plot_boot(self, *args, **kwargs):
		"""Making sure we are plotting with in euclidean time."""
		kwargs["x"] = self.x
		super(QtQ0EuclideanAnalyser, self).plot_boot(*args, **kwargs)

	# def autocorrelation(self, store_raw_ac_error_correction=True, method="wolff"):
	# 	"""Overriding the ac class."""
	# 	super(QtQ0EuclideanAnalyser, self).autocorrelation(store_raw_ac_error_correction=True, method="luscher")

	def __str__(self):
		info_string = lambda s1, s2: "\n{0:<20s}: {1:<20s}".format(s1, s2)
		return_string = ""
		return_string += "\n" + "="*100
		return_string += info_string("Data batch folder", self.batch_data_folder)
		return_string += info_string("Batch name", self.batch_name)
		return_string += info_string("Observable", self.observable_name_compact)
		return_string += info_string("Beta", "%.2f" % self.beta)
		return_string += info_string("Flow time:", "%.2f" % self.flow_time)
		return_string += info_string("Euclidean time:", "%d" % self.euclidean_time)
		return_string += "\n" + "="*100
		return return_string

def main():
	exit("Module QtQ0EuclideanAnalyser not intended for standalone usage.")

if __name__ == '__main__':
	main()