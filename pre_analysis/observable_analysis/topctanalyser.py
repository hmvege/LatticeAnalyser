from pre_analysis.core.flowanalyser import FlowAnalyser
from tools.folderreadingtools import check_folder
import copy
import os

class TopctAnalyser(FlowAnalyser):
	"""Analysis of the topological charge in Euclidean Time."""
	observable_name = "Topological Charge in Euclidean Time"
	observable_name_compact = "topct"
	x_label = r"$\sqrt{8t_{f}}[fm]$"
	y_label = r"$\langle Q_{t_\mathrm{euclidean}} \rangle$"

	def __init__(self, *args, **kwargs):
		super(TopctAnalyser, self).__init__(*args, **kwargs)
		self.y_original = copy.deepcopy(self.y)
		self.observable_output_folder_path_old = self.observable_output_folder_path
		self.post_analysis_folder_old = self.post_analysis_folder
		self.NT = self.y_original.shape[-1]

	def setEQ0(self, t_euclidean_index):
		"""
		Sets the Euclidean time we are to analyse for. Q_{t_E=}
		q_flow_time_zero_percent: float between 0.0 and 1.0, in which we choose what percentage point of the data we set as q0.
		E.g. if it is 0.9, it will be the Q that is closest to 90% of the whole flowed time.

		Args:
			t_euclidean_index: integer of what time point we will look at
		"""
		# Finds the euclidean time zero index
		self.t_euclidean_index = t_euclidean_index

		# Sets file name
		self.observable_name = r"$Q_{t_{euclidean}}$ at $i_{euclidean}=%d$" % self.t_euclidean_index

		# Manual method for multiplying the matrices
		self.y = copy.deepcopy(self.y_original[:,:,self.t_euclidean_index])

		# Creates a new folder to store t0 results in
		self.observable_output_folder_path = os.path.join(self.observable_output_folder_path_old, "%04d" % self.t_euclidean_index)
		check_folder(self.observable_output_folder_path, self.dryrun, self.verbose)

		# Checks that {post_analysis_folder}/{observable_name}/{time interval} exist
		self.post_analysis_folder = os.path.join(self.post_analysis_folder_old, "%04d" % self.t_euclidean_index)
		check_folder(self.post_analysis_folder, self.dryrun, self.verbose)

		# Resets some of the ac, jk and bs variable
		self.bootstrap_performed = False
		self.jackknife_performed = False
		self.autocorrelation_performed = False

	def __str__(self):
		info_string = lambda s1, s2: "\n{0:<20s}: {1:<20s}".format(s1, s2)
		return_string = ""
		return_string += "\n" + self.section_seperator
		return_string += info_string("Data batch folder", self.batch_data_folder)
		return_string += info_string("Batch name", self.batch_name)
		return_string += info_string("Observable", self.observable_name_compact)
		return_string += info_string("Beta", "%.2f" % self.beta)
		return_string += info_string("Euclidean time", "%d" % self.t_euclidean_index)
		return_string += "\n" + self.section_seperator
		return return_string
