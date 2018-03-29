from flowanalyser import FlowAnalyser
from tools.folderreadingtools import check_folder
import copy
import os

class EuclideanIntervalAnalyser(FlowAnalyser):
	"""
	Base class for splitting in either Euclidean and creating separate data
	sets from that.
	"""
	def __init__(self, *args, **kwargs):
		super(EuclideanIntervalAnalyser, self).__init__(*args, **kwargs)
		self.y_original = copy.deepcopy(self.y)

	def set_t_interval(self, t_interval):
		self.y = copy.deepcopy(self.y_original)
		self.y = self.y[:,:,t_interval[0]:t_interval[1]]
		self.t_interval = t_interval

		# Creates a new folder to store t0 results in
		self.observable_output_folder_path = os.path.join(self.observable_output_folder_path_old, "int%03d-%03d" % t_interval)
		check_folder(self.observable_output_folder_path, self.dryrun, self.verbose)

		# Checks that {post_analysis_folder}/{observable_name}/{time interval} exist
		self.post_analysis_folder = os.path.join(self.post_analysis_folder_old, "%03d-%03d" % self.t_interval)
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
		return_string += info_string("Time interval", "[%d,%d)" % self.t_interval)
		return_string += "\n" + "="*100
		return return_string
