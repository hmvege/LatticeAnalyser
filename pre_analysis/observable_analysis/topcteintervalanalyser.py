from pre_analysis.core.euclideanintervalanalysercore \
	import EuclideanIntervalAnalyser
import numpy as np

class TopcteIntervalAnalyser(EuclideanIntervalAnalyser):
	"""
	Analysis where one can split the topological in Euclidean time intervals
	to obtain an estimate of the topological charge.
	"""
	observable_name = "Topological Charge in Euclidean Time"
	observable_name_compact = "topcte"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
 	y_label = r"$\langle Q \rangle$"

	def __init__(self, *args, **kwargs):
		super(TopcteIntervalAnalyser, self).__init__(*args, **kwargs)
		self.NT = self.y_original.shape[-1]

		self.post_analysis_folder_old = self.post_analysis_folder
		self.observable_output_folder_path_old = \
			self.observable_output_folder_path

	def set_t_interval(self, *args):
		"""Runs first the inherited time setter function, then its own."""
		super(TopcteIntervalAnalyser, self).set_t_interval(*args)
		self.y = np.sum(self.y, axis=2)
		self.observable_name = (r"Q in Euclidean time $t=[%d,%d)$" 
			% self.t_interval)
