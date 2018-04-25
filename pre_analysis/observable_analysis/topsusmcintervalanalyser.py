from pre_analysis.core.topsusanalysercore import TopsusAnalyserCore
from pre_analysis.core.mcintervalanalysercore import MCIntervalAnalyser

class TopsusMCIntervalAnalyser(MCIntervalAnalyser, TopsusAnalyserCore):
	"""
	Analysis where one can split the topological susceptibility in Monte Carlo
	time to obtain an estimate of the topological charge.
	"""
	observable_name = "Topological Susceptibility in MC Time"
	observable_name_compact = "topsusMC"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi^{1/4} [GeV]$"

	def __init__(self, *args, **kwargs):
		super(TopsusMCIntervalAnalyser, self).__init__(*args, **kwargs)
		self.NT = self.y_original.shape[-1]

		# Squares the values
		self.y_original **= 2

		self.observable_output_folder_path_old = \
			self.observable_output_folder_path

	def set_MC_interval(self, *args):
		"""Runs first the inherited time setter function, then its own."""
		super(TopsusMCIntervalAnalyser, self).set_MC_interval(*args)
		self.observable_name = (r"$\chi(\langle Q^2 \rangle)^{1/4}$ in MC "
			"interval $[%d,%d)$" % self.MC_interval)
