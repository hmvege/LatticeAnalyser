from post_analysis.core.multiplotcore import MultiPlotCore
from post_analysis.core.topsuscore import TopsusCore
from tools.folderreadingtools import check_folder
import os

class TopsusMCIntervalPostAnalysis(MultiPlotCore, TopsusCore):
	"""Post-analysis of the topsus in MC time intervals."""
	observable_name = "Topological Susceptibility in MC Time"
	observable_name_compact = "topsusMC"
	obs_name_latex = r"\chi^{1/4}(\expect{Q^2_{MC_\text{int}}})"
	x_label = r"$\sqrt{8t_{f}}[fm]$"
	y_label = r"$\chi^{1/4} [GeV]$"
	sub_obs = True
	descr = "Intervals in Monte Carlo time"
	subfolder_type = "MCInt"

	# def plot_continuum(self, fit_target, interval_keys, **kwargs):
	# 	"""
	# 	Continuum plotter for topsus in intervals in MC-time.

	# 	Args:
	# 		fit_target: float value at which we extrapolate to continuum from.
	# 		interval_index: int for a given interval specified from 
	# 			set_interval().
	# 	"""

	# 	# Backs up old variables
	# 	self.plot_values_old = self.plot_values
	# 	self.output_folder_path_old = self.output_folder_path

	# 	# Sets plot values
	# 	self._initiate_plot_values(self.data[self.analysis_data_type],
	# 		self.data_raw[self.analysis_data_type],
	# 		interval_keys=interval_keys)

	# 	self.output_folder_path = os.path.join(self.output_folder_path,
	# 		"int%s" % "_".join(interval_keys))

	# 	check_folder(self.output_folder_path, False, self.verbose)

	# 	# Sets up MC intervals in usefull formats
	# 	self.intervals = [self.plot_values[i]["interval"] \
	# 		for i in sorted(self.plot_values.keys())]
	# 	self.intervals_str = ", ".join([i for i in self.intervals])
	# 	self.extra_continuum_msg = \
	# 		"\n    MC intervals: %s" % self.intervals_str

	# 	title_addendum = ", MC-Intervals:" + self.intervals_str

	# 	super(TopsusMCIntervalPostAnalysis, self).plot_continuum(fit_target,
	# 		title_addendum=title_addendum, **kwargs)

	# 	# Resets the plot values and output folder path
	# 	self.plot_values = self.plot_values_old
	# 	self.output_folder_path = self.output_folder_path_old

def main():
	exit(("Exit: TopsusMCIntervalPostAnalysis not intended to be a standalone "
		"module."))

if __name__ == '__main__':
	main()