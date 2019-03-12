from post_analysis.core.multiplotcore import MultiPlotCore
from post_analysis.core.topsuscore import TopsusCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import os

class TopsusteIntervalPostAnalysis(MultiPlotCore, TopsusCore):
	"""Post-analysis of the topsus in euclidean time intervals."""
	observable_name = "Topological Susceptibility in Euclidean time intervals"
	observable_name_compact = "topsuste"
	obs_name_latex = r"\chi^{1/4}(\expect{Q^2_{E_\text{int}}})"
	x_label = r"$\sqrt{8t_{f}}$ [fm]"
	y_label = r"$\chi_{t_f}^{1/4}$ [GeV]"
	sub_obs = True
	descr = "Intervals in euclidean time"
	subfolder_type = "teInt"

	# def plot_continuum(self, fit_target, interval_keys, title_addendum="", **kwargs):
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

	# 	self.interval = str(sorted(self.plot_values.values())[0]["interval"])

	# 	# Sets up Euclidean intervals in usefull formats
	# 	self.intervals = [self.plot_values[i]["interval"] \
	# 		for i in sorted(self.plot_values.keys())]
	# 	self.intervals_str = ", ".join([i for i in self.intervals])
	# 	self.extra_continuum_msg = \
	# 		"\n    MC intervals: %s" % self.intervals_str

	# 	title_addendum = r", $t_{e}$-interval: $[%s)$" % self.interval
	# 	super(TopsusteIntervalPostAnalysis, self).plot_continuum(fit_target,
	# 		title_addendum=title_addendum, **kwargs)

	# 	# Resets the plot values and output folder path
	# 	self.plot_values = self.plot_values_old
	# 	self.output_folder_path = self.output_folder_path_old

	def _initialize_topsus_func_const(self):
		"""Sets the constant in the topsus function for found batch beta
		values."""
		for bn in self.batch_names:
			a, a_err = get_lattice_spacing(self.beta_values[bn])
			V = self.lattice_sizes[bn][0]**3
			V *= float(self.lattice_sizes[bn][1])/self.N_intervals
			self.chi_const[bn] = self.hbarc/a/V**0.25
			self.chi_const_err[bn] = self.hbarc*a_err/a**2/V**0.25

def main():
	exit(("Exit: TopsusteIntervalPostAnalysis not intended to be a "
		"standalone module."))

if __name__ == '__main__':
	main()