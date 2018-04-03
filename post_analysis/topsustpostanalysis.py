from core.multiplotcore import MultiPlotCore
from core.topsuscore import TopsusCore
from tools.folderreadingtools import check_folder
import os

class TopsustPostAnalysis(MultiPlotCore, TopsusCore):
	"""Post-analysis of the topsus with with one Q at fixed euclidean time."""
	observable_name = "Topological Susceptibility with a fixed Euclidean Time"
	observable_name_compact = "topsust"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi^{1/4}(\langle Q_t Q_{t_{euclidean}} \rangle) [GeV]$"
	sub_obs = True

	# Continuum plot variables
	y_label_continuum = r"$\chi^{1/4}(\langle Q_t Q_{t_{euclidean}} \rangle)[GeV]$"

	def plot_continuum(self, fit_target, interval_index, title_addendum=""):
		# Backs up old variables
		self.plot_values_old = self.plot_values
		self.output_folder_path_old = self.output_folder_path

		# Sets plot values
		data = self._get_analysis_data(self.analysis_data_type)
		self._initiate_plot_values(data, interval_index=interval_index)

		t_euclidean = int(sorted(self.plot_values.values())[0]["interval"])

		self.output_folder_path = os.path.join(self.output_folder_path, "int%d" % t_euclidean)
		check_folder(self.output_folder_path, False, self.verbose)
		title_addendum = r", $t_{e}=%d$" % t_euclidean
		super(TopsustPostAnalysis, self).plot_continuum(fit_target, title_addendum=title_addendum)

		# Resets the plot values and output folder path
		self.plot_values = self.plot_values_old
		self.output_folder_path = self.output_folder_path_old

	def _convert_label(self, label):
		"""Short method for formatting time in labels."""
		try:
			return r"$t_e=%d$" % int(label)
		except ValueError:
			return r"$%s$" % label

def main():
	exit("Exit: TopsustPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()