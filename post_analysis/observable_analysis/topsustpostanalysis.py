from post_analysis.core.multiplotcore import MultiPlotCore
from post_analysis.core.topsuscore import TopsusCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import os

class TopsustPostAnalysis(MultiPlotCore, TopsusCore):
	"""Post-analysis of the topsus with with one Q at fixed euclidean time."""
	observable_name = "Topological Susceptibility with a fixed Euclidean Time"
	observable_name_compact = "topsust"
	obs_name_latex = r"\chi^{1/4}(\expect{Q^2_{t_f}})"
	x_label = r"$\sqrt{8t_{f}}$ [fm]"
	y_label = r"$\chi_{t_f}^{1/4}(\langle Q_t Q_{t_\mathrm{euclidean}} \rangle)$ [GeV]"
	sub_obs = True
	descr = "One Q at fixed euclidean time"
	subfolder_type = "te"

	# Continuum plot variables
	y_label_continuum = r"$\chi^{1/4}(\langle Q_t Q_{t_\mathrm{euclidean}} \rangle)[GeV]$"

	def _initialize_topsus_func_const(self):
		"""Sets the constant in the topsus function for found batch beta
	 	values."""
		for bn in self.batch_names:
			a, a_err = get_lattice_spacing(self.beta_values[bn])
			V = float(self.lattice_sizes[bn][0]**3)
			self.chi_const[bn] = self.hbarc/a/V**0.25
			self.chi_const_err[bn] = self.hbarc*a_err/a**2/V**0.25

	def _convert_label(self, label):
		"""Short method for formatting time in labels."""
		try:
			return r"$t_e/a=%d$" % int(label)
		except ValueError:
			return r"$%s$" % label

def main():
	exit("Exit: TopsustPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()