from post_analysis.core.multiplotcore import MultiPlotCore
from post_analysis.core.topsuscore import TopsusCore
from tools.folderreadingtools import check_folder
import os

class TopsusQtQ0PostAnalysis(MultiPlotCore, TopsusCore):
	"""Post-analysis of the topsus at a fixed flow time."""
	observable_name = r"$\chi_{t_f}(\langle Q_{t_f} Q_{t_{f,0}} \rangle)^{1/4}$"
	observable_name_compact = "topsusqtq0"
	obs_name_latex = r"\chi_{t_f}^{1/4}\expect{Q_{t_f}Q_{t_{f,0}}}"
	x_label = r"$\sqrt{8t_{f}}$ [fm]"
	y_label = r"$\chi_{t_f}(\langle Q_{t_f} Q_{t_{f,0}} \rangle)^{1/4}$ [GeV]"
	sub_obs = True
	descr = "One Q at fixed flow time"
	subfolder_type = "tf"

	# Continuum plot variables
	y_label_continuum = r"$\chi^{1/4}(\langle Q_{t} Q_{t_0} \rangle)[GeV]$"

	def _convert_label(self, label):
		"""Short method for formatting time in labels."""
		try:
			return r"$t_{f}=%.2f$" % (float(label)/100)
		except ValueError:
			return r"$%s$" % label

def main():
	exit("Exit: TopsusQtQ0PostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()