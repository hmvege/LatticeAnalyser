from core.multiplotcore import MultiPlotCore
from tools.folderreadingtools import check_folder
import os

class QtQ0EuclideanPostAnalysis(MultiPlotCore):
	"""Class for plotting different QteQte0 a specific flow time together."""
	observable_name = ""
	observable_name_compact = "qtq0e"
	x_label = r"$t_e$"
	y_label = r"$\langle Q_{t_e} Q_{t_{e,0}} \rangle$" # $\chi_t^{1/4}[GeV]$
	sub_obs = True

