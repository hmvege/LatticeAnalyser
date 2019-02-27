from pre_analysis.observable_analysis.qtq0effectivemass import \
    QtQ0EffectiveMassAnalyser
from tools.folderreadingtools import check_folder
import statistics.parallel_tools as ptools
import copy
import numpy as np
import os

class QtQ0EffGif(QtQ0EffectiveMassAnalyser):
	"""Correlator of <QtQ0> in euclidean time analysis class."""
	observable_name = r""
	observable_name_compact = "qtq0eff_gif"
	x_label = r"$t_e$ [fm]"
	y_label = r"$am_\mathrm{eff} = \ln \frac{\langle Q_{t_e} Q_0 \rangle}{\langle Q_{t_e+1} Q_0 \rangle}$"
	mark_interval = 1
	error_mark_interval = 1

def main():
	exit("Module QtQ0EffGif not intended for standalone usage.")

if __name__ == '__main__':
	main()