from pre_analysis.observable_analysis.qtq0euclideananalyser import \
    QtQ0EuclideanAnalyser
from tools.folderreadingtools import check_folder
import copy
import numpy as np
import os
import statistics.parallel_tools as ptools

class QtQ0EGif(QtQ0EuclideanAnalyser):
	"""Correlator of <QtQ0> in euclidean time analysis class."""
	observable_name = r""
	observable_name_compact = "qtq0e_gif"
	x_label = r"$t_e[fm]$"
	y_label = r"$\langle Q_{t_e} Q_{t_{e,0}} \rangle [GeV]$"
	mark_interval = 1
	error_mark_interval = 1

def main():
	exit("Module QtQ0EGif not intended for standalone usage.")

if __name__ == '__main__':
	main()