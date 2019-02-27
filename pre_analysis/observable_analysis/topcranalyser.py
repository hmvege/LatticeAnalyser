from pre_analysis.core.flowanalyser import FlowAnalyser
import copy
import numpy as np
import os
from tools.folderreadingtools import check_folder
import statistics.parallel_tools as ptools

class TopcrAnalyser(FlowAnalyser):
	"""Cumulant ratio <Q^4>_C/<Q^2> analysis class."""
	observable_name = r"$R=\frac{\langle Q^4 \rangle_C}{\langle Q^2 \rangle}$"
	observable_name_compact = "topqr"
	x_label = r"$\sqrt{8t_{f}}$ [fm]"
	y_label = r"$R=\frac{\langle Q^4 \rangle_C}{\langle Q^2 \rangle}$"

	formula = r", $\langle Q^4_C \rangle = \langle Q^4 \rangle - 3 \langle Q^2 \rangle^2 $"

	lattice_sizes = {6.0: 24**3*48, 6.1: 28**3*56, 6.2: 32**3*64, 6.45: 48**3*96}
	hbarc = 0.19732697 #eV micro m
	observable_name = r""
	observable_name_compact = "topcr"
	mark_interval = 1
	error_mark_interval = 1

	def __init__(self, *args, **kwargs):
		super(TopcrAnalyser, self).__init__(*args, **kwargs)
		self.y = (self.y**4 - 3*self.y**2) / self.y

	# def C(self, qtq0):
	# 	"""Correlator for qtq0."""
	# 	return self.const*qtq0

	# def C_std(self, qtq0, qtq0_std):
	# 	"""Correlator for qtq0 with error propagation."""
	# 	return self.const*qtq0_std

	# def jackknife(self, F=None, F_error=None, store_raw_jk_values=True):
	# 	"""Overriding the jackknife class by adding the Correaltor function"""
	# 	super(TopcrAnalyser, self).jackknife(F=self.C,
	# 		F_error=self.C_std, store_raw_jk_values=store_raw_jk_values)

	# def boot(self, N_bs, F=None, F_error=None, store_raw_bs_values=True):
	# 	"""Overriding the bootstrap class by adding the Correaltor function"""
	# 	super(TopcrAnalyser, self).boot(N_bs, F=self.C,
	# 		F_error=self.C_std, store_raw_bs_values=store_raw_bs_values)

	# def plot_jackknife(self, *args, **kwargs):
	# 	"""Making sure we are plotting with in euclidean time."""
	# 	kwargs["x"] = self.x
	# 	super(TopcrAnalyser, self).plot_jackknife(*args, **kwargs)

	# def plot_boot(self, *args, **kwargs):
	# 	"""Making sure we are plotting with in euclidean time."""
	# 	kwargs["x"] = self.x
	# 	super(TopcrAnalyser, self).plot_boot(*args, **kwargs)

	def __str__(self):
		info_string = lambda s1, s2: "\n{0:<20s}: {1:<20s}".format(s1, s2)
		return_string = ""
		return_string += "\n" + self.section_seperator
		return_string += info_string("Data batch folder", self.batch_data_folder)
		return_string += info_string("Batch name", self.batch_name)
		return_string += info_string("Observable", self.observable_name_compact)
		return_string += info_string("Beta", "%.2f" % self.beta)
		return_string += "\n" + self.section_seperator
		return return_string

def main():
	exit("Module TopcrAnalyser not intended for standalone usage.")

if __name__ == '__main__':
	main()