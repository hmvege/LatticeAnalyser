from post_analysis.core.multiplotcore import MultiPlotCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import itertools
import types
import subprocess

class QtQ0EuclideanPostAnalysis(MultiPlotCore):
	"""Class for plotting different QteQte0 a specific flow time together."""
	observable_name = ""
	observable_name_compact = "qtq0e"
	x_label = r"$t_e[fm]$"
	y_label = r"$\langle Q_{t_e} Q_{t_{e,0}} \rangle$" # $\chi_t^{1/4}[GeV]$
	sub_obs = True
	sub_sub_obs = True

	def __init__(self, *args, **kwargs):
		super(QtQ0EuclideanPostAnalysis, self).__init__(*args, **kwargs)

	def _get_euclidean_index(self, euclidean_percent, beta):
		"""Internal method for getting the euclidean index."""
		euclidean_index = int(self.lattice_sizes[beta][1] * euclidean_percent)
		if euclidean_index != 0:
		 	euclidean_index -= 1
		return euclidean_index

	def _convert_label(self, lab):
		return float(lab[-6:])

	# def set_gif_folder(self, gif_euclidean_time, gif_folder=None):
	# 	"""
	# 	Creates a folder for storing the smearing gif in.

	# 	Args:
	# 		gif_euclidean_time: float, what euclidean time percent we are
	# 			creating the gif at.
	# 		gif_folder: optional, default is None. will override default
	# 			location.
	# 	"""

	# 	self.verbose = False
	# 	if isinstance(gif_folder, types.NoneType):
	# 		self.gif_folder = os.path.join(self.output_folder_path, "gif")
	# 	else:
	# 		self.gif_folder = gif_folder

	# def plot_gif_image(self, q0_flow_time, euclidean_percent, **kwargs):
	# 	"""Wrapper for plotting gifs."""
	# 	kwargs["figure_folder"] = self.gif_folder
	# 	self.plot_interval(q0_flow_time, euclidean_percent, **kwargs)

	# def create_gif(self):
	# 	"""Creates a gif from images in gif the gif folder."""
	# 	fig_base_name = "post_analysis_%s_%s_tf*.png" % (
	# 		self.observable_name_compact, self.analysis_data_type)

	# 	gif_path = os.path.join(self.output_folder_path,
	# 		"%s_smearing.gif" % self.observable_name_compact)

	# 	input_paths = os.path.join(self.gif_folder, fig_base_name)
	# 	cmd = ['convert', '-delay', '4', '-loop', '0', input_paths,
	# 		gif_path]

	# 	print "> %s" % " ".join(cmd)	
	# 	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	# 	read_out = proc.stdout.read()
	# 	print read_out
	# 	print "\nGif creation done.\n"

	def _initiate_plot_values(self, data, data_raw, euclidean_percent, 
		q0_flow_time=None):
		"""interval_index: int, should be in euclidean time."""

		# Sorts data into a format specific for the plotting method
		for beta in sorted(data.keys()):
			euclidean_index = self._get_euclidean_index(euclidean_percent, beta)
			te_index = "te%04d" % euclidean_index
			values = {}
			if q0_flow_time == None:
				# Case where we have sub sections of observables, 
				# e.g. in euclidean time.
				for sub_obs in self.observable_intervals[beta]:
					sub_values = {}
					sub_values["a"], sub_values["a_err"] = get_lattice_spacing(beta)
					sub_values["x"] = np.linspace(0, 
						self.lattice_sizes[beta][1] * sub_values["a"], 
						self.lattice_sizes[beta][1])

					sub_values["y"] = data[beta][sub_obs][te_index]["y"]
					sub_values["y_err"] = data[beta][sub_obs][te_index]["y_error"]

					sub_values["y_raw"] = data_raw[beta] \
						[self.observable_name_compact][sub_obs]

					if self.with_autocorr:
						sub_values["tau_int"] = \
							data[beta][sub_obs][te_index]["ac"]["tau_int"]
						sub_values["tau_int_err"] = \
							data[beta][sub_obs][te_index]["ac"]["tau_int_err"]

					sub_values["label"] = r"%s, $\beta=%2.2f$, $t_{f,0}=%.2f$" \
						% (self.size_labels[beta], beta, self._convert_label(sub_obs))

					values[sub_obs] = sub_values
				self.plot_values[beta] = values

			else:
				tf_index = "tflow%04.4f" % q0_flow_time
				values = {}
				values["a"], values["a_err"] = get_lattice_spacing(beta)
				
				# FOR EXACT BOX SIZE:
				values["x"] = np.linspace(0,
					self.lattice_sizes[beta][1] * values["a"],
					self.lattice_sizes[beta][1])

				values["y"] = data[beta][tf_index][te_index]["y"]
				values["y_err"] = data[beta][tf_index][te_index]["y_error"]

				values["y_raw"] = data_raw[beta] \
					[self.observable_name_compact][tf_index]

				if self.with_autocorr:
					values["tau_int"] = data[beta][tf_index][te_index]["ac"]["tau_int"]
					values["tau_int_err"] = data[beta][tf_index][te_index]["ac"]["tau_int_err"]

				values["label"] = r"%s $\beta=%2.2f$, $t_f=%.2f$, $t_e=%d$" % (
					self.size_labels[beta], beta, q0_flow_time, euclidean_index)

				self.plot_values[beta] = values

	def set_analysis_data_type(self, euclidean_percent, analysis_data_type="bootstrap"):
		self.plot_values = {}

		# Makes it a global constant so it can be added in plot figure name
		self.analysis_data_type = analysis_data_type

		# Initiates plot values
		self._initiate_plot_values(self.data[self.analysis_data_type],
			self.data_raw[self.analysis_data_type],
			euclidean_percent=euclidean_percent)

	def _get_plot_figure_name(self, output_folder=None, figure_name_appendix=""):
		"""Retrieves appropriate figure file name."""
		if isinstance(output_folder, types.NoneType):
			# Sets up slices folder containing all euclidean times
			output_folder = os.path.join(self.output_folder_path, "slices")
			check_folder(output_folder, False, True)
	
			# Sets up euclidean time folder
			output_folder = os.path.join(output_folder,
				"te%04d" % self.interval_index[1])

		check_folder(output_folder, False, True)

		fname = "post_analysis_%s_%s_tf%4.4f%s.png" % (self.observable_name_compact,
			self.analysis_data_type, self.interval_index[0], figure_name_appendix)
		return os.path.join(output_folder, fname)

	def plot_interval(self, q0_flow_time, euclidean_percent, **kwargs):
		"""
		Sets and plots only one interval.

		Args:
			q0_flow_time: float, flow time
			euclidean_index: integer for euclidean time
		"""
		self.interval_index = [q0_flow_time, euclidean_percent]
		self.plot_values = {}
		self._initiate_plot_values(self.data[self.analysis_data_type],
			self.data_raw[self.analysis_data_type],
			euclidean_percent=euclidean_percent, q0_flow_time=q0_flow_time)

		# Sets the x-label to proper units
		x_label_old = self.x_label
		self.x_label = r"$t_f[fm]$"

		# Makes it a global constant so it can be added in plot figure name
		self.plot(**kwargs)

		self.x_label = x_label_old

	def plot_series(self, euclidean_percent, indexes, beta="all", x_limits=False, 
		y_limits=False, plot_with_formula=False):
		"""
		Method for plotting 4 axes together.

		Args:
			indexes: list containing integers of which intervals to plot together.
			beta: beta values to plot. Default is "all". Otherwise, 
				a list of numbers or a single beta value is provided.
			x_limits: limits of the x-axis. Default is False.
			y_limits: limits of the y-axis. Default is False.
			plot_with_formula: bool, default is false, is True will look for 
				formula for the y-value to plot in title.
		"""

		self.plot_values = {}
		self._initiate_plot_values(self.data[self.analysis_data_type],
			self.data_raw[self.analysis_data_type],
			euclidean_percent=euclidean_percent)

		fname = "post_analysis_%s_%s_te%04d.png" % (
			self.observable_name_compact,
			self.analysis_data_type, euclidean_percent)

		self._series_plot_core(indexes, beta="all", x_limits=False, 
		y_limits=False, plot_with_formula=False, fname=fname)

def main():
	exit("%s not intendd for standalone usage." % __name__)

if __name__ == '__main__':
	main()