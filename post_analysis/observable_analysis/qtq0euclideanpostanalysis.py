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
	x_label = r"$t_e$"
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
		return float(lab[-4:])

	def set_gif_folder(self, gif_euclidean_time, gif_folder=None):
		"""
		Creates a folder for storing the smearing gif in.

		Args:
			gif_euclidean_time: float, what euclidean time percent we are
				creating the gif at.
			gif_folder: optional, default is None. will override default
				location.
		"""

		self.verbose = False
		if isinstance(gif_folder, types.NoneType):
			self.gif_folder = os.path.join(self.output_folder_path, "gif")
		else:
			self.gif_folder = gif_folder

	def plot_gif_image(self, q0_flow_time, euclidean_percent, **kwargs):
		"""Wrapper for plotting gifs."""
		kwargs["figure_folder"] = self.gif_folder
		self.plot_interval(q0_flow_time, euclidean_percent, **kwargs)

	def create_gif(self):
		"""Creates a gif from images in gif the gif folder."""
		fig_base_name = "post_analysis_%s_%s_tf*.png" % (
			self.observable_name_compact, self.analysis_data_type)

		input_paths = os.path.join(self.gif_folder, fig_base_name)
		cmd = ['convert', '-delay', '1', '-loop', '0', input_paths,
			self.output_folder_path]

		print "> %s" % " ".join(cmd)	
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		read_out = proc.stdout.read()

	def _initiate_plot_values(self, data, data_raw, euclidean_percent, 
		q0_flow_time=None):
		"""interval_index: int, should be in euclidean time."""

		# Sorts data into a format specific for the plotting method
		for beta in sorted(data.keys()):
			euclidean_index = self._get_euclidean_index(euclidean_percent, beta)
			te_index = "te%04d" % euclidean_index
			values = {}
			if q0_flow_time == None:
				# Case where we have sub sections of observables, e.g. in euclidean time
				for sub_obs in self.observable_intervals[beta]:
					sub_values = {}
					sub_values["a"] = get_lattice_spacing(beta)
					sub_values["x"] = np.linspace(0, 
						self.lattice_sizes[beta][1] * sub_values["a"], 
						self.lattice_sizes[beta][1])

					sub_values["y"] = data[beta][sub_obs][te_index]["y"]
					sub_values["y_err"] = data[beta][sub_obs][te_index]["y_error"]
					sub_values["bs"] = data_raw[beta] \
						[self.observable_name_compact][sub_obs][te_index]

					sub_values["label"] = r"%s, $\beta=%2.2f$, $\sqrt{8t_{f,0}}=%.2f$" \
						% (self.size_labels[beta], beta, self._convert_label(sub_obs))

					sub_values["color"] = self.colors[beta]
					values[sub_obs] = sub_values
				self.plot_values[beta] = values

			else:
				tf_index = "tflow%04.4f" % q0_flow_time
				values = {}
				values["a"] = get_lattice_spacing(beta)
				
				# FOR EXACT BOX SIZE:
				values["x"] = np.linspace(0,
					self.lattice_sizes[beta][1] * values["a"],
					self.lattice_sizes[beta][1])

				values["y"] = data[beta][tf_index][te_index]["y"]
				values["y_err"] = data[beta][tf_index][te_index]["y_error"]
				values["bs"] = data_raw[beta][self.observable_name_compact] \
					[tf_index][te_index]

				values["label"] = r"%s $\beta=%2.2f$, $t_f=%.2f$, $t_e=%d$" % (
					self.size_labels[beta], beta, q0_flow_time, euclidean_index)

				values["color"] = self.colors[beta]
				self.plot_values[beta] = values

	def set_analysis_data_type(self, euclidean_percent, analysis_data_type="bootstrap"):
		self.plot_values = {}

		# Makes it a global constant so it can be added in plot figure name
		self.analysis_data_type = analysis_data_type

		# Initiates plot values
		self._initiate_plot_values(self.data[self.analysis_data_type],
			self.data_raw[self.analysis_data_type],
			euclidean_percent=euclidean_percent)

	def _get_plot_figure_name(self, output_folder=None):
		"""Retrieves appropriate figure file name."""
		if isinstance(output_folder, types.NoneType):
			output_folder = os.path.join(self.output_folder_path, "slices")
		check_folder(output_folder, False, True)

		# Sets up euclidean time folder
		output_folder = os.path.join(output_folder,
			"te%04d" % self.interval_index[1])
		check_folder(output_folder, False, True)

		fname = "post_analysis_%s_%s_tf%4.4f.png" % (self.observable_name_compact,
			self.analysis_data_type, self.interval_index[0])
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

		# old_rc_paramx = plt.rcParams['xtick.labelsize']
		# old_rc_paramy = plt.rcParams['ytick.labelsize']
		# plt.rcParams['xtick.labelsize'] = 6
		# plt.rcParams['ytick.labelsize'] = 6

		# # Starts plotting
		# # fig = plt.figure(sharex=True)
		# fig, axes = plt.subplots(2, 2, sharey=True, sharex=True)

		# # Ensures beta is a list
		# if not isinstance(beta, list):
		# 	beta = [beta]

		# # Sets the beta values to plot
		# if beta[0] == "all" and len(beta) == 1:
		# 	bvalues = self.plot_values
		# else:
		# 	bvalues = beta

		# # print axes
		# for ax, i in zip(list(itertools.chain(*axes)), indexes):
		# 	for ibeta in bvalues:
		# 		# Retrieves the values deepending on the indexes provided and beta values
		# 		value = self.plot_values[ibeta] \
		# 			[sorted(self.observable_intervals[ibeta])[i]]

		# 		x = value["x"]
		# 		y = value["y"]
		# 		y_err = value["y_err"]
		# 		# ax.plot(x, y, "-", label=value["label"], color=value["color"])
		# 		# ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, edgecolor='',
		# 		# 	facecolor=value["color"])
		# 		if error_shape == "band":
		# 			ax.plot(x, y, "-", label=value["label"], color=value["color"])
		# 			ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, 
		# 				edgecolor='', facecolor=value["color"])
		# 		elif error_shape == "bars":
		# 			ax.errorbar(x, y, yerr=y_err, capsize=5, fmt="_", ls=":", 
		# 				label=value["label"], color=value["color"], 
		# 				ecolor=value["color"])

				
		# 		# Basic plotting commands
		# 		ax.grid(True)
		# 		ax.legend(loc="best", prop={"size":5})

		# 		# Sets axes limits if provided
		# 		if x_limits != False:
		# 			ax.set_xlim(x_limits)
		# 		if y_limits != False:
		# 			ax.set_ylim(y_limits)

		# # Set common labels
		# # https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
		# fig.text(0.52, 0.035, self.x_label, ha='center', va='center', 
		# 	fontsize=9)
		# fig.text(0.03, 0.5, self.y_label, ha='center', va='center', 
		# 	rotation='vertical', fontsize=11)

		# # Sets the title string
		# title_string = r"%s" % self.observable_name
		# if plot_with_formula:
		# 	title_string += r" %s" % self.formula
		# plt.suptitle(title_string)
		# plt.tight_layout(pad=1.7)

		# # Saves and closes figure
		# if beta == "all":
		# 	folder_name = "beta%s" % beta
		# else:
		# 	folder_name = "beta%s" % "-".join([str(i) for i in beta])
		# folder_name += "_N%s" % "".join([str(i) for i in indexes])
		# folder_path = os.path.join(self.output_folder_path, folder_name)
		# check_folder(folder_path, False, True)

		fname = "post_analysis_%s_%s_te%04d.png" % (
			self.observable_name_compact,
			self.analysis_data_type, euclidean_percent)

		self._series_plot_core(indexes, beta="all", x_limits=False, 
		y_limits=False, plot_with_formula=False, fname=fname)

		# plt.savefig(fname, dpi=400)
		# print "Figure saved in %s" % fname
		# # plt.show()
		# plt.close(fig)

		# plt.rcParams['xtick.labelsize'] = old_rc_paramx
		# plt.rcParams['ytick.labelsize'] = old_rc_paramy