from postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from tools.folderreadingtools import check_folder
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

class MultiPlotCore(PostCore):
	"""
	Class to be inheritedfrom in case we got intervals or sub elements of the 
	same observable.
	"""
	sub_obs = True
	analysis_data_type = "bootstrap"

	def _initiate_plot_values(self, data, data_raw, interval_index=None):
		# Sorts data into a format specific for the plotting method
		for beta in sorted(data.keys()):
			values = {}
			if interval_index == None:
				# Case where we have sub sections of observables, e.g. in 
				# euclidean time.
				for sub_obs in self.observable_intervals[beta]:
					if beta == 6.45: self.flow_time *= 2
					sub_values = {}
					sub_values["a"] = get_lattice_spacing(beta)
					sub_values["x"] = sub_values["a"]*np.sqrt(8*self.flow_time)
					sub_values["y"] = data[beta][sub_obs]["y"]
					sub_values["y_err"] = data[beta][sub_obs]["y_error"]
					sub_values["bs"] = data_raw[beta]\
						[self.observable_name_compact][sub_obs]
					sub_values["label"] = r"%s $\beta=%2.2f$ %s" % (
						self.size_labels[beta], beta, 
						self._convert_label(sub_obs))
					sub_values["color"] = self.colors[beta]
					values[sub_obs] = sub_values
			else:
				sorted_intervals = sorted(data[beta].keys())
				values["a"] = get_lattice_spacing(beta)
				values["x"] = values["a"]* np.sqrt(8*self.flow_time)
				values["y"] = data[beta] \
					[sorted_intervals[interval_index]]["y"]
				values["y_err"] = data[beta] \
					[sorted_intervals[interval_index]]["y_error"]
				values["bs"] = data_raw[beta][self.observable_name_compact] \
					[sorted_intervals[interval_index]]
				values["label"] = r"%s $\beta=%2.2f$ %s" % (
					self.size_labels[beta], beta, 
					self._convert_label(sorted_intervals[interval_index]))
				values["color"] = self.colors[beta]
				values["interval"] = sorted_intervals[interval_index]
			self.plot_values[beta] = values

	def _convert_label(self, label):
		"""Short method for formatting time in labels."""
		try:
			return r"$%d$" % int(label)
		except ValueError:
			return r"$%s$" % label

	def set_analysis_type(self, analysis_data_type):
		"""Sets a global analysis type."""
		self.analysis_data_type = analysis_data_type

	def plot_interval(self, interval_index, **kwargs):
		"""Sets and plots only one interval."""
		self.interval_index = interval_index
		self.plot_values = {}
		# data, _ = self._get_analysis_data(self.analysis_data_type)
		self._initiate_plot_values(self.data[self.analysis_data_type],
			self.data_raw[self.analysis_data_type],
			interval_index=interval_index)
		# Makes it a global constant so it can be added in plot figure name
		self.plot(**kwargs)

	def _get_plot_figure_name(self, output_folder=None):
		"""Retrieves appropriate figure file name."""
		if isinstance(output_folder, types.NoneType):
			output_folder = os.path.join(self.output_folder_path, "slices")
		check_folder(output_folder, False, True)
		fname = "post_analysis_%s_%s_int%d.png" % (self.observable_name_compact,
			self.analysis_data_type, self.interval_index)
		return os.path.join(output_folder, fname)

	def get_N_intervals(self):
		"""Returns possible intervals for us to plot."""
		if self.verbose:
			print "Intervals N=%d, possible for %s: " % (
				len(self.observable_intervals),
				self.observable_name_compact),

			print self.observable_intervals

		return (len(self.observable_intervals.values()[0]), 
			self.observable_intervals)

	def plot_series(self, indexes, beta="all", x_limits=False, 
		y_limits=False, plot_with_formula=False, error_shape="band"):
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
			error_shape: plot with error bands or with error bars.
		"""
		self.plot_values = {}
		self._initiate_plot_values(self.data[self.analysis_data_type],
			self.data_raw[self.analysis_data_type])

		old_rc_paramx = plt.rcParams['xtick.labelsize']
		old_rc_paramy = plt.rcParams['ytick.labelsize']
		plt.rcParams['xtick.labelsize'] = 6
		plt.rcParams['ytick.labelsize'] = 6

		# Starts plotting
		# fig = plt.figure(sharex=True)
		fig, axes = plt.subplots(2, 2, sharey=True, sharex=True)

		# Ensures beta is a list
		if not isinstance(beta, list):
			beta = [beta]

		# Sets the beta values to plot
		if beta[0] == "all" and len(beta) == 1:
			bvalues = self.plot_values
		else:
			bvalues = beta

		for ax, i in zip(list(itertools.chain(*axes)), indexes):
			for ibeta in bvalues:
				# Retrieves the values deepending on the indexes provided and
				# beta values.
				value = self.plot_values[ibeta] \
					[sorted(self.observable_intervals[ibeta])[i]]

				x = value["x"]
				y = value["y"]
				y_err = value["y_err"]
				# ax.plot(x, y, "-", label=value["label"], color=value["color"])
				# ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, edgecolor='',
				# 	facecolor=value["color"])
				
				if error_shape == "band":
					ax.plot(x, y, "-", label=value["label"], color=value["color"])
					ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, 
						edgecolor='', facecolor=value["color"])
				elif error_shape == "bars":
					ax.errorbar(x, y, yerr=y_err, capsize=5, fmt="_", ls=":", 
						label=value["label"], color=value["color"], 
						ecolor=value["color"])


				# Basic plotting commands
				ax.grid(True)
				ax.legend(loc="best", prop={"size":5})

				# Sets axes limits if provided
				if x_limits != False:
					ax.set_xlim(x_limits)
				if y_limits != False:
					ax.set_ylim(y_limits)

		# Set common labels
		# https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
		fig.text(0.52, 0.035, self.x_label, ha='center', va='center', 
			fontsize=9)
		fig.text(0.03, 0.5, self.y_label, ha='center', va='center', 
			rotation='vertical', fontsize=11)

		# Sets the title string
		title_string = r"%s" % self.observable_name
		if plot_with_formula:
			title_string += r" %s" % self.formula
		plt.suptitle(title_string)
		plt.tight_layout(pad=1.7)

		# Saves and closes figure
		if beta == "all":
			folder_name = "beta%s" % beta
		else:
			folder_name = "beta%s" % "-".join([str(i) for i in beta])
		folder_name += "_N%s" % "".join([str(i) for i in indexes])
		folder_path = os.path.join(self.output_folder_path, folder_name)
		check_folder(folder_path, False, True)

		fname = os.path.join(folder_path, "post_analysis_%s_%s.png" % (
			self.observable_name_compact, self.analysis_data_type))
		plt.savefig(fname, dpi=400)
		print "Figure saved in %s" % fname
		# plt.show()
		plt.close(fig)

		plt.rcParams['xtick.labelsize'] = old_rc_paramx
		plt.rcParams['ytick.labelsize'] = old_rc_paramy

	def _series_plot_core(self, indexes, beta="all", x_limits=False, 
		y_limits=False, plot_with_formula=False, error_shape="band", fname=None):
		"""
		Core structure of the series plot, allows to easily be expanded upon 
		by the needs of the different observables.
		"""
		her!!!
