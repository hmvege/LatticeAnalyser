from postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from tools.folderreadingtools import check_folder
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import types


class MultiPlotCore(PostCore):
	"""
	Class to be inheritedfrom in case we got intervals or sub elements of the 
	same observable.
	"""
	sub_obs = True
	analysis_data_type = "bootstrap"

	def _initiate_plot_values(self, data, data_raw, interval_index=None):
		"""Sorts data into a format specific for the plotting method."""

		for beta in sorted(data.keys()):
			values = {}
			if interval_index == None:
				# Case where we have sub sections of observables, e.g. in 
				# euclidean time.
				for sub_obs in self.observable_intervals[beta]:
					sub_values = {}
					sub_values["a"] = get_lattice_spacing(beta)
					sub_values["x"] = sub_values["a"] * \
						np.sqrt(8*data[beta][sub_obs]["x"])
					sub_values["y"] = data[beta][sub_obs]["y"]
					sub_values["y_err"] = data[beta][sub_obs]["y_error"]

					if self.with_autocorr:
						sub_values["tau_int"] = \
							data[beta][sub_obs]["ac"]["tau_int"]
						sub_values["tau_int_err"] = \
							data[beta][sub_obs]["ac"]["tau_int_err"]
					
					# Retrieves raw data
					sub_values["y_raw"] = \
						data_raw[beta][self.observable_name_compact][sub_obs]
					# sub_values[self.analysis_data_type] = \
					# 	data_raw[beta][self.observable_name_compact][sub_obs]
					
					sub_values["label"] = r"%s $\beta=%2.2f$ %s" % (
						self.size_labels[beta], beta, 
						self._convert_label(sub_obs))
					values[sub_obs] = sub_values
			else:
				sorted_intervals = sorted(data[beta].keys())

				# Modulo division in order to avoid going out of range in 
				# intervals.
				int_key = sorted_intervals[interval_index % len(sorted_intervals)]
				self.interval.append(int_key)

				values["a"] = get_lattice_spacing(beta)
				values["x"] = values["a"] * np.sqrt(8*data[beta][int_key]["x"])
				values["y"] = data[beta][int_key]["y"]
				values["y_err"] = data[beta][int_key]["y_error"]

				if self.with_autocorr:
					values["tau_int"] = data[beta][int_key]["ac"]["tau_int"]
					values["tau_int_err"] = data[beta] \
						[int_key]["ac"]["tau_int_err"]

				values["y_raw"] = \
					data_raw[beta][self.observable_name_compact][int_key]
				values["label"] = r"%s $\beta=%2.2f$ %s" % (
					self.size_labels[beta], beta, 
					self._convert_label(int_key))
				values["interval"] = int_key
			self.plot_values[beta] = values

	def _convert_label(self, label):
		"""Short method for formatting time in labels."""
		try:
			return r"$%d$" % float(label)
		except ValueError:
			return r"$%s$" % label

	def set_analysis_type(self, analysis_data_type):
		"""Sets a global analysis type."""
		self.analysis_data_type = analysis_data_type

	def plot_interval(self, interval_index, **kwargs):
		"""Sets and plots only one interval."""
		self.interval_index = interval_index
		self.interval = [] # Resets interval list
		self.plot_values = {}
		# data, _ = self._get_analysis_data(self.analysis_data_type)
		self._initiate_plot_values(self.data[self.analysis_data_type],
			self.data_raw[self.analysis_data_type],
			interval_index=interval_index)
		# Makes it a global constant so it can be added in plot figure name
		self.plot(**kwargs)

	def _get_plot_figure_name(self, output_folder=None, figure_name_appendix=""):
		"""Retrieves appropriate figure file name."""
		if isinstance(output_folder, types.NoneType):
			output_folder = os.path.join(self.output_folder_path, "slices")
		check_folder(output_folder, False, True)
		fname = "post_analysis_%s_%s_int%d%s.png" % (self.observable_name_compact,
			self.analysis_data_type, self.interval_index, figure_name_appendix)
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
				Options: band, bars
		"""
		self.plot_values = {}
		self._initiate_plot_values(self.data[self.analysis_data_type],
			self.data_raw[self.analysis_data_type])

		self._series_plot_core(indexes, beta=beta, x_limits=x_limits, 
		y_limits=y_limits, plot_with_formula=plot_with_formula, 
		error_shape=error_shape)

	def _series_plot_core(self, indexes, beta="all", x_limits=False, 
		y_limits=False, plot_with_formula=False, error_shape="band", fname=None):
		"""
		Core structure of the series plot, allows to easily be expanded upon 
		by the needs of the different observables.

		Args:
			indexes: list containing integers of which intervals to plot together.
			beta: beta values to plot. Default is "all". Otherwise, 
				a list of numbers or a single beta value is provided.
			x_limits: limits of the x-axis. Default is False.
			y_limits: limits of the y-axis. Default is False.
			plot_with_formula: bool, default is false, is True will look for 
				formula for the y-value to plot in title.
			error_shape: plot with error bands or with error bars. 
				Options: band, bars
			fname: str, figure name. Default is 
				post_analysis_{obs_name}_{analysis_type}.png
		"""

		old_rc_paramx = plt.rcParams['xtick.labelsize']
		old_rc_paramy = plt.rcParams['ytick.labelsize']
		plt.rcParams['xtick.labelsize'] = 6
		plt.rcParams['ytick.labelsize'] = 6

		# Starts plotting
		fig, axes = plt.subplots(2, 2, sharey=True, sharex=True)

		# Ensures beta is a list
		if not isinstance(beta, list):
			beta = [beta]

		# Sets the beta values to plot
		if beta[0] == "all" and len(beta) == 1:
			beta_values = self.plot_values
		else:
			beta_values = beta

		# Checks that we actually have enough different data points to plot
		comparer = lambda b, ind: len(self.plot_values[b]) > max(ind)
		asrt_msg = "Need at least %d different values. Currently have %d: %s" \
			% (max(indexes), len(self.plot_values.values()[0]), 
				", ".join(self.plot_values.values()[0].keys()))
		if not np.all([comparer(b, indexes) for b in beta_values]):
			print "WARNING:", asrt_msg
			return

		for ax, i in zip(list(itertools.chain(*axes)), indexes):
			for ibeta in beta_values:
				# Retrieves the values deepending on the indexes provided and
				# beta values.
				value = self.plot_values[ibeta] \
					[sorted(self.observable_intervals[ibeta])[i]]

				# Retrieves values to plot
				x = value["x"]
				y = value["y"]
				y_err = value["y_err"]
				
				if error_shape == "band":
					ax.plot(x, y, "-", label=value["label"], 
						color=self.colors[ibeta])
					ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, 
						edgecolor='', facecolor=self.colors[ibeta])
				elif error_shape == "bars":
					ax.errorbar(x, y, yerr=y_err, capsize=5, fmt="_", ls=":", 
						label=value["label"], color=self.colors[ibeta], 
						ecolor=self.colors[ibeta])
				else:
					raise KeyError("%s is not a valid error bar shape." % 
						error_shape)

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

		if isinstance(fname, types.NoneType):
			fpath = os.path.join(folder_path, "post_analysis_%s_%s.png" % (
				self.observable_name_compact, self.analysis_data_type))
		else:
			fpath = os.path.join(folder_path, fname)

		plt.savefig(fpath, dpi=self.dpi)
		if self.verbose:
			print "Figure saved in %s" % fpath
		# plt.show()
		plt.close(fig)

		plt.rcParams['xtick.labelsize'] = old_rc_paramx
		plt.rcParams['ytick.labelsize'] = old_rc_paramy


if __name__ == '__main__':
	exit(("Module %s intended to be called from a derived analysis, that in "
		"turn should be run from LQCDAnalyser. \nExiting"))