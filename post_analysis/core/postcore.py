from tools.postanalysisdatareader import PostAnalysisDataReader
from tools.latticefunctions import get_lattice_spacing
from tools.folderreadingtools import check_folder, get_NBoots
import matplotlib.pyplot as plt
import numpy as np
import os
import types

from matplotlib import rc
rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})

class PostCore(object):
	"""Post analysis base class. Based on paper DOI 10.1002/mas.20100."""
	observable_name = "Observable"
	observable_name_compact = "obs"
	formula = ""
	x_label = r""
	y_label = r""
	dpi = 350
	size_labels = {
		6.0  : r"$24^3 \times 48$",
		6.1  : r"$28^3 \times 56$",
		6.2  : r"$32^3 \times 64$",
		6.45 : r"$48^3 \times 96$",
	}
	lattice_sizes = {
		6.0  : [24, 48],
		6.1  : [28, 56],
		6.2  : [32, 64],
		6.45 : [48, 96],
	}
	r0 = 0.5
	sub_obs = False
	sub_sub_obs = False
	colors = {
		6.0: "#5cbde0", # blue
		6.1: "#6fb718", # green
		6.2: "#bc232e", # red
		6.45: "#8519b7" # purple
	}

	# font_type = {"fontname": "modern"}

	# colors = {
	# 	6.0: "#3366ff", # blue
	# 	6.1: "#00ffff", # green
	# 	6.2: "#ffff33", # red
	# 	6.45: "#ff3333" # purple
	# }

	interval = None

	def __init__(self, data, with_autocorr=True, figures_folder="../figures",
		verbose=False, dryrun=False):
		
		if with_autocorr:
			self.ac = "with_autocorr"
		else:
			self.ac = "without_autocorr"
		
		self.with_autocorr = with_autocorr
		observable = self.observable_name_compact

		self.verbose = verbose
		self.dryrun = dryrun

		self.beta_values = sorted(data.beta_values)

		# self._setup_flow_times()

		self._setup_analysis_types(data.analysis_types)

		self.data = {atype: {beta: {} for beta in self.beta_values} \
			for atype in self.analysis_types}
		
		# Only sets this variable if we have sub-intervals in order to avoid bugs.
		if self.sub_obs:
			self.observable_intervals = {}
			for beta in self.beta_values:
				self.observable_intervals[beta] = {}

		# Checks that the observable is among the available data
		assert_msg = ("%s is not among current data(%s). Have the pre analysis"
			" been performed?" % (observable, ", ".join(data.observable_list)))
		assert observable in data.observable_list, assert_msg

		for atype in self.analysis_types:
			for beta in self.beta_values:
				if self.sub_obs:
					if self.sub_sub_obs:
						for subobs in data.data_observables[observable][beta]:
							# Sets sub-sub intervals
							self.observable_intervals[beta][subobs] = \
								data.data_observables[observable] \
								[beta][subobs].keys()
							
							# Sets up additional subsub-dictionaries
							self.data[atype][beta][subobs] = {}

							for subsubobs in data.data_observables \
								[observable][beta][subobs]:

								self.data[atype][beta][subobs][subsubobs] = \
									data.data_observables[observable][beta] \
									[subobs][subsubobs][self.ac][atype]

								if self.with_autocorr:
									self.data[atype][beta][subobs][subsubobs] \
										["ac"] = data.data_observables \
										[observable][beta][subobs][subsubobs] \
										["with_autocorr"]["autocorr"]

					else:
						# Fills up observable intervals
						self.observable_intervals[beta] = \
							data.data_observables[observable][beta].keys()

						for subobs in data.data_observables[observable][beta]:

							self.data[atype][beta][subobs] = \
								data.data_observables[observable][beta] \
								[subobs][self.ac][atype]

							if self.with_autocorr:
								self.data[atype][beta][subobs]["ac"] = \
									data.data_observables[observable][beta] \
									[subobs]["with_autocorr"]["autocorr"]

				else:
					self.data[atype][beta] = data.data_observables \
						[observable][beta][self.ac][atype]

					if self.with_autocorr:
						self.data[atype][beta]["ac"] = \
							data.data_observables[observable][beta] \
							["with_autocorr"]["autocorr"]

		self.data_raw = {}
		for atype in data.raw_analysis:
			if atype == "autocorrelation":
				self.ac_raw = data.raw_analysis[atype]
			else:
				self.data_raw[atype] = data.raw_analysis[atype]

		# Small test to ensure that the number of bootstraps and number of 
		# different beta batches match
		err_msg = ("Number of bootstraps do not match number "
			"of different beta values")
		assert np.asarray([get_NBoots(self.data_raw["bootstrap"][i]) \
			for i in self.data_raw["bootstrap"].keys()]).all(), err_msg

		self.NBoots = get_NBoots(self.data_raw["bootstrap"])

		# Creates base output folder for post analysis figures
		self.figures_folder = figures_folder
		check_folder(self.figures_folder, dryrun=self.dryrun, 
			verbose=self.verbose)
		check_folder(os.path.join(self.figures_folder, data.batch_name),
			dryrun=self.dryrun, verbose=self.verbose)

		# Creates output folder
		self.post_anlaysis_folder = os.path.join(self.figures_folder, 
			data.batch_name, "post_analysis")
		check_folder(self.post_anlaysis_folder, dryrun=self.dryrun,
			verbose=self.verbose)

		# Creates observable output folder
		self.output_folder_path = os.path.join(self.post_anlaysis_folder,
			self.observable_name_compact)
		check_folder(self.output_folder_path, dryrun=self.dryrun, 
			verbose=self.verbose)

	def _setup_flow_times(self):
		"""Initializes flow times to be used by others."""
		self.flow_times = {b: np.arange(0, 10, 0.01) for b in self.beta_values}
		self.flow_times[6.45] = np.arange(0, 20, 0.02)

	def _setup_analysis_types(self, atypes):
		"""
		Stores the number of analysis types from the data container, while 
		removing the autocorrelation one.
		"""
		self.analysis_types = atypes
		if "autocorrelation" in self.analysis_types:
			self.analysis_types.remove("autocorrelation")

	def _check_plot_values(self):
		"""Checks if we have set the analysis data type yet."""
		if not hasattr(self, "plot_values"):
			raise AttributeError("set_analysis_data_type() has not been set yet.")

	def set_analysis_data_type(self, analysis_data_type="bootstrap"):
		"""Sets the analysis type and retrieves correct analysis data."""

		# Makes it a global constant so it can be added in plot figure name
		self.analysis_data_type = analysis_data_type

		self.plot_values = {} # Clears old plot values
		self._initiate_plot_values(self.data[analysis_data_type],
			self.data_raw[analysis_data_type])

	def _initiate_plot_values(self, data, data_raw):
		"""Sorts data into a format specific for the plotting method."""
		for beta in sorted(data.keys()):
			values = {}
			values["a"] = get_lattice_spacing(beta)
			values["x"] = values["a"]* np.sqrt(8*data[beta]["x"])
			values["y"] = data[beta]["y"]
			values["y_err"] = data[beta]["y_error"]
			values["y_raw"] = data_raw[beta][self.observable_name_compact]
			if self.with_autocorr:
				values["tau_int"] = data[beta]["ac"]["tau_int"]
				values["tau_int_err"] = data[beta]["ac"]["tau_int_err"]
			values["label"] = r"%s $\beta=%2.2f$" % (
				self.size_labels[beta], beta)
			values["color"] = self.colors[beta]

			self.plot_values[beta] = values

	def plot(self, x_limits=False, y_limits=False, plot_with_formula=False,
		error_shape="band", figure_folder=None, plot_vline_at=None):
		"""
		Function for making a basic plot of all the different beta values
		together.

		Args:
			x_limits: limits of the x-axis. Default is False.
			y_limits: limits of the y-axis. Default is False.
			plot_with_formula: bool, default is false, is True will look for 
				formula for the y-value to plot in title.
			figure_folder: optional, default is None. If default, will place
				figures in figures/{batch_name}/post_analysis/{observable_name}
			plot_vline_at: optional, float. If present, will plot a vline at 
				position given position.
		"""

		if self.verbose:
			print "Plotting %s for betas %s together" % (
				self.observable_name_compact,
				", ".join([str(b) for b in self.beta_values]))

		fig = plt.figure(dpi=self.dpi)
		ax = fig.add_subplot(111)

		self._check_plot_values()

		# Retrieves values to plot
		for beta in sorted(self.plot_values):
			value = self.plot_values[beta]
			x = value["x"]
			y = value["y"]
			y_err = value["y_err"]
			if error_shape == "band":
				ax.plot(x, y, "-", label=value["label"], color=self.colors[beta])
				ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, 
					edgecolor="", facecolor=self.colors[beta])
			elif error_shape == "bars":
				ax.errorbar(x, y, yerr=y_err, capsize=5, fmt="_", ls=":", 
					label=value["label"], color=self.colors[beta], 
					ecolor=self.colors[beta])
			else:
				raise KeyError("%s not a recognized plot type" % error_shape)

		# Sets the title string
		title_string = r"%s" % self.observable_name
		if plot_with_formula:
			title_string += r" %s" % self.formula

		# Basic plotting commands
		ax.grid(True)
		ax.set_title(r"%s" % title_string)
		ax.set_xlabel(r"%s" % self.x_label)
		ax.set_ylabel(r"%s" % self.y_label)
		ax.legend(loc="lower right", prop={"size": 8})

		# if self.observable_name_compact == "energy":
		# 	ax.ticklabel_format(style="sci", axis="y", scilimits=(1,10))

		# Sets axes limits if provided
		if x_limits != False:
			ax.set_xlim(x_limits)
		if y_limits != False:
			ax.set_ylim(y_limits)

		# Plots a vertical line at position "plot_vline_at"
		if not isinstance(plot_vline_at, types.NoneType):
			ax.axvline(plot_vline_at, linestyle="--", color="0", alpha=0.3)

		# Saves and closes figure
		fname = self._get_plot_figure_name(output_folder=figure_folder)
		plt.savefig(fname)
		if self.verbose:
			print "Figure saved in %s" % fname

		# if self.observable_name_compact == "energy":
		# 	plt.show()

		plt.close(fig)

	def _get_plot_figure_name(self, output_folder=None):
		"""Retrieves appropriate figure file name."""
		if isinstance(output_folder, types.NoneType):
			output_folder = self.output_folder_path
		fname = "post_analysis_%s_%s.png" % (self.observable_name_compact,
			self.analysis_data_type)
		return os.path.join(output_folder, fname)

	def __str__(self):
		"""Class string representation method."""
		msg = "\n" +"="*100
		msg += "\nPost analaysis for:        " + self.observable_name_compact
		msg += "\n" + self.__doc__
		msg += "\nAnalysis-type:             " + self.analysis_data_type
		msg += "\nIncluding autocorrelation: " + self.ac
		msg += "\nOutput folder:             " + self.output_folder_path
		msg += "\n" + "="*100
		return msg

def main():
	exit("Exit: PostCore is not intended to be used as a standalone module.")

if __name__ == "__main__":
	main()