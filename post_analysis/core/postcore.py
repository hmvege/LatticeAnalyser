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
	"""Post analysis base class."""
	observable_name = "Observable"
	observable_name_compact = "obs"
	formula = ""
	x_label = r""
	y_label = r""
	dpi = 350
	r0 = 0.5
	sub_obs = False
	sub_sub_obs = False
	interval = []

	def __init__(self, data, with_autocorr=True, figures_folder="../figures",
		verbose=False, dryrun=False):
		"""
		Base class for analysing beta values together after initial analysis.

		Args:
			data: PostAnalysisDataReader object, contains all of the 
				observable data.
			with_autocorr: bool, optional. Will perform analysis on data
				corrected by autocorrelation sqrt(2*tau_int). Default is True.
			figures_folder: str, optional. Default output folder is ../figures.
			verbose: bool, optional. A more verbose output. Default is False.
			dryrun: bool, optional. No major changes will be performed. 
				Default is False.
		"""

		if with_autocorr:
			self.ac = "with_autocorr"
		else:
			self.ac = "without_autocorr"
		
		self.with_autocorr = with_autocorr
		self.reference_values = data.reference_values
		observable = self.observable_name_compact

		self.verbose = verbose
		self.dryrun = dryrun

		self.beta_values = sorted(data.beta_values)
		self.colors = data.colors
		self.lattice_sizes = data.lattice_sizes
		self.size_labels = data.labels
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

	# def get_observable_at(self, tf):
	# 	return self.


	def set_analysis_data_type(self, analysis_data_type="bootstrap"):
		"""Sets the analysis type and retrieves correct analysis data."""

		# Makes it a global constant so it can be added in plot figure name
		self.analysis_data_type = analysis_data_type

		self.plot_values = {} # Clears old plot values
		self._initiate_plot_values(self.data[analysis_data_type],
			self.data_raw[analysis_data_type])

	def print_estimates(self, flow_times=[]):
		"""
		Prints the topsus values given the flow time.
		"""
		for bval, tf0 in zip(self.beta_values, flow_times):
			print self.beta_values[bval]["y"][tf0]

	def _initiate_plot_values(self, data, data_raw):
		"""Sorts data into a format specific for the plotting method."""
		for beta in sorted(data.keys()):
			values = {}
			values["a"], values["a_err"] = get_lattice_spacing(beta)
			values["x"] = values["a"]* np.sqrt(8*data[beta]["x"])
			values["y"] = data[beta]["y"]
			values["y_err"] = data[beta]["y_error"]
			values["y_raw"] = data_raw[beta][self.observable_name_compact]
			values["y_uraw"] = self.data_raw["unanalyzed"][beta]\
				[self.observable_name_compact]
			if self.with_autocorr:
				values["tau_int"] = data[beta]["ac"]["tau_int"]
				values["tau_int_err"] = data[beta]["ac"]["tau_int_err"]
			else:
				values["tau_int"] = None
				values["tau_int_err"] = None
			values["label"] = r"%s $\beta=%2.2f$" % (
				self.size_labels[beta], beta)
			values["color"] = self.colors[beta]

			self.plot_values[beta] = values

	def plot(self, x_limits=False, y_limits=False, plot_with_formula=False,
		error_shape="band", figure_folder=None, plot_vline_at=None,
		plot_hline_at=None, figure_name_appendix=""):
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
			plot_hline_at: optional, float. If present, will plot a hline at 
				position given position.
			figure_name_appendix: optional, str, adds provided string to 
				filename. Default is adding nothing.
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

		# # Sets the title string
		# title_string = r"%s" % self.observable_name
		# if plot_with_formula:
		# 	title_string += r" %s" % self.formula

		# Basic plotting commands
		ax.grid(True)
		# ax.set_title(r"%s" % title_string)
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

		# Plots a horizontal line at position "plot_hline_at"
		if not isinstance(plot_hline_at, types.NoneType):
			ax.axhline(plot_hline_at, linestyle="--", color="0", alpha=0.3)

		# Saves and closes figure
		fname = self._get_plot_figure_name(output_folder=figure_folder, 
			figure_name_appendix=figure_name_appendix)
		plt.savefig(fname)
		if self.verbose:
			print "Figure saved in %s" % fname

		plt.close(fig)

	def _get_plot_figure_name(self, output_folder=None, figure_name_appendix=""):
		"""Retrieves appropriate figure file name."""
		if isinstance(output_folder, types.NoneType):
			output_folder = self.output_folder_path
		fname = "post_analysis_%s_%s%s.png" % (self.observable_name_compact,
			self.analysis_data_type, figure_name_appendix)
		return os.path.join(output_folder, fname)

	def get_values(self, tf, atype=None, extrap_method=None):
		"""
		Method for retrieving values a given flow time t_f.

		Args:
			tf: float or str. (float) flow time at a given t_f/a^2. If 
				string is "t0", will return the flow time at reference value
				t0/a^2 from t^2<E>=0.3 for a given beta.
			atype: str, type of analysis we have performed.
			extrap_method: str, type of extrapolation technique used.

		Returns:
			{beta: {t0, y0, y0_error}
		"""

		if isinstance(atype, types.NoneType):
			atype = self.analysis_data_type
		if isinstance(extrap_method, types.NoneType):
			if hasattr(self, "extrapolation_method"):
				extrap_method = self.extrapolation_method
			else:
				raise KeyError("Missing extrapolation method.")

		values = {}

		print self.reference_values.keys()
		print self.reference_values[atype].keys()

		if tf == "t0":
			for b in self.beta_values:
				tf0 = self.plot_values["beta"]["x"]
				values[b] = {}
			# return self.y
			np.argmin()
			pass
		else:
			assert isinstance(tf, float), "tf not of type float"


		exit("woop")
		raise NotImplementedError("Not implemented method for retrieving values yet.")

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