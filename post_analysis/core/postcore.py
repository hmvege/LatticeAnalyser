from tools.postanalysisdatareader import PostAnalysisDataReader
from tools.latticefunctions import get_lattice_spacing
from tools.folderreadingtools import check_folder, get_NBoots
import matplotlib.pyplot as plt
import numpy as np
import os

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

		# Retrieves relevant data values and sorts them by beta values
		self.flow_time = data.flow_time

		self.data = {}

		self.beta_values = sorted(data.beta_values)

		self.analysis_types = data.analysis_types
		if "autocorrelation" in self.analysis_types:
			self.analysis_types.remove("autocorrelation")
		for atype in self.analysis_types:
			self.data[atype] = {beta: {} for beta in self.beta_values}
		
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
					else:
						# Fills up observable intervals
						self.observable_intervals[beta] = \
							data.data_observables[observable][beta].keys()

						for subobs in data.data_observables[observable][beta]:
							self.data[atype][beta][subobs] = \
								data.data_observables[observable][beta] \
								[subobs][self.ac][atype]
				else:
					self.data[atype][beta] = data.data_observables \
						[observable][beta][self.ac][atype]

		self.data_raw = {}
		for key in data.raw_analysis:
			if key == "autocorrelation":
				self.ac_raw = data.raw_analysis[key]
			else:
				self.data_raw[key] = data.raw_analysis[key]

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

	def _check_plot_values(self):
		"""Checks if we have set the analysis data type yet."""
		if not hasattr(self, "plot_values"):
			raise AttributeError("set_analysis_data_type() has not been set yet.")

	def set_analysis_data_type(self, analysis_data_type="bootstrap"):
		"""Sets the analysis type and retrieves correct analysis data."""
		self.plot_values = {} # Clears old plot values
		self._initiate_plot_values(self.data[analysis_data_type], 
			self.data_raw[analysis_data_type])
		# Makes it a global constant so it can be added in plot figure name
		self.analysis_data_type = analysis_data_type

	def _initiate_plot_values(self, data, data_raw):
		"""Sorts data into a format specific for the plotting method."""
		for beta in sorted(data.keys()):
			if beta == 6.45: self.flow_time *= 2
			values = {}
			values["a"] = get_lattice_spacing(beta)
			values["x"] = values["a"]* np.sqrt(8*self.flow_time)
			values["y"] = data[beta]["y"]
			# values["bs"] = data_raw[beta][self.observable_name_compact]
			values["y_err"] = data[beta]["y_error"]
			values["label"] = r"%s $\beta=%2.2f$" % (self.size_labels[beta], beta)
			values["color"] = self.colors[beta]
			self.plot_values[beta] = values

	def plot(self, x_limits=False, y_limits=False, plot_with_formula=False,
		error_shape="band", figure_folder=None):
		"""
		Function for making a basic plot of all the different beta values
		together.

		Args:
			x_limits: limits of the x-axis. Default is False.
			y_limits: limits of the y-axis. Default is False.
			plot_with_formula: bool, default is false, is True will look for 
				formula for the y-value to plot in title.
			figure_folder: optional, default is None. If default, will place
				figures in 
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
				ax.plot(x, y, "-", label=value["label"], color=value["color"])
				ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, 
					edgecolor='', facecolor=value["color"])
			elif error_shape == "bars":
				ax.errorbar(x, y, yerr=y_err, capsize=5, fmt="_", ls=":", 
					label=value["label"], color=value["color"], 
					ecolor=value["color"])
			else:
				raise KeyError("%s not a recognized plot type" % error_shape)

		# Sets the title string
		title_string = r"%s" % self.observable_name
		if plot_with_formula:
			title_string += r" %s" % self.formula

		# Basic plotting commands
		ax.grid(True)
		ax.set_title(title_string)
		ax.set_xlabel(self.x_label)
		ax.set_ylabel(self.y_label)
		ax.legend(loc="best", prop={"size": 8})

		# Sets axes limits if provided
		if x_limits != False:
			ax.set_xlim(x_limits)
		if y_limits != False:
			ax.set_ylim(y_limits)

		# plt.tight_layout()

		# Saves and closes figure
		fname = self._get_plot_figure_name()
		plt.savefig(fname)
		if self.verbose:
			print "Figure saved in %s" % fname
		# plt.show()
		plt.close(fig)

	def _get_beta_values_to_fit(self, fit_target, fit_interval, axis,
								fit_type="bootstrap_fit",
								fit_function_modifier=lambda x: x,
								plot_fit_window=False):
		"""
		Retrieves a line fitted value at a target t0.
		Available fit_types:
			- bootstrap_fit (default)
			- linefit_data
			- nearest
		"""
		self.beta_fit = []

		self._check_plot_values()

		# Populates values to be plotted and 
		for beta in self.plot_values:
			bfit = {}
			# Sets beta value for data
			bfit["beta"] = beta

			# Retrieves fit value as well as its error
			if fit_type == "bootstrap_fit":
				bfit["t0"], bfit["t0_err"] = fit_line_from_bootstrap(	
					values["x"], values["bs"], self.observable_name_compact,
					beta, fit_target, fit_interval, axis=axis,
					fit_function_modifier=fit_function_modifier,
					plot_fit_window=plot_fit_window)
			elif fit_type == "data_line_fit":
				bfit["t0"], bfit["t0_err"] = fit_line(	
					values["x"], values["y"], values["y_err"],
					self.observable_name_compact, beta,
					fit_target, fit_interval, axis=axis,
					fit_function_modifier=fit_function_modifier,
					plot_fit_window=plot_fit_window)
			elif fit_type == "nearest_val_fit":
				raise NotImplementedError(
					("'nearest_val_fit' not implemented "
					"as a fit type yet."))
			else:
				raise KeyError(
					("No fit_type named %s. Options: 'bootstrap_fit', "
					"'data_line_fit' or 'nearest_val_fit'" % fit_type))

			# Adds lattice spacing to fit
			bfit["a"] = getLatticeSpacing(bfit["beta"])

			# Adds to list of batch-values
			self.beta_fit.append(bfit)

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

if __name__ == '__main__':
	main()