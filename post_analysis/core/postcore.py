from tools.postanalysisdatareader import PostAnalysisDataReader
from tools.latticefunctions import get_lattice_spacing
from tools.folderreadingtools import check_folder
import matplotlib.pyplot as plt
import numpy as np
import os

class PostCore(object):
	"""Post analysis base class."""
	observable_name = "Observable"
	observable_name_compact = "obs"
	formula = ""
	x_label = r""
	y_label = r""
	dpi=350
	size_labels = {
		6.0  : r"$24^3 \times 48$",
		6.1  : r"$28^3 \times 56$",
		6.2  : r"$32^3 \times 64$",
		6.45 : r"$48^3 \times 96$",
	}
	r0 = 0.5
	sub_obs = False
	observable_intervals = {}
	# blue, green, red purple
	beta_colors = ["#5cbde0", "#6fb718", "#bc232e", "#8519b7"]

	def __init__(self, data, with_autocorr=True, figures_folder="../figures", verbose=False):
		if with_autocorr:
			self.ac = "with_autocorr"
		else:
			self.ac = "without_autocorr"
		self.with_autocorr = with_autocorr
		observable = self.observable_name_compact

		self.verbose = verbose

		# Retrieves relevant data values and sorts them by beta values
		self.flow_time = data.flow_time
		self.unanalyzed_data = {}
		self.bootstrap_data	= {}
		self.jackknife_data = {}

		# Checks that the observable is among the available data
		assert_msg = ("%s is not among current data(%s). Have the pre analysis"
			" been performed?" % (observable, ", ".join(data.observable_list)))
		assert observable in data.observable_list, assert_msg

		for beta in sorted(data.data_observables[observable].keys()):
			if self.sub_obs:
				self.observable_intervals[beta] = data.data_observables[observable][beta].keys()
				if not beta in self.unanalyzed_data:
					self.unanalyzed_data[beta] = {}
					self.bootstrap_data[beta] = {}
					self.jackknife_data[beta] = {}
				for subobs in data.data_observables[observable][beta]:
					self.unanalyzed_data[beta][subobs] = data.data_observables[observable][beta][subobs][self.ac]["unanalyzed"]
					self.bootstrap_data[beta][subobs] = data.data_observables[observable][beta][subobs][self.ac]["bootstrap"]
					self.jackknife_data[beta][subobs] = data.data_observables[observable][beta][subobs][self.ac]["jackknife"]
			else:
				self.unanalyzed_data[beta] = data.data_observables[observable][beta][self.ac]["unanalyzed"]
				self.bootstrap_data[beta] = data.data_observables[observable][beta][self.ac]["bootstrap"]
				self.jackknife_data[beta] = data.data_observables[observable][beta][self.ac]["jackknife"]

		self.bs_raw = data.raw_analysis["bootstrap"]
		self.jk_raw = data.raw_analysis["jackknife"]
		self.ac_corrections	= data.raw_analysis["autocorrelation"]

		if not isinstance(self.bs_raw[self.bs_raw.keys()[0]][self.observable_name_compact], np.ndarray):
			self.NBoots = self.bs_raw.values()[0][self.observable_name_compact].values()[0].shape[-1]
		else:
			self.NBoots = self.bs_raw[self.bs_raw.keys()[0]][self.observable_name_compact].shape[-1]

		# Small test to ensure that the number of bootstraps and number of different beta batches match
		err_msg = "Number of bootstraps do not match number of different beta values"
		if self.sub_obs:
			chk_bs_len = lambda _a, _i: _a[_i][self.observable_name_compact].values()[0].shape[-1] == self.NBoots
		else:
			chk_bs_len = lambda _a, _i: _a[_i][self.observable_name_compact].shape[-1] == self.NBoots

		assert sum([True for i in self.bs_raw.keys() if chk_bs_len(self.bs_raw, i)]) == data.N_betas, err_msg

		# Creates base output folder for post analysis figures
		self.figures_folder = figures_folder
		check_folder(self.figures_folder, dryrun=False, verbose=self.verbose)
		check_folder(os.path.join(self.figures_folder, data.batch_name), dryrun=False, verbose=self.verbose)

		# Creates output folder
		self.post_anlaysis_folder = os.path.join(self.figures_folder, data.batch_name, "post_analysis")
		check_folder(self.post_anlaysis_folder, dryrun=False, verbose=self.verbose)

		# Creates observable output folder
		self.output_folder_path = os.path.join(self.post_anlaysis_folder, self.observable_name_compact)
		check_folder(self.output_folder_path, dryrun=False, verbose=self.verbose)

		# Creates colors to use
		self.colors = {}
		for color, beta in zip(self.beta_colors, sorted(data.data_observables[observable].keys())):
			self.colors[beta] = color

	def _check_plot_values(self):
		"""Checks if we have set the analysis data type yet."""
		if not hasattr(self,"plot_values"):
			raise AttributeError("set_analysis_data_type() has not been set yet.")

	def _get_analysis_data(self, analysis_data_type):
		"""Retrieving data depending on analysis type we are choosing"""
		if analysis_data_type == "bootstrap":
			return self.bootstrap_data
		elif analysis_data_type == "jackknife":
			return self.jackknife_data
		elif analysis_data_type == "unanalyzed":
			return self.unanalyzed_data
		else:
			raise KeyError("Analysis %s not recognized" % analysis_data_type)

	def set_analysis_data_type(self, analysis_data_type="bootstrap"):
		self.plot_values = {}

		data = self._get_analysis_data(analysis_data_type)

		# Makes it a global constant so it can be added in plot figure name
		self.analysis_data_type = analysis_data_type

		# Initiates plot values
		self._initiate_plot_values(data)

	def _initiate_plot_values(self, data):
		"""Sorts data into a format specific for the plotting method."""
		for beta in sorted(data.keys()):
			if beta == 6.45: self.flow_time *= 2
			values = {}
			values["a"] = get_lattice_spacing(beta)
			values["x"] = values["a"]* np.sqrt(8*self.flow_time)
			values["y"] = data[beta]["y"]
			values["bs"] = self.bs_raw[beta][self.observable_name_compact]
			values["y_err"] = data[beta]["y_error"] # negative since the minus sign will go away during linear error propagation
			values["label"] = r"%s $\beta=%2.2f$" % (self.size_labels[beta], beta)
			values["color"] = self.colors[beta]
			self.plot_values[beta] = values

	def plot(self, x_limits=False, y_limits=False, plot_with_formula=False):
		"""
		Function for making a basic plot of all the different beta values together.

		Args:
			x_limits: limits of the x-axis. Default is False.
			y_limits: limits of the y-axis. Default is False.
			plot_with_formula: bool, default is false, is True will look for 
				formula for the y-value to plot in title.
		"""
		if self.verbose:
			print "Plotting %s for betas %s together" % (
				self.observable_name_compact,
				", ".join([str(b) for b in sorted(self.unanalyzed_data.keys())]))

		fig = plt.figure(dpi=self.dpi)
		ax = fig.add_subplot(111)

		self._check_plot_values()

		# Retrieves values to plot
		for beta in sorted(self.plot_values):
			value = self.plot_values[beta]
			x = value["x"]
			y = value["y"]
			y_err = value["y_err"]
			ax.plot(x, y, "-", label=value["label"], color=value["color"])
			ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, edgecolor='', facecolor=value["color"])

		# print self.flow_time[1:]**2*self._energy_continuum(self.flow_time[1:])[0]
		# ax.plot(self.flow_time[1:]/self.r0**2,self.flow_time[1:]**2*self._energy_continuum(self.flow_time[1:])[0],color="b")

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

		plt.tight_layout()

		# Saves and closes figure
		fname = self._get_plot_figure_name()
		plt.savefig(fname)
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
				bfit["t0"], bfit["t0_err"] = fit_line_from_bootstrap(	values["x"], values["bs"], self.observable_name_compact,
																		beta, fit_target, fit_interval, axis=axis,
																		fit_function_modifier=fit_function_modifier,
																		plot_fit_window=plot_fit_window)
			elif fit_type == "data_line_fit":
				bfit["t0"], bfit["t0_err"] = fit_line(	values["x"], values["y"], values["y_err"],
														self.observable_name_compact, beta,
														fit_target, fit_interval, axis=axis,
														fit_function_modifier=fit_function_modifier,
														plot_fit_window=plot_fit_window)
			elif fit_type == "nearest_val_fit":
				raise NotImplementedError("'nearest_val_fit' not implemented as a fit type yet.")
			else:
				raise KeyError("No fit_type named %s. Options: 'bootstrap_fit', 'data_line_fit' or 'nearest_val_fit'" % fit_type)

			# Adds lattice spacing to fit
			bfit["a"] = getLatticeSpacing(bfit["beta"])

			# Adds to list of batch-values
			self.beta_fit.append(bfit)

	def _get_plot_figure_name(self):
		"""Retrieves appropriate figure file name."""
		output_folder = self.output_folder_path
		fname = "post_analysis_%s_%s.png" % (self.observable_name_compact, self.analysis_data_type)
		return os.path.join(output_folder, fname)

	def __str__(self):
		msg = "\n" +"="*100
		msg += "\nPost analaysis for:        " + self.observable_name_compact
		msg += "\nIncluding autocorrelation: " + self.ac
		msg += "\nOutput folder:             " + self.output_folder_path
		msg += "\n" + "="*100
		return msg

def main():
	exit("Exit: PostCore is not intended to be used as a standalone module.")

if __name__ == '__main__':
	main()