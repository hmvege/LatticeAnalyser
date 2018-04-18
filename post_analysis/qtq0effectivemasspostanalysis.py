from core.multiplotcore import MultiPlotCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

class QtQ0EffectiveMassPostAnalysis(MultiPlotCore):
	"""Post-analysis of the effective mass."""
	observable_name = r"Effective mass $am_{eff} = \log \frac{C(t_e)}{C(t_e+1)}$"
	observable_name_compact = "qtq0eff"
	x_label = r"$t_e[fm]$"
	y_label = r"$am_{eff}$"
	sub_obs = True
	hbarc = 0.19732697 #eV micro m

	def __init__(self, *args, **kwargs):
		# Ensures we load correct data
		self.observable_name_compact_old = self.observable_name_compact
		self.observable_name_compact = "qtq0eff"

		super(QtQ0EffectiveMassPostAnalysis, self).__init__(*args, **kwargs)
		
		# Resets the observable name after data has been loaded.
		self.observable_name_compact = self.observable_name_compact_old

	def _convert_label(self, lab):
		return int(lab[-4:])

	def effMass(self, Q, axis=0):
		"""Correlator for qtq0."""
		return np.log(Q/np.roll(Q, -1, axis=axis))

	def effMass_err(self, Q, dQ, axis=0):
		"""Correlator for qtq0 with error propagation."""
		q = np.roll(Q, -1, axis=axis)
		dq = np.roll(dQ, -1, axis=axis)
		return np.sqrt((dQ/Q)**2 + (dq/q)**2 - 2*dq*dQ/(q*Q))

	def analyse_raw(self, data_raw):
		_y_temp = self.effMass(data_raw, axis=0)
		y = np.mean(_y_temp, axis=1)
		y_err = np.std(_y_temp, axis=1)

		# C = np.mean(data_raw, axis=1)
		# C_err = np.std(data_raw, axis=1)
		# y = self.effMass(C, axis=0)
		# y_err = self.effMass_err(C, C_err, axis=0)

		return y, y_err

	def _initiate_plot_values(self, data, data_raw, flow_index=None):
		"""interval_index: int, should be in euclidean time."""

		# Sorts data into a format specific for the plotting method
		for beta in sorted(data.keys()):
			values = {}

			if flow_index == None:
				# Case where we have sub sections of observables, e.g. in euclidean time
				for sub_obs in self.observable_intervals[beta]:
					sub_values = {}
					sub_values["a"] = get_lattice_spacing(beta)
					sub_values["x"] = np.linspace(0, 
						self.lattice_sizes[beta][1] * sub_values["a"], 
						self.lattice_sizes[beta][1])
					sub_values["y"], sub_values["y_err"] = self.analyse_raw(data_raw[beta][self.observable_name_compact][sub_obs])
					sub_values["label"] = r"%s, $\beta=%2.2f$, $t_f=%d$" % (
						self.size_labels[beta], beta, self._convert_label(sub_obs))
					sub_values["color"] = self.colors[beta]
					values[sub_obs] = sub_values
				self.plot_values[beta] = values

			else:
				tf_index = "tflow%04d" % flow_index
				values["a"] = get_lattice_spacing(beta)
				
				# For exact box sizes
				values["x"] = np.linspace(0,
					self.lattice_sizes[beta][1] * values["a"],
					self.lattice_sizes[beta][1])

				# values["y"] = self.C(data[beta][tf_index]["y"])
				# values["y_err"] = self.C_std(data[beta][tf_index]["y"], data[beta][tf_index]["y_error"])
				values["y"], values["y_err"] = self.analyse_raw(data_raw[beta][self.observable_name_compact][tf_index])
				values["label"] = r"%s $\beta=%2.2f$, $t_f=%d$" % (
					self.size_labels[beta], beta, flow_index)
				values["color"] = self.colors[beta]
				self.plot_values[beta] = values

	def plot_interval(self, flow_index, **kwargs):
		"""
		Sets and plots only one interval.

		Args:
			flow_index: flow time integer
			euclidean_index: integer for euclidean time
		"""
		self.plot_values = {}
		self.interval_index = flow_index
		data, data_raw = self._get_analysis_data(self.analysis_data_type)
		self._initiate_plot_values(data, data_raw, flow_index=flow_index)

		# Sets the x-label to proper units
		x_label_old = self.x_label
		self.x_label = r"$t_f[fm]$"

		# SET THIS TO ZERO IF NO Y-AXIS SCALING IS TO BE DONE
		# kwargs["y_limits"] = [-0.1,1]
		kwargs["error_shape"] = "bars"

		# Makes it a global constant so it can be added in plot figure name
		self.plot(**kwargs)

		self.x_label = x_label_old

	def plot(self, *args, **kwargs):
		"""Ensuring I am plotting with formule in title."""
		kwargs["plot_with_formula"] = True
		kwargs["y_limits"] = [-2,2]
		super(QtQ0EffectiveMassPostAnalysis, self).plot(*args, **kwargs)

	def plot_series(self, indexes, beta="all", x_limits=False, 
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
		data, data_raw = self._get_analysis_data(self.analysis_data_type)
		self._initiate_plot_values(data, data_raw)

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

		# print axes
		for ax, i in zip(list(itertools.chain(*axes)), indexes):
			for ibeta in bvalues:
				# Retrieves the values deepending on the indexes provided and beta values
				value = self.plot_values[ibeta][sorted(self.observable_intervals[ibeta])[i]]
				x = value["x"]
				y = value["y"]
				y_err = value["y_err"]
				# ax.plot(x, y, "o", label=value["label"], color=value["color"])
				# ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, edgecolor='',
				# 	facecolor=value["color"])
				ax.errorbar(x, y, yerr=y_err, fmt=".", color=value["color"], ecolor=value["color"],
					label=value["label"])
				
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

def main():
	exit("Exit: QtQ0EffectiveMassPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()