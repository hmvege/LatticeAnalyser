from post_analysis.core.multiplotcore import MultiPlotCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import types

# Temporary needed for assessing the autocorrelation in the eff mass data.
from statistics.autocorrelation import Autocorrelation
import multiprocessing
import statistics.parallel_tools as ptools

class QtQ0EffectiveMassMCIntervalsPostAnalysis(MultiPlotCore):
	"""Post-analysis of the effective mass."""
	observable_name = r"Effective mass, "
	observable_name += r"$am_\textrm{eff} = \log \frac{C(t_e)}{C(t_e+1)}$, "
	observable_name += r"$C(t_e)=\langle q_t q_0\rangle$"
	observable_name_compact = "qtq0effmc"
	x_label = r"$t_e[\textrm{fm}]$"
	y_label = r"$r_0 m_\textrm{eff}$"
	sub_obs = True
	sub_sub_obs = True
	hbarc = 0.19732697 #eV micro m
	dpi=400
	fold = True
	fold_range = 20
	subfolder_type = "tflow"

	def __init__(self, *args, **kwargs):
		# Ensures we load correct data
		self.observable_name_compact_old = self.observable_name_compact

		super(QtQ0EffectiveMassMCIntervalsPostAnalysis, self).__init__(*args, **kwargs)
		
		# Resets the observable name after data has been loaded.
		self.observable_name_compact = self.observable_name_compact_old

	def fold_array(self, arr, axis=0):
		"""Method for folding an array by its last values."""
		# OLD FOLD METHOD - DO NOT INCREASE STATISTICS BY THIS WAY:|
		# folded_array = np.roll(arr, self.fold_range, axis=axis)
		# folded_array = folded_array[:self.fold_range*2]
		# folded_array[:self.fold_range] *= -1

		# fold_range = int(arr.shape[-1]/2) - 1
		fold_range = arr.shape[-1]/2
		folded_array = arr[:fold_range+1]
		last_part = arr[fold_range+1:] * (-1)
		folded_array[1:-1] = (folded_array[1:-1] + np.flip(last_part, axis=0))*0.5
		# folded_array[1:-1] *= 0.5 
		return folded_array

	def fold_error_array(self, arr, axis=0):
		"""Method for folding an array by its last values."""
		# OLD FOLD METHOD - DO NOT INCREASE STATISTICS BY THIS WAY:|
		# folded_array = np.roll(arr, self.fold_range, axis=axis)
		# folded_array = folded_array[:self.fold_range*2]
		# folded_array[:self.fold_range] *= -1

		fold_range = arr.shape[-1]/2
		folded_array = arr[:fold_range+1]
		last_part = arr[fold_range+1:] * (-1)
		folded_array[1:-1] = (folded_array[1:-1] + np.flip(last_part, axis=0))*0.5
		folded_array[1:-1] = np.sqrt((0.5*folded_array[1:-1])**2 + (0.5*np.flip(last_part, axis=0))**2)
		return folded_array

	def _convert_label(self, lab):
		return float(lab[-6:])

	def effMass(self, Q, axis=0):
		"""Correlator for qtq0."""
		return np.log(Q/np.roll(Q, -1, axis=axis)) # C(t)/C(t+1)

	def effMass_err(self, Q, dQ, axis=0):
		"""Correlator for qtq0 with error propagation."""
		q = np.roll(Q, -1, axis=axis)
		dq = np.roll(dQ, -1, axis=axis)
		# return np.sqrt((dQ/Q)**2 + (dq/q)**2 - 2*dq*dQ/(q*Q))
		return np.sqrt((dQ/Q)**2 + (dq/q)**2 - 2*dq*dQ/(q*Q))

	def analyse_raw(self, data, data_raw):
		"""
		Method for analysis <QteQ0>_i where i is index of bootstrapped,
		jackknifed or unanalyzed samples.
		"""

		# Using bs samples
		y = self.effMass(data["y"])
		y_err = self.effMass_err(data["y"], data["y_error"])

		return y, y_err

	def analyse_data(self, data):
		"""Method for analysis <QteQ0>."""
		return self.effMass(data["y"]), self.effMass_err(data["y"], \
			data["y_error"])

	def _get_plot_figure_name(self, output_folder=None, 
		figure_name_appendix=""):
		"""Retrieves appropriate figure file name."""
		if isinstance(output_folder, types.NoneType):
			output_folder = os.path.join(self.output_folder_path, "slices")
		check_folder(output_folder, False, True)
		fname = "post_analysis_%s_%s_tf%f_mc%s%s.png" % (
			self.observable_name_compact,self.analysis_data_type, 
			self.interval_index, self.mc_int_str, figure_name_appendix)
		return os.path.join(output_folder, fname)

	def _initiate_plot_values(self, data, data_raw, tf0, mc_int=None):
		"""mc_int: int, should be in mc time."""

		tf0_key = "tflow%.4f" % tf0

		# Sorts data into a format specific for the plotting method
		for beta in self.beta_values:
			values = {}

			if isinstance(mc_int, types.NoneType):
				# Case where we have sub sections of observables, e.g. in 
				# mc time.
				for sub_obs in self.observable_intervals[beta]:
					sub_values = {}
					sub_values["a"], sub_values["a_err"] = get_lattice_spacing(beta)
					sub_values["x"] = np.linspace(0, 
						self.lattice_sizes[beta][1] * sub_values["a"], 
						self.lattice_sizes[beta][1])
					sub_values["y"], sub_values["y_err"] = self.analyse_raw(
						data[beta][sub_obs][tf0_key],
						data_raw[beta][self.observable_name_compact][sub_obs][tf0_key])

					sub_values["label"] = r"%s, $\beta=%2.2f$, $t_f=%.2f$, $MC:%s$" % (
						self.size_labels[beta], beta, tf0, sub_obs)

					sub_values["raw"] = data_raw[beta] \
						[self.observable_name_compact][sub_obs][tf0_key]

					if self.fold:
						# OLD FOLD METHOD - DO NOT INCREASE STATISTICS BY THIS WAY:|
						# sub_values["x"] = np.linspace(-self.fold_range*sub_values["a"], 
						# 	(self.fold_range-1)*sub_values["a"], self.fold_range*2)
						sub_values["x"] = np.linspace(0, 
							(int(sub_values["y"].shape[0]/2))*sub_values["a"],
							int(sub_values["y"].shape[0]/2)+1)
						sub_values["y"] = self.fold_array(sub_values["y"]) * self.r0 / sub_values["a"]
						sub_values["y_err"] = \
							self.fold_error_array(sub_values["y_err"])  * self.r0 / sub_values["a"]
						# sub_values["raw"] = self.fold_array(sub_values["raw"],
						# 	axis=0)
						self.fold_position = sub_values["x"][self.fold_range]

					if self.with_autocorr:
						sub_values["tau_int"] = \
							data[beta][sub_obs][tf0_key]["ac"]["tau_int"]
						sub_values["tau_int_err"] = \
							data[beta][sub_obs][tf0_key]["ac"]["tau_int_err"]

					values[sub_obs] = sub_values
				self.plot_values[beta] = values

			else:
				mc_dict = {b: mc_int[ib] for ib, b in enumerate(self.beta_values)}
				# raise NotImplementedError("This section is not complete")
				values["a"], values["a_err"] = get_lattice_spacing(beta)
				
				# For exact box sizes
				values["x"] = np.linspace(0,
					self.lattice_sizes[beta][1] * values["a"],
					self.lattice_sizes[beta][1])

				values["y_raw"] = data_raw[beta] \
					[self.observable_name_compact][mc_dict[beta]][tf0_key]

				if self.with_autocorr:
					values["tau_int"] = data[beta][mc_dict[beta]][tf0_key]["ac"]["tau_int"]
					values["tau_int_err"] = \
						data[beta][mc_dict[beta]][tf0_key]["ac"]["tau_int_err"]

				values["y"], values["y_err"] = \
					self.analyse_data(data[beta][mc_dict[beta]][tf0_key])

				if self.fold:
					# # OLD METHOD
					# values["x"] = np.linspace(-self.fold_range*values["a"], 
					# 		(self.fold_range-1)*values["a"], self.fold_range*2)
					values["x"] = np.linspace(0, 
							(int(values["y"].shape[0]/2))*values["a"],
							int(values["y"].shape[0]/2)+1)

					values["y"] = self.fold_array(values["y"]) * self.r0 / values["a"]
					values["y_err"] = \
						self.fold_error_array(values["y_err"]) * self.r0 / values["a"]
					# values["y_raw"] = self.fold_array(values["y_raw"], axis=0)
					self.fold_position = values["x"][self.fold_range]

				values["label"] = r"%s $\beta=%2.2f$, $t_f=%.2f$, $MC=%s$" % (
					self.size_labels[beta], beta, tf0,
					", ".join(["[%s)" % i for i in mc_int]))

				self.plot_values[beta] = values

	def set_analysis_data_type(self, tf0, analysis_data_type="bootstrap"):
		"""Sets the analysis type and retrieves correct analysis data."""

		# Makes it a global constant so it can be added in plot figure name
		self.analysis_data_type = analysis_data_type

		self.plot_values = {} # Clears old plot values
		self._initiate_plot_values(self.data[analysis_data_type],
			self.data_raw[analysis_data_type], tf0)

	def plot_interval(self, tf0, mc_int, **kwargs):
		"""
		Sets and plots only one interval.

		Args:
			flow_index: flow time integer
			euclidean_index: integer for euclidean time
		"""
		self.plot_values = {}
		self.interval_index = tf0
		self.mc_int_str = "_".join(mc_int)
		self._initiate_plot_values(self.data[self.analysis_data_type], 
			self.data_raw[self.analysis_data_type], tf0, 
			mc_int=mc_int)

		# Sets the x-label to proper units
		x_label_old = self.x_label
		self.x_label = r"$t_e[fm]$"

		# SET THIS TO ZERO IF NO Y-AXIS SCALING IS TO BE DONE
		# kwargs["y_limits"] = [-1,1]
		kwargs["error_shape"] = "bars"
		# Makes it a global constant so it can be added in plot figure name
		self.plot(**kwargs)

		self.x_label = x_label_old


	def plot_series(self, indexes, tf0, beta="all", x_limits=False, 
		y_limits=False, plot_with_formula=False, error_shape="band"):
		"""
		Method for plotting 4 axes together.
		"""
		self.plot_values = {}
		self._initiate_plot_values(self.data[self.analysis_data_type],
			self.data_raw[self.analysis_data_type], tf0)

		self._series_plot_core(indexes, beta=beta, x_limits=[-0.1, 1], 
		y_limits=[-10, 10], plot_with_formula=plot_with_formula, 
		error_shape=error_shape)


	def plot(self, *args, **kwargs):
		"""Ensuring I am plotting with formule in title."""
		kwargs["plot_with_formula"] = True
		# kwargs["y_limits"] = [-2,2]
		kwargs["y_limits"] = [-10,10]
		# if self.fold:
		# 	kwargs["plot_vline_at"] = self.fold_position
		# else:
		# 	kwargs["x_limits"] = [-0.1,4.7]
		kwargs["x_limits"] = [-0.1,1]
		super(QtQ0EffectiveMassMCIntervalsPostAnalysis, self).plot(*args, **kwargs)

def main():
	exit(("Exit: QtQ0EffectiveMassMCIntervalsPostAnalysis not intended to be a "
		"standalone module."))

if __name__ == '__main__':
	main()