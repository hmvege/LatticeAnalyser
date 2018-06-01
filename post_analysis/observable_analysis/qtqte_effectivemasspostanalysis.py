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

class QtQ0EffectiveMassPostAnalysis(MultiPlotCore):
	"""Post-analysis of the effective mass."""
	observable_name = r"Effective mass, "
	observable_name += r"$am_{eff} = \log \frac{C(t_e)}{C(t_e+1)}$, "
	observable_name += r"$C(t_e)=\langle q_t q_0\rangle$"
	observable_name_compact = "qtq0eff"
	x_label = r"$t_e[fm]$"
	y_label = r"$am_{eff}$"
	sub_obs = True
	hbarc = 0.19732697 #eV micro m
	dpi=400

	def __init__(self, *args, **kwargs):
		# Ensures we load correct data
		self.observable_name_compact_old = self.observable_name_compact
		self.observable_name_compact = "qtq0eff"

		super(QtQ0EffectiveMassPostAnalysis, self).__init__(*args, **kwargs)
		
		# Resets the observable name after data has been loaded.
		self.observable_name_compact = self.observable_name_compact_old

	def _convert_label(self, lab):
		return float(lab[-4:])

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
		# y = self.effMass(data["y"])
		# y_err = self.effMass_err(data["y"], data["y_error"])

		NEucl, NCfgs = data_raw.shape
		if self.analysis_data_type=="unanalyzed":
			N_BS = 500
			y_raw = np.zeros((NEucl, N_BS)) 	# Correlator, G
			index_lists = np.random.randint(NCfgs, size=(N_BS, NCfgs))
			# Performing the bootstrap samples
			for i in xrange(NEucl):
				for j in xrange(N_BS):
					y_raw[i,j] = np.mean(data_raw[i][index_lists[j]])
		else:
			y_raw = data_raw

		y_raw = np.log(y_raw/np.roll(y_raw, -1, axis=0))
		y = np.mean(y_raw, axis=1)
		y_err = np.std(y_raw, axis=1)

		# Runs parallel processes
		input_values = zip(	[data_raw[iEucl] for iEucl in range(NEucl)],
							[None for _d in range(NEucl)],
							[{} for _d in range(NEucl)])

		pool = multiprocessing.Pool(processes=8)				
		res = pool.map(ptools._autocorrelation_propagated_parallel_core, input_values)
		pool.close()

		error_correction = np.ones(NEucl)
		for i, _data in enumerate(data_raw):
			error_correction[i] = np.sqrt(2*res[i][2])

		# y = np.mean(_y_temp, axis=1)
		# y_err = np.std(_y_temp, axis=1) * error_correction
		y_err *= error_correction

		# print "\n"
		# print y[:10]
		# print y_err[:10],"\n"

		# for _res in results:
		# 	y_err *= np.sqrt(2*_res[2])


		# C = np.mean(data_raw, axis=1)
		# C_err = np.std(data_raw, axis=1)
		# y = self.effMass(C, axis=0)
		# y_err = self.effMass_err(C, C_err, axis=0)

		return y, y_err

	def analyse_data(self, data):
		"""Method for analysis <QteQ0>."""
		return self.effMass(data["y"]), self.effMass_err(data["y"], data["y_error"])

	def _get_plot_figure_name(self, output_folder=None):
		"""Retrieves appropriate figure file name."""
		if isinstance(output_folder, types.NoneType):
			output_folder = os.path.join(self.output_folder_path, "slices")
		check_folder(output_folder, False, True)
		fname = "post_analysis_%s_%s_q0_%f.png" % (self.observable_name_compact,
			self.analysis_data_type, self.interval_index)
		return os.path.join(output_folder, fname)

	def _initiate_plot_values(self, data, data_raw, flow_index=None):
		"""interval_index: int, should be in euclidean time."""

		# Sorts data into a format specific for the plotting method
		for beta in self.beta_values:
			values = {}

			if flow_index == None:
				# Case where we have sub sections of observables, e.g. in 
				# euclidean time.
				for sub_obs in self.observable_intervals[beta]:
					sub_values = {}
					sub_values["a"], sub_values["a_err"] = \
						get_lattice_spacing(beta)
					sub_values["x"] = np.linspace(0, 
						self.lattice_sizes[beta][1] * sub_values["a"], 
						self.lattice_sizes[beta][1])

					sub_values["y"], sub_values["y_err"] = self.analyse_raw(
						data[beta][sub_obs],
						data_raw[beta][self.observable_name_compact][sub_obs])

					sub_values["label"] = r"%s, $\beta=%2.2f$, $t_f=%.2f$" % (
						self.size_labels[beta], beta, 
						self._convert_label(sub_obs))

					sub_values["raw"] = data_raw[beta] \
						[self.observable_name_compact][sub_obs]

					if self.with_autocorr:
						sub_values["tau_int"] = \
							data[beta][sub_obs]["ac"]["tau_int"]
						sub_values["tau_int_err"] = \
							data[beta][sub_obs]["ac"]["tau_int_err"]

					values[sub_obs] = sub_values
				self.plot_values[beta] = values

			else:
				tf_index = "tflow%04.2f" % flow_index
				values["a"], values["a_err"] = get_lattice_spacing(beta)
				
				# For exact box sizes
				values["x"] = np.linspace(0,
					self.lattice_sizes[beta][1] * values["a"],
					self.lattice_sizes[beta][1])

				values["y_raw"] = data_raw[beta] \
					[self.observable_name_compact][tf_index]

				if self.with_autocorr:
					values["tau_int"] = data[beta][tf_index]["ac"]["tau_int"]
					values["tau_int_err"] = \
						data[beta][tf_index]["ac"]["tau_int_err"]

				values["y"], values["y_err"] = \
					self.analyse_data(data[beta][tf_index])

				values["label"] = r"%s $\beta=%2.2f$, $t_f=%.2f$" % (
					self.size_labels[beta], beta, flow_index)

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
		self._initiate_plot_values(self.data[self.analysis_data_type], 
			self.data_raw[self.analysis_data_type], flow_index=flow_index)

		# Sets the x-label to proper units
		x_label_old = self.x_label
		self.x_label = r"$t_e[fm]$"

		# SET THIS TO ZERO IF NO Y-AXIS SCALING IS TO BE DONE
		# kwargs["y_limits"] = [-1,1]
		kwargs["x_limits"] = [-0.1,1]
		kwargs["error_shape"] = "bars"

		# Makes it a global constant so it can be added in plot figure name
		self.plot(**kwargs)

		self.x_label = x_label_old

	def plot(self, *args, **kwargs):
		"""Ensuring I am plotting with formule in title."""
		kwargs["plot_with_formula"] = True
		# kwargs["y_limits"] = [-2,2]
		kwargs["y_limits"] = [-1,1]
		kwargs["x_limits"] = [-0.1,4.7]
		kwargs["x_limits"] = [-0.1,1]
		super(QtQ0EffectiveMassPostAnalysis, self).plot(*args, **kwargs)

def main():
	exit(("Exit: QtQ0EffectiveMassPostAnalysis not intended to be a "
		"standalone module."))

if __name__ == '__main__':
	main()