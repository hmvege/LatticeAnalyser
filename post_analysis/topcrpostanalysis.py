from core.postcore import PostCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import numpy as np
import os

class TopcRPostAnalysis(PostCore):
	"""Post-analysis of the topc ratio with Q^4/Q^2. Requires that Q4 and Q2 has been imported."""
	observable_name = r"$R=\frac{\langle Q^4 \rangle_C}{\langle Q^2 \rangle}$"
	observable_name_compact = "topqr"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$R=\frac{\langle Q^4 \rangle_C}{\langle Q^2 \rangle}$"

	formula = r", $\langle Q^4_C \rangle = \langle Q^4 \rangle - 3 \langle Q^2 \rangle^2 $"

	lattice_sizes = {6.0: 24**3*48, 6.1: 28**3*56, 6.2: 32**3*64, 6.45: 48**3*96}
	hbarc = 0.19732697 #eV micro m

	def __init__(self, data, with_autocorr=True, figures_folder="../figures", verbose=False, dryrun=False):
		"""
		Initializes this specialized form of finding the ratio of different topological charge definitions
		"""
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

		# Q^2
		self.topc2_unanalyzed_data = {}
		self.topc2_bootstrap_data	= {}
		self.topc2_jackknife_data = {}

		# Q^4
		self.topc4_unanalyzed_data = {}
		self.topc4_bootstrap_data	= {}
		self.topc4_jackknife_data = {}

		# Q^4_C
		self.topc4C_unanalyzed_data = {}
		self.topc4C_bootstrap_data	= {}
		self.topc4C_jackknife_data = {}

		# R = Q^4_C / Q^2
		self.topcR_unanalyzed_data = {}
		self.topcR_bootstrap_data	= {}
		self.topcR_jackknife_data = {}

		# Q^2 and Q^4 raw bs values
		self.topc2_raw_bs = {}
		self.topc4_raw_bs = {}
		self.topc4c_raw_bs = {}
		self.topcR_raw_bs = {}

		# First, gets the topc2, then topc4
		self.beta_values = []
		for beta in sorted(data.beta_values):
			# Q^2
			self.topc2_unanalyzed_data[beta] = data.data_observables["topq2"][beta][self.ac]["unanalyzed"]
			self.topc2_bootstrap_data[beta] = data.data_observables["topq2"][beta][self.ac]["bootstrap"]
			self.topc2_jackknife_data[beta] = data.data_observables["topq2"][beta][self.ac]["jackknife"]

			# Q^4			
			self.topc4_unanalyzed_data[beta] = data.data_observables["topq4"][beta][self.ac]["unanalyzed"]
			self.topc4_bootstrap_data[beta] = data.data_observables["topq4"][beta][self.ac]["bootstrap"]
			self.topc4_jackknife_data[beta] = data.data_observables["topq4"][beta][self.ac]["jackknife"] 

			# Q^4_C
			self.topc4C_unanalyzed_data[beta] = {}
			self.topc4C_bootstrap_data[beta] = {}
			self.topc4C_jackknife_data[beta] = {}

			# R = Q^4_C / Q^2
			self.topcR_unanalyzed_data[beta] = {}
			self.topcR_bootstrap_data[beta] = {}
			self.topcR_jackknife_data[beta] = {}

			self.topc2_raw_bs[beta] = data.raw_analysis["jackknife"][beta]["topq2"]
			self.topc4_raw_bs[beta] = data.raw_analysis["jackknife"][beta]["topq4"]
			self.topc4c_raw_bs[beta] = {}

			self.beta_values.append(beta)

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

		# Creates colors to use
		self.colors = {}
		for color, beta in zip(self.beta_colors, data.beta_values):
			self.colors[beta] = color

		self._calculate_R()

	def _calculate_R(self):
		"""Calculates R = Q^4_C / Q^2."""

		# The 3 comes from subtracting the disconnected diagrams.
		Q4C = lambda q4, q2: q4 - 3 * q2**2

		Q4C_error = lambda q4, q4err, q2, q2err: np.sqrt(
			q4err**2 + (6*q2err*q2)**2 - 6*q2*q4err*q2err)

		# Q4C_error = lambda q4, q4err, q2, q2err: np.sqrt(
		# 	q4err**2 + (6*q2err*q2)**2)


		# First, gets the Q^4_C
		for beta in self.beta_values:
			self.topc4C_unanalyzed_data[beta]["y"] = Q4C(
				self.topc4_unanalyzed_data[beta]["y"], 
				self.topc2_unanalyzed_data[beta]["y"])
			self.topc4C_bootstrap_data[beta]["y"] = Q4C(
				self.topc4_bootstrap_data[beta]["y"], 
				self.topc2_bootstrap_data[beta]["y"])
			self.topc4C_jackknife_data[beta]["y"] = Q4C(
				self.topc4_jackknife_data[beta]["y"], 
				self.topc2_jackknife_data[beta]["y"])

			# Finds the propagated error
			self.topc4C_unanalyzed_data[beta]["y_error"] = Q4C_error(
				self.topc4_unanalyzed_data[beta]["y"],
				self.topc4_unanalyzed_data[beta]["y_error"],
				self.topc2_unanalyzed_data[beta]["y"],
				self.topc2_unanalyzed_data[beta]["y_error"])
			self.topc4C_bootstrap_data[beta]["y_error"] = Q4C_error(
				self.topc4_bootstrap_data[beta]["y"],
				self.topc4_bootstrap_data[beta]["y_error"],
				self.topc2_bootstrap_data[beta]["y"],
				self.topc2_bootstrap_data[beta]["y_error"])
			self.topc4C_jackknife_data[beta]["y_error"] = Q4C_error(
				self.topc4_jackknife_data[beta]["y"],
				self.topc4_jackknife_data[beta]["y_error"],
				self.topc2_jackknife_data[beta]["y"],
				self.topc2_jackknife_data[beta]["y_error"])

			# Finds raw bs data values
			self.topc4c_raw_bs[beta] = Q4C(self.topc4_raw_bs[beta], 
				self.topc2_raw_bs[beta])

		R = lambda q4c, q2: q4c / q2

		R_error = lambda q4c, q4cerr, q2, q2err: np.sqrt(
			(q4cerr / q2)**2 + (q4c * q2err / q2**2)**2 - q4c/q2**3*q4cerr*q2err)

		# R_error = lambda q4c, q4cerr, q2, q2err: np.sqrt(
		# 	(q4cerr / q2)**2 + (q4c * q2err / q2**2)**2)

		# First, gets R
		for beta in self.beta_values:
			self.topcR_unanalyzed_data[beta]["y"] = R(
				self.topc4C_unanalyzed_data[beta]["y"], 
				self.topc2_unanalyzed_data[beta]["y"])
			self.topcR_bootstrap_data[beta]["y"] = R(
				self.topc4C_bootstrap_data[beta]["y"],
				self.topc2_bootstrap_data[beta]["y"])
			self.topcR_jackknife_data[beta]["y"] = R(
				self.topc4C_jackknife_data[beta]["y"],
				self.topc2_jackknife_data[beta]["y"])

			self.topcR_unanalyzed_data[beta]["y_error"] = R_error(
				self.topc4C_unanalyzed_data[beta]["y"], 
				self.topc4C_unanalyzed_data[beta]["y_error"], 
				self.topc2_unanalyzed_data[beta]["y"], 
				self.topc2_unanalyzed_data[beta]["y_error"])
			self.topcR_bootstrap_data[beta]["y_error"] = R_error(
				self.topc4C_bootstrap_data[beta]["y"], 
				self.topc4C_bootstrap_data[beta]["y_error"], 
				self.topc2_bootstrap_data[beta]["y"], 
				self.topc2_bootstrap_data[beta]["y_error"])
			self.topcR_jackknife_data[beta]["y_error"] = R_error(
				self.topc4C_jackknife_data[beta]["y"], 
				self.topc4C_jackknife_data[beta]["y_error"], 
				self.topc2_jackknife_data[beta]["y"], 
				self.topc2_jackknife_data[beta]["y_error"])

			self.topcR_raw_bs[beta] = {"y": np.mean(R(self.topc4c_raw_bs[beta],
				self.topc2_raw_bs[beta]), axis=1), "y_error": np.std(R(self.topc4c_raw_bs[beta],
				self.topc2_raw_bs[beta]), axis=1)}

		self.unanalyzed_data = self.topcR_unanalyzed_data
		self.bootstrap_data	= self.topcR_bootstrap_data
		self.jackknife_data = self.topcR_jackknife_data

		self.bootstrap_data	= self.topcR_raw_bs

		# Q2 = self.topc2_bootstrap_data[6.2]["y"]
		# Q4 = self.topc4_bootstrap_data[6.2]["y"]
		# print Q2[-10:]**2*3
		# print Q4[-10:]
		# print Q4[-10:] - 3 * Q2[-10:]**2
		# # print self.topc4C_bootstrap_data[6.2]["y"]
		# exit(1)

	def _initiate_plot_values(self, data):
		"""Sorts data into a format specific for the plotting method."""
		for beta in sorted(data.keys()):
			if beta == 6.45: self.flow_time *= 2
			values = {}
			values["a"] = get_lattice_spacing(beta)
			values["x"] = values["a"]* np.sqrt(8*self.flow_time)
			values["y"] = data[beta]["y"]
			values["y_err"] = data[beta]["y_error"]
			values["label"] = r"%s $\beta=%2.2f$" % (self.size_labels[beta], beta)
			values["color"] = self.colors[beta]
			self.plot_values[beta] = values

	def plot(self, *args, **kwargs):
		"""Ensuring I am plotting with formule in title."""
		kwargs["plot_with_formula"] = True
		super(TopcRPostAnalysis, self).plot(*args, **kwargs)

def main():
	exit("Exit: TopcRPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()