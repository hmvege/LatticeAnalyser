from core.postcore import PostCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import numpy as np
import os

class TopcRPostAnalysis(PostCore):
	"""Post-analysis of the topc ratio with Q^4_C/Q^2. Requires that Q4 and Q2 has been imported."""
	observable_name = r"$R=\frac{\langle Q^4 \rangle_C}{\langle Q^2 \rangle}$"
	observable_name_compact = "topqr"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$R=\frac{\langle Q^4 \rangle_C}{\langle Q^2 \rangle}$"

	formula = r", $\langle Q^4_C \rangle = \langle Q^4 \rangle - 3 \langle Q^2 \rangle^2 $"

	lattice_sizes = {6.0: 24**3*48, 6.1: 28**3*56, 6.2: 32**3*64, 6.45: 48**3*96}

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

		# # Q^2
		# self.topc2_unanalyzed_data = {}
		# self.topc2_bootstrap_data	= {}
		# self.topc2_jackknife_data = {}

		# # Q^4
		# self.topc4_unanalyzed_data = {}
		# self.topc4_bootstrap_data	= {}
		# self.topc4_jackknife_data = {}

		# # Q^4_C
		# self.topc4C_unanalyzed_data = {}
		# self.topc4C_bootstrap_data	= {}
		# self.topc4C_jackknife_data = {}

		# # R = Q^4_C / Q^2
		# self.topcR_unanalyzed_data = {}
		# self.topcR_bootstrap_data	= {}
		# self.topcR_jackknife_data = {}

		# # Q^2 and Q^4 raw bs values
		# self.topc2_raw_bs = {}
		# self.topc4_raw_bs = {}
		# self.topc4c_raw_bs = {}
		# self.topcR_raw_bs = {}

		# # First, gets the topc2, then topc4
		# self.beta_values = []
		# for beta in sorted(data.beta_values):
		# 	# Q^2
		# 	self.topc2_unanalyzed_data[beta] = data.data_observables["topq2"][beta][self.ac]["unanalyzed"]
		# 	self.topc2_bootstrap_data[beta] = data.data_observables["topq2"][beta][self.ac]["bootstrap"]
		# 	self.topc2_jackknife_data[beta] = data.data_observables["topq2"][beta][self.ac]["jackknife"]

		# 	# Q^4			
		# 	self.topc4_unanalyzed_data[beta] = data.data_observables["topq4"][beta][self.ac]["unanalyzed"]
		# 	self.topc4_bootstrap_data[beta] = data.data_observables["topq4"][beta][self.ac]["bootstrap"]
		# 	self.topc4_jackknife_data[beta] = data.data_observables["topq4"][beta][self.ac]["jackknife"] 

		# 	# Q^4_C
		# 	self.topc4C_unanalyzed_data[beta] = {}
		# 	self.topc4C_bootstrap_data[beta] = {}
		# 	self.topc4C_jackknife_data[beta] = {}

		# 	# R = Q^4_C / Q^2
		# 	self.topcR_unanalyzed_data[beta] = {}
		# 	self.topcR_bootstrap_data[beta] = {}
		# 	self.topcR_jackknife_data[beta] = {}

		# 	self.topc2_raw_bs[beta] = data.raw_analysis["jackknife"][beta]["topq2"]
		# 	self.topc4_raw_bs[beta] = data.raw_analysis["jackknife"][beta]["topq4"]
		# 	self.topc4c_raw_bs[beta] = {}

		# 	self.beta_values.append(beta)


		# Q^2 and Q^4 raw bs values
		self.topc2_raw = {}
		self.topc4_raw = {}
		self.topc4c_raw = {}
		self.topcR_raw = {}

		# Data dictionaries
		self.unanalyzed_data = {}
		self.bootstrap_data = {}
		self.jackknife_data = {}

		# First, gets the topc2, then topc4
		self.beta_values = sorted(data.beta_values)
		self.analysis_types = data.analysis_types

		# Sets up dictionaries
		for beta in self.beta_values:
			self.topc2_raw[beta] = {}
			self.topc4_raw[beta] = {}
			self.topc4c_raw[beta] = {}
			self.topcR_raw[beta] = {}

		for beta in self.beta_values:
			for atype in self.analysis_types:
				if atype == "autocorrelation":
					continue

				self.topc2_raw[beta][atype] = data.raw_analysis[atype][beta]["topq2"]
				self.topc4_raw[beta][atype] = data.raw_analysis[atype][beta]["topq4"]
				# self.topc4c_raw[beta][atype] = data.raw_analysis[atype][beta]["topq4"] - 3*self.topc2_raw[beta][atype]**2
				# self.topcR_raw[beta][atype] = self.topc4c_raw[beta][atype] / data.raw_analysis[atype][beta]["topq2"]
	
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
			q4err**2 + (6*q2err*q2)**2 - 12*q2*q4err*q2err)

		R = lambda q4c, q2: q4c/q2
		R_error = lambda q4c, q4cerr, q2, q2err: np.sqrt((q4cerr/q2)**2 + (q4c*q2err / q2**2)**2 - 2*(q4cerr*q4c*q2err/q2**3))

		comp_lattices = { # D ensembles
			6.0: {"size": 14**4, "q2": 3.028, "q4": 28.14, "q4c": 0.63, "R": 0.209, "L": 14, "beta_article": 5.96},
			6.1: {"size": 19**4, "q2": 3.523, "q4": 37.8, "q4c": 0.56, "R": 0.16, "L": 19, "beta_article": 6.13},
			6.2: {"size": 21**4, "q2": 3.266, "q4": 32.7, "q4c": 0.68, "R": 0.21, "L": 21, "beta_article": 6.21},
		}

		for beta in self.beta_values:
			print "="*100
			for atype in self.analysis_types:
				if atype == "autocorrelation": continue
				self.topc4c_raw[beta][atype] = self.topc4_raw[beta][atype] - 3*self.topc2_raw[beta][atype]**2
				self.topcR_raw[beta][atype] = self.topc4c_raw[beta][atype] / self.topc2_raw[beta][atype]

				_q2_mean = np.mean(self.topc2_raw[beta][atype], axis=1)
				_q2_err = np.std(self.topc2_raw[beta][atype], axis=1)
				_q4_mean = np.mean(self.topc4_raw[beta][atype], axis=1)
				_q4_err = np.std(self.topc4_raw[beta][atype], axis=1)

				if atype=="jackknife": # Bias corrects error
					_q2_err *= np.sqrt(self.topc2_raw[beta][atype].shape[-1])
					_q4_err *= np.sqrt(self.topc4_raw[beta][atype].shape[-1])

				_q4c_mean = Q4C(_q4_mean, _q2_mean)
				_q4c_err = Q4C_error(_q4_mean, _q4_err, _q2_mean, _q2_err)

				_R_mean = R(_q4c_mean, _q2_mean)
				_R_err = R_error(_q4c_mean, _q4c_err, _q2_mean, _q2_err)

				scaling = self.lattice_sizes[beta] / float(comp_lattices[beta]["size"])
				msg =  "\nBeta         %.2f" % beta
				msg += "\nBeta_article %.2f" % comp_lattices[beta]["beta_article"]
				msg += "\nFlow time 9.99"
				msg += "\nScaling = Volume / Volume article = 2*%d^4 / %d^4 = %d / %d = %f" % ((self.lattice_sizes[beta]/2.)**(0.25), comp_lattices[beta]["L"], self.lattice_sizes[beta], comp_lattices[beta]["size"], scaling)
				q2_scaled = _q2_mean[-1]
				msg += "\nQ^2   %-14.4f" % np.mean(self.topc2_raw[beta][atype],axis=1)[-1]
				msg += "   Q^2_scaled   %10.4f  Q^2_article   %10.4f  Difference(scaled-article): %10.4f  Factor_difference(scaled/article) %10.4f" % (q2_scaled, comp_lattices[beta]["q2"], q2_scaled-comp_lattices[beta]["q2"], q2_scaled/comp_lattices[beta]["q2"])
				q4_scaled = _q4_mean[-1]
				msg += "\nQ^4   %-14.4f" % np.mean(self.topc4_raw[beta][atype],axis=1)[-1]
				msg += "   Q^4_scaled   %10.4f  Q^4_article   %10.4f  Difference(scaled-article): %10.4f  Factor_difference(scaled/article) %10.4f" % (q4_scaled, comp_lattices[beta]["q4"], q4_scaled-comp_lattices[beta]["q4"], q4_scaled/comp_lattices[beta]["q4"])
				q4c_scaled = _q4c_mean[-1]
				msg += "\nQ^4_C %-14.4f" % (np.mean(self.topc4c_raw[beta][atype],axis=1)[-1])
				msg += "   Q^4_C_scaled %10.4f  Q^4_C_article %10.4f  Difference(scaled-article): %10.4f  Factor_difference(scaled/article) %10.4f" % (q4c_scaled, comp_lattices[beta]["q4c"], q4c_scaled - comp_lattices[beta]["q4c"], q4c_scaled/comp_lattices[beta]["q4c"])
				R_scaled = _R_mean[-1]
				# R_scaled = np.mean(self.topcR_raw[beta][atype],axis=1)[-1]
				msg += "\nR     %-14.4f" % np.mean(self.topcR_raw[beta][atype],axis=1)[-1]
				msg += "   R_scaled     %10.4f  R_article     %10.4f  Difference(scaled-article): %10.4f  Factor_difference(scaled/article) %10.4f" % (R_scaled, comp_lattices[beta]["R"], R_scaled - comp_lattices[beta]["R"], R_scaled/comp_lattices[beta]["R"])
				if atype == "bootstrap":
					print msg
					print self.topc2_raw[beta][atype].shape

				if atype=="unanalyzed":
					self.unanalyzed_data[beta] = {
						"y": _R_mean,
						"y_error": _R_err}
				elif atype=="bootstrap":
					self.bootstrap_data[beta] = {
						"y": _R_mean,
						"y_error": _R_err}
				elif atype=="jackknife":
					self.jackknife_data[beta] = {
						"y": _R_mean,
						"y_error": _R_err}
				else:
					print "oops: should not see this"

		# for beta in self.beta_values:
			# self.unanalyzed_data[beta] = {
			# 	"y": np.mean(self.topcR_raw[beta]["unanalyzed"], axis=1), 
			# 	"y_error": np.std(self.topcR_raw[beta]["unanalyzed"], axis=1)}
			# self.bootstrap_data[beta] = {
			# 	"y": np.mean(self.topcR_raw[beta]["bootstrap"], axis=1),
			# 	"y_error": np.std(self.topcR_raw[beta]["bootstrap"], axis=1)}
			# self.jackknife_data[beta] = {
			# 	"y": np.mean(self.topcR_raw[beta]["jackknife"], axis=1),
			# 	"y_error": np.std(self.topcR_raw[beta]["jackknife"], axis=1)*np.sqrt(self.topcR_raw[beta]["jackknife"].shape[-1])}

		# R = lambda q4c, q2: q4c / q2

		# R_error = lambda q4c, q4cerr, q2, q2err: np.sqrt(
		# 	(q4cerr / q2)**2 + (q4c * q2err / q2**2)**2 - q4c/q2**3*q4cerr*q2err)

		# # First, gets R
		# for beta in self.beta_values:
		# 	self.topcR_unanalyzed_data[beta]["y"] = R(
		# 		self.topc4C_unanalyzed_data[beta]["y"], 
		# 		self.topc2_unanalyzed_data[beta]["y"])
		# 	self.topcR_bootstrap_data[beta]["y"] = R(
		# 		self.topc4C_bootstrap_data[beta]["y"],
		# 		self.topc2_bootstrap_data[beta]["y"])
		# 	self.topcR_jackknife_data[beta]["y"] = R(
		# 		self.topc4C_jackknife_data[beta]["y"],
		# 		self.topc2_jackknife_data[beta]["y"])

		# 	self.topcR_unanalyzed_data[beta]["y_error"] = R_error(
		# 		self.topc4C_unanalyzed_data[beta]["y"], 
		# 		self.topc4C_unanalyzed_data[beta]["y_error"], 
		# 		self.topc2_unanalyzed_data[beta]["y"], 
		# 		self.topc2_unanalyzed_data[beta]["y_error"])
		# 	self.topcR_bootstrap_data[beta]["y_error"] = R_error(
		# 		self.topc4C_bootstrap_data[beta]["y"], 
		# 		self.topc4C_bootstrap_data[beta]["y_error"], 
		# 		self.topc2_bootstrap_data[beta]["y"], 
		# 		self.topc2_bootstrap_data[beta]["y_error"])
		# 	self.topcR_jackknife_data[beta]["y_error"] = R_error(
		# 		self.topc4C_jackknife_data[beta]["y"], 
		# 		self.topc4C_jackknife_data[beta]["y_error"], 
		# 		self.topc2_jackknife_data[beta]["y"], 
		# 		self.topc2_jackknife_data[beta]["y_error"])

		# 	self.topcR_raw_bs[beta] = {"y": np.mean(R(self.topc4c_raw_bs[beta],
		# 		self.topc2_raw_bs[beta]), axis=1), "y_error": np.std(R(self.topc4c_raw_bs[beta],
		# 		self.topc2_raw_bs[beta]), axis=1)}

		# self.unanalyzed_data = self.topcR_unanalyzed_data
		# self.bootstrap_data	= self.topcR_bootstrap_data
		# self.jackknife_data = self.topcR_jackknife_data

		# self.bootstrap_data	= self.topcR_raw_bs

		# Q2 = self.topc2_bootstrap_data[6.2]["y"]
		# Q4 = self.topc4_bootstrap_data[6.2]["y"]
		# print Q2[-10:]**2*3
		# print Q4[-10:]
		# print Q4[-10:] - 3 * Q2[-10:]**2
		# # print self.topc4C_bootstrap_data[6.2]["y"]
		# exit(1)

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
		"""Sets the analysis type and retrieves correct analysis data."""
		self.plot_values = {}
		data = self._get_analysis_data(analysis_data_type)
		self._initiate_plot_values(data)

		# Makes it a global constant so it can be added in plot figure name
		self.analysis_data_type = analysis_data_type

	def _initiate_plot_values(self, data):
		"""Sorts data into a format specific for the plotting method."""
		for beta in self.beta_values:
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