from post_analysis.core.postcore import PostCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
import numpy as np
import os

class TopcRPostAnalysis(PostCore):
	"""
	Post-analysis of the topc ratio with Q^4_C/Q^2. Requires that Q4 and Q2 
	has been imported.
	"""

	observable_name = r"$R=\frac{\langle Q^4 \rangle_C}{\langle Q^2 \rangle}$"
	observable_name_compact = "topcr"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$R$"

	formula = r", $R = \langle Q^4_C \rangle = \langle Q^4 \rangle - $"
	formula += r"$3 \langle Q^2 \rangle^2$"

	total_lattice_sizes = {
		6.0: 24**3*48, 6.1: 28**3*56, 6.2: 32**3*64, 6.45: 48**3*96
	}

	def __init__(self, data, with_autocorr=True, figures_folder="../figures", 
		verbose=False, dryrun=False):
		"""
		Initializes this specialized form of finding the ratio of different 
		topological charge definitions.
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

		self.beta_values = sorted(data.beta_values)

		self.analysis_types = data.analysis_types
		if "autocorrelation" in self.analysis_types:
			self.analysis_types.remove("autocorrelation")

		# Q^2
		self.topc2 = {atype: {beta: {} for beta in self.beta_values} \
			for atype in self.analysis_types}
		
		# Q^4
		self.topc4 = {atype: {beta: {} for beta in self.beta_values} \
			for atype in self.analysis_types}

		# Q^4_C
		self.topc4C = {atype: {beta: {} for beta in self.beta_values} \
			for atype in self.analysis_types}

		# R = Q^4_C / Q^2
		self.topcR = {atype: {beta: {} for beta in self.beta_values} \
			for atype in self.analysis_types}

		# Data will be copied from R
		self.data = {atype: {beta: {} for beta in self.beta_values} \
			for atype in self.analysis_types}

		# Q^2 and Q^4 raw bs values
		self.topc2_raw = {}
		self.topc4_raw = {}
		self.topc4c_raw = {}
		self.topcR_raw = {}
		self.data_raw = {}

		for atype in data.raw_analysis:
			if atype == "autocorrelation":
				self.ac_raw = data.raw_analysis[atype]
			else:
				self.data_raw[atype] = data.raw_analysis[atype]

		# First, gets the topc2, then topc4
		for atype in self.analysis_types:
			for beta in self.beta_values:
				# Q^2
				self.topc2[atype][beta] = data.data_observables["topq2"] \
						[beta][self.ac][atype]

				# Q^4
				self.topc4[atype][beta] = data.data_observables["topq4"] \
					[beta][self.ac][atype]

				if self.with_autocorr:
					self.topc2[atype][beta]["ac"] = \
						data.data_observables["topq2"][beta] \
						["with_autocorr"]["autocorr"]

					self.topc4[atype][beta]["ac"] = \
						data.data_observables["topq4"][beta] \
						["with_autocorr"]["autocorr"]

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

		self._setup_volumes()
		# self._normalize_Q()
		self._calculate_R()

	def _setup_volumes(self):
		vol = lambda b: self.total_lattice_sizes[b]*get_lattice_spacing(b)**4
		self.V = {b: vol(b) for b in self.beta_values}

	def _calculate_Q4C(self, q2, q2raw, q4, q4raw, qraw, beta):
		"""Caluclates the 4th cumulent,

		<Q^4>_C = 1/N_conf sum^{N_conf}_{i=1} ((Q_i)^4 - 3<Q^2>)
		"""

		assert q2raw.shape == q4raw.shape

		V = self.V[beta] / get_lattice_spacing(beta)**4
		# Q2 = q2["y"] / V
		# Q4 = q4["y"] / V**2
		Q2 = q2["y"]
		Q4 = q4["y"]

		# Q4C = np.zeros(qraw.shape)
		# print qraw.shape
		# for iCfg in xrange(qraw.shape[1]):
		# 	Q4C[:,iCfg] = qraw[:,iCfg]**4 - 3*Q2**2

		# Q4C = np.mean(Q4C, axis=1)

		Q4C = Q4 - 3*Q2**2
		R = Q4C / Q2

		print "\n"
		print "Beta =", beta
		print "L    =", get_lattice_spacing(beta)*self.lattice_sizes[beta][0]
		print "V    =", V
		t_flow = [600, 601]
		print "Unnormalized"
		print "Q2[t_flow]      =", Q2[t_flow]
		print "3*Q2[t_flow]**2 =", 3*Q2[t_flow]**2
		print "Q4[t_flow]      =", Q4[t_flow]
		print "Q4C[t_flow]     =", Q4C[t_flow]
		print "R[t_flow]       =", R[t_flow]

		print "Normalized with just a^4 * V and a^8 * V^2:"
		V2 = V * get_lattice_spacing(beta)**4 
		V4 = V**2 * get_lattice_spacing(beta)**8
		print "Q2[t_flow]      =", Q2[t_flow]/V2
		print "3*Q2[t_flow]**2 =", 3*(Q2[t_flow]/V2)**2
		print "Q4[t_flow]      =", Q4[t_flow]/V4
		print "Q4C[t_flow]     =", Q4[t_flow]/V4 - 3*(Q2[t_flow]/V2)
		print "R[t_flow]       =", (Q4[t_flow]/V4 - 3*(Q2[t_flow]/V2)**2)/(Q2[t_flow]/V2)

		print "Q4 normalized with a^4 * V and a^4 * V:"
		# V2 = V * get_lattice_spacing(beta)**(4)
		V2 = 1
		V4 = V * get_lattice_spacing(beta)**(4)
		# print "Q2[t_flow]      =", Q2[t_flow/V2
		# print "3*Q2[t_flow]**2 =", 3*(Q2[t_flow]/V2)**2
		# print "Q4[t_flow]      =", Q4[t_flow]/V4
		# print "Q4C[t_flow]     =", Q4[t_flow]/V4 - 3*(Q2[t_flow]/V2)**2
		# print "R[t_flow]       =", (Q4[t_flow]/V4 - 3*(Q2[t_flow]/V2)**2)/(Q2[t_flow]/V2)
		print "Q2[t_flow]      =", Q2[t_flow]
		print "3*Q2[t_flow]**2 =", 3*(Q2[t_flow])**2
		print "Q4[t_flow]      =", Q4[t_flow]
		print "Q4C[t_flow]     =", (Q4[t_flow] - 3*(Q2[t_flow])**2)/V4
		print "R[t_flow]       =", (Q4[t_flow] - 3*(Q2[t_flow])**2)/V4/(Q2[t_flow])


		# import matplotlib.pyplot as plt
		# plt.plot(R)
		# plt.show()

		# exit("Exiting before returning..")

	def _normalize_Q(self):
		"""Normalizes Q4 and Q2"""
		for atype in self.analysis_types:
			for beta in self.beta_values:
				self.topc2[atype][beta]["y"] /= self.V[beta]
				self.topc2[atype][beta]["y_error"] /= self.V[beta]
				self.topc4[atype][beta]["y"] /= self.V[beta]**2
				self.topc4[atype][beta]["y_error"] /= self.V[beta]**2



	def _calculate_R(self):
		"""Calculates R = Q^4_C / Q^2."""

		# The 3 comes from subtracting the disconnected diagrams.
		Q4C = lambda q4, q2: q4 - 3 * q2**2

		Q4C_error = lambda q4, q4err, q2, q2err: np.sqrt(
			q4err**2 + (6*q2err*q2)**2 - 12*q2*q4err*q2err)

		R = lambda q4c, q2: q4c/q2

		R_error = lambda q4c, q4cerr, q2, q2err: np.sqrt(
			(q4cerr/q2)**2 + (q4c*q2err / q2**2)**2 - 2*q4cerr*q4c*q2err/q2**3)

		# Gets Q4C and R
		for atype in self.analysis_types:
			for beta in self.beta_values:

				self.topc4C[atype][beta] = self._calculate_Q4C(
					self.topc2[atype][beta], self.data_raw[atype][beta]["topq2"],
					self.topc4[atype][beta], self.data_raw[atype][beta]["topq4"],
					self.data_raw[atype][beta]["topc"], beta)

				self.topc4C[atype][beta] = {

					"y": Q4C(
						self.topc4[atype][beta]["y"], 
						self.topc2[atype][beta]["y"]),

					"y_error": Q4C_error(
						self.topc4[atype][beta]["y"],
						self.topc4[atype][beta]["y_error"],
						self.topc2[atype][beta]["y"],
						self.topc2[atype][beta]["y_error"]),
				}

				self.topcR[atype][beta] = {
				
					"y": R(
						self.topc4C[atype][beta]["y"], 
						self.topc2[atype][beta]["y"]),

					"y_error": R_error(
						self.topc4C[atype][beta]["y"], 
						self.topc4C[atype][beta]["y_error"], 
						self.topc2[atype][beta]["y"], 
						self.topc2[atype][beta]["y_error"])
				}

				self.data[atype][beta] = self.topcR[atype][beta]

		# comp_lattices = { # D ensembles
		# 	6.0: {"size": 14**4, "q2": 3.028, "q4": 28.14, "q4c": 0.63, "R": 0.209, "L": 14, "beta_article": 5.96},
		# 	6.1: {"size": 19**4, "q2": 3.523, "q4": 37.8, "q4c": 0.56, "R": 0.16, "L": 19, "beta_article": 6.13},
		# 	6.2: {"size": 21**4, "q2": 3.266, "q4": 32.7, "q4c": 0.68, "R": 0.21, "L": 21, "beta_article": 6.21},
		# }

		# for beta in self.beta_values:
		# 	print "="*100
		# 	for atype in self.analysis_types:
		# 		self.topc4c_raw[beta][atype] = Q4C(self.topc4_raw[beta][atype], self.topc2_raw[beta][atype])
		# 		self.topcR_raw[beta][atype] = R(self.topc4c_raw[beta][atype], self.topc2_raw[beta][atype])

		# 		_q2_mean = np.mean(self.topc2_raw[beta][atype], axis=1)
		# 		_q2_err = np.std(self.topc2_raw[beta][atype], axis=1)
		# 		_q4_mean = np.mean(self.topc4_raw[beta][atype], axis=1)
		# 		_q4_err = np.std(self.topc4_raw[beta][atype], axis=1)

		# 		if atype=="jackknife": # Bias corrects error
		# 			_q2_err *= np.sqrt(self.topc2_raw[beta][atype].shape[-1])
		# 			_q4_err *= np.sqrt(self.topc4_raw[beta][atype].shape[-1])

		# 		_q4c_mean = Q4C(_q4_mean, _q2_mean)
		# 		_q4c_err = Q4C_error(_q4_mean, _q4_err, _q2_mean, _q2_err)

		# 		_R_mean = R(_q4c_mean, _q2_mean)
		# 		_R_err = R_error(_q4c_mean, _q4c_err, _q2_mean, _q2_err)

		# 		scaling = self.total_lattice_sizes[beta] / float(comp_lattices[beta]["size"])
		# 		msg =  "\nBeta         %.2f" % beta
		# 		msg += "\nBeta_article %.2f" % comp_lattices[beta]["beta_article"]
		# 		msg += "\nFlow time 9.99"
		# 		msg += "\nScaling = Volume / Volume article = 2*%d^4 / %d^4 = %d / %d = %f" % ((self.total_lattice_sizes[beta]/2.)**(0.25), comp_lattices[beta]["L"], self.total_lattice_sizes[beta], comp_lattices[beta]["size"], scaling)
		# 		q2_scaled = _q2_mean[-1]/scaling
		# 		msg += "\nQ^2   %-14.4f" % self.topc2[beta][atype]["y"][-1]
		# 		msg += "   Q^2_scaled   %10.4f  Q^2_article   %10.4f  Difference(scaled-article): %10.4f  Factor_difference(scaled/article) %10.4f" % (q2_scaled, comp_lattices[beta]["q2"], q2_scaled-comp_lattices[beta]["q2"], q2_scaled/comp_lattices[beta]["q2"])
		# 		q4_scaled = _q4_mean[-1]/scaling
		# 		msg += "\nQ^4   %-14.4f" % self.topc4[beta][atype]["y"][-1]
		# 		msg += "   Q^4_scaled   %10.4f  Q^4_article   %10.4f  Difference(scaled-article): %10.4f  Factor_difference(scaled/article) %10.4f" % (q4_scaled, comp_lattices[beta]["q4"], q4_scaled-comp_lattices[beta]["q4"], q4_scaled/comp_lattices[beta]["q4"])
		# 		q4c_scaled = _q4c_mean[-1]/scaling
		# 		msg += "\nQ^4_C %-14.4f" % (self.topc4C[beta][atype]["y"][-1])
		# 		msg += "   Q^4_C_scaled %10.4f  Q^4_C_article %10.4f  Difference(scaled-article): %10.4f  Factor_difference(scaled/article) %10.4f" % (q4c_scaled, comp_lattices[beta]["q4c"], q4c_scaled - comp_lattices[beta]["q4c"], q4c_scaled/comp_lattices[beta]["q4c"])
		# 		R_scaled = _R_mean[-1]/scaling
		# 		# R_scaled = np.mean(self.topcR_raw[beta][atype],axis=1)[-1]
		# 		msg += "\nR     %-14.4f" % self.data[atype][beta]["y"][-1]
		# 		msg += "   R_scaled     %10.4f  R_article     %10.4f  Difference(scaled-article): %10.4f  Factor_difference(scaled/article) %10.4f" % (R_scaled, comp_lattices[beta]["R"], R_scaled - comp_lattices[beta]["R"], R_scaled/comp_lattices[beta]["R"])
		# 		if atype == "bootstrap":
		# 			print msg

		# 		self.data[atype][beta] = {"y": _R_mean, "y_error": _R_err}
		# 		self.data_raw[atype][beta] = self.topcR_raw[beta][atype]



	def set_analysis_data_type(self, analysis_data_type="bootstrap"):
		"""Sets the analysis type and retrieves correct analysis data."""
		self.analysis_data_type = analysis_data_type
		self.plot_values = {}
		self._initiate_plot_values(self.data[self.analysis_data_type],
			None)

	def _initiate_plot_values(self, data, data_raw):
		"""Sorts data into a format specific for the plotting method."""
		for beta in self.beta_values:
			if beta == 6.45: self.flow_time *= 2
			values = {}
			values["a"] = get_lattice_spacing(beta)
			values["x"] = values["a"]* np.sqrt(8*self.flow_time)
			values["y"] = data[beta]["y"]
			values["y_err"] = data[beta]["y_error"]
			# values["y_raw"] = data_raw[beta][self.observable_name_compact]
			# values["tau_int"] = data[beta]["ac"]["tau_int"]
			values["label"] = r"%s $\beta=%2.2f$" % (
				self.size_labels[beta], beta)
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