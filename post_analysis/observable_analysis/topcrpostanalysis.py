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

	dpi=400

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

		self.beta_values = sorted(data.beta_values)

		self._setup_flow_times()

		self._setup_analysis_types(data.analysis_types)

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
				self.topc2[atype][beta] = data.data_observables["topc2"] \
						[beta][self.ac][atype]

				# Q^4
				self.topc4[atype][beta] = data.data_observables["topc4"] \
					[beta][self.ac][atype]

				if self.with_autocorr:
					self.topc2[atype][beta]["ac"] = \
						data.data_observables["topc2"][beta] \
						["with_autocorr"]["autocorr"]

					self.topc4[atype][beta]["ac"] = \
						data.data_observables["topc4"][beta] \
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

		self._setup_article_values()
		self._normalize_article_values()

		self._setup_volumes()
		self._normalize_Q()
		self._calculate_Q4C()
		self._calculate_R()

	def _setup_volumes(self):
		"""Sets up lattice volumes."""
		vol = lambda b: self.total_lattice_sizes[b]*get_lattice_spacing(b)**4
		self.V = {b: vol(b) for b in self.beta_values}

		# print self.V

		# self.V2 = {}
		# for beta in self.beta_values:
		# 	a = get_lattice_spacing(beta)
		# 	L_s = a*self.lattice_sizes[beta][0]
		# 	L_t = a*self.lattice_sizes[beta][1]
		# 	self.V2[beta] = L_s**3 * L_t
		# 	print beta, a, L_s, L_t, self.V2[beta]
		# exit(1)

	def _normalize_Q(self):
		"""Normalizes Q4 and Q2"""
		for atype in self.analysis_types:
			for beta in self.beta_values:
				self.topc2[atype][beta]["y"] /= self.V[beta]
				self.topc2[atype][beta]["y_error"] /= self.V[beta]
				self.topc4[atype][beta]["y"] /= self.V[beta]**2
				self.topc4[atype][beta]["y_error"] /= self.V[beta]**2

	@staticmethod
	def Q4C(q4, q2):
		"""4th cumulant."""
		return q4 - 3 * q2**2

	@staticmethod
	def Q4C_error(q4, q4err, q2, q2err):
		"""4th cumulant error."""
		return np.sqrt(q4err**2 + (6*q2err*q2)**2 - 12*q2*q4err*q2err)

	@staticmethod 
	def R(q4c, q2):
		"""Returns the ratio <Q^4>C/<Q^2>"""
		return q4c/q2

	@staticmethod 
	def R_error(q4c, q4cerr, q2, q2err):
		"""Returns the ratio <Q^4>C/<Q^2>"""
		return np.sqrt(
			(q4cerr/q2)**2 + (q4c*q2err / q2**2)**2 - 2*q4cerr*q4c*q2err/q2**3)


	def _calculate_Q4C(self):
		"""Caluclates the 4th cumulant for my data."""

		# Gets Q4C and R
		for atype in self.analysis_types:

			for beta in self.beta_values:

				self.topc4C[atype][beta] = {

					"x": self.topc2[atype][beta]["x"],

					"y": self.Q4C(
						self.topc4[atype][beta]["y"], 
						self.topc2[atype][beta]["y"]),

					"y_error": self.Q4C_error(
						self.topc4[atype][beta]["y"],
						self.topc4[atype][beta]["y_error"],
						self.topc2[atype][beta]["y"],
						self.topc2[atype][beta]["y_error"]),
				}

	def _calculate_R(self):
		"""Calculates R = Q^4_C / Q^2 for my data."""

		# Gets Q4C and R
		for atype in self.analysis_types:

			for beta in self.beta_values:

				self.topcR[atype][beta] = {
				
					"x": self.topc2[atype][beta]["x"],

					"y": self.R(
						self.topc4C[atype][beta]["y"], 
						self.topc2[atype][beta]["y"]),

					"y_error": self.R_error(
						self.topc4C[atype][beta]["y"], 
						self.topc4C[atype][beta]["y_error"], 
						self.topc2[atype][beta]["y"], 
						self.topc2[atype][beta]["y_error"])
				}

				self.data[atype][beta] = self.topcR[atype][beta]

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

	def compare_lattice_values(self, atype="bootstrap"):
		"""
		Compares values at flow times given by the data we are comparing against
		"""

		x_pvals_article = []
		y_pvals_article = []

		x_pvals_me = []
		y_pvals_me = []

		article_data2 = {}
		for data_set in sorted(self.data_article.keys()):
			for size in sorted(self.data_article[data_set].keys()):
				article_data2[size] = {}

		for data_set in sorted(self.data_article.keys()):
			for size in sorted(self.data_article[data_set].keys()):
				article_data2[size][data_set] = self.data_article[data_set][size]


		# for size in sorted(self.data_article[data_set].keys()):
		for size in sorted(article_data2.keys()):
			t0 = self.data_article["B"][size]["t0"]
			print "="*150
			print "Reference value t0: %f" % t0
			print "\nMy data:"

			for beta in self.beta_values:
				# Gets the approximate same t0 ref. value
				t0_index = np.argmin(np.abs(self.topc2[atype][beta]["x"] - t0))
				print "Beta: %4.2f Q2: %10.5f Q2_err: %10.5f Q4: %10.5f \
Q4_err: %10.5f Q4C: %10.5f Q4C_err: %10.5f R: %10.5f R_err: %10.5f" % (beta,
					self.topc2[atype][beta]["y"][t0_index], 
					self.topc2[atype][beta]["y_error"][t0_index],
					self.topc4[atype][beta]["y"][t0_index], 
					self.topc4[atype][beta]["y_error"][t0_index],
					self.topc4C[atype][beta]["y"][t0_index], 
					self.topc4C[atype][beta]["y_error"][t0_index],
					self.topcR[atype][beta]["y"][t0_index], 
					self.topcR[atype][beta]["y_error"][t0_index])
			print "\nArticle data(normalized by volume):"

			for data_set in sorted(article_data2[size].keys()):

				print "Dataset: %s Beta: %2.2f Volume: %f t0: %f" % (
					data_set, self.data_article[data_set][size]["beta"],
					self.data_article[data_set][size]["V"],	t0)
				print "Q2:  %10.5f Q2_err:  %10.5f" % (
					self.data_article[data_set][size]["Q2_norm"],
					self.data_article[data_set][size]["Q2Err_norm"])
				print "Q4:  %10.5f Q4_err:  %10.5f" % (
					self.data_article[data_set][size]["Q4_norm"],
					self.data_article[data_set][size]["Q4Err_norm"])
				print "Q4C: %10.5f Q4C_err: %10.5f" % (
					self.data_article[data_set][size]["Q4C_norm"],
					self.data_article[data_set][size]["Q4CErr_norm"])
				print "R:   %10.5f R_err:   %10.5f" % (
					self.data_article[data_set][size]["R_norm"],
					self.data_article[data_set][size]["RErr_norm"])

				if size==1:
					x_pvals_article.append(t0)
					y_pvals_article.append((
						self.data_article[data_set][size]["R_norm"],
						self.data_article[data_set][size]["RErr_norm"])
					)
			print ""


	def set_analysis_data_type(self, analysis_data_type="bootstrap"):
		"""Sets the analysis type and retrieves correct analysis data."""
		self.analysis_data_type = analysis_data_type
		self.plot_values = {}
		self._initiate_plot_values(self.data[self.analysis_data_type],
			None)

	def _initiate_plot_values(self, data, data_raw):
		"""Sorts data into a format specific for the plotting method."""
		for beta in self.beta_values:
			values = {}
			values["a"] = get_lattice_spacing(beta)
			values["x"] = values["a"]* np.sqrt(8*data[beta]["x"])
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
		kwargs["error_shape"] = "band"
		kwargs["y_limits"] = [-1,1]
		# kwargs["x_limits"] = [0.5, 0.6]
		super(TopcRPostAnalysis, self).plot(*args, **kwargs)


	def _normalize_article_values(self):
		"""
		Normalizes values from article based on physical volume.
		"""
		for data_set in self.data_article:
			for size in self.data_article[data_set]:
				# Set up volume in physical units
				L = self.data_article[data_set][size]["L"]
				a = self.data_article[data_set][size]["a"]
				self.data_article[data_set][size]["aL"] = a*L
				V = float(self.data_article[data_set][size]["aL"]**4)
				self.data_article[data_set][size]["V"] = V

				# Normalize Q^2 by V
				Q2 = self.data_article[data_set][size]["Q2"]
				Q2Err = self.data_article[data_set][size]["Q2Err"]
				Q2_norm = Q2/V
				Q2Err_norm = Q2Err/V
				self.data_article[data_set][size]["Q2_norm"] = Q2_norm
				self.data_article[data_set][size]["Q2Err_norm"] = Q2Err_norm

				# Normalize Q^4 by V
				Q4 = self.data_article[data_set][size]["Q4"]
				Q4Err = self.data_article[data_set][size]["Q4Err"]
				Q4_norm = Q4/V**2
				Q4Err_norm = Q4Err/V**2
				self.data_article[data_set][size]["Q4_norm"] = Q4_norm
				self.data_article[data_set][size]["Q4Err_norm"] = Q4Err_norm

				# Recalculates 4th cumulant
				Q4C_norm = self.Q4C(Q4_norm, Q2_norm)
				Q4CErr_norm = self.Q4C_error(Q4_norm, Q4Err_norm, Q2_norm, 
					Q2Err_norm)
				self.data_article[data_set][size]["Q4C_norm"] = Q4C_norm
				self.data_article[data_set][size]["Q4CErr_norm"] = Q4CErr_norm

				# Recalculates R
				R_norm = self.R(Q4C_norm, Q2_norm)
				RErr_norm = self.R_error(Q4C_norm, Q4CErr_norm, Q2_norm, 
					Q2Err_norm)
				self.data_article[data_set][size]["R_norm"] = R_norm
				self.data_article[data_set][size]["RErr_norm"] = RErr_norm

		# for data_set in sorted(self.data_article.keys()):
		# 	for size in sorted(self.data_article[data_set].keys()):
		# 		print "="*50
		# 		print "Dataset: %s Size number: %s Volume: %f" % (
		# 			data_set, size, self.data_article[data_set][size]["V"])
		# 		print "Q2: %10.5f %10.5f" % (
		# 			self.data_article[data_set][size]["Q2_norm"],
		# 			self.data_article[data_set][size]["Q2Err_norm"])
		# 		print "Q4: %10.5f %10.5f" % (
		# 			self.data_article[data_set][size]["Q4_norm"],
		# 			self.data_article[data_set][size]["Q4Err_norm"])
		# 		print "Q4C: %10.5f %10.5f" % (
		# 			self.data_article[data_set][size]["Q4C_norm"],
		# 			self.data_article[data_set][size]["Q4CErr_norm"])
		# 		print "R: %10.5f %10.5f" % (
		# 			self.data_article[data_set][size]["R_norm"],
		# 			self.data_article[data_set][size]["RErr_norm"])

	def _setup_article_values(self):
		"""
		Sets up the article values from https://arxiv.org/abs/1506.06052

		Format:
			{Lattice type}/{Beta value}/{all other stuff}
		"""

		self.data_article = {
			"A": 
			{
				1: {
					"beta": 5.96,
					"t0": 2.79, # t0/a^2
					"L": 10,
					# "aL": 1.0, # [fm]
					"a": 0.102, # [fm]
					"Q2": 0.701,
					"Q2Err": 0.006,
					"Q4": 1.75,
					"Q4Err": 0.04,
					"Q4C": 0.273,
					"Q4CErr": 0.020,
					"R": 0.39,
					"RErr": 0.03,
				},
			},

			"B": 
			{
				1: {
					"beta": 5.96,
					"t0": 2.79, # t0/a^2
					"L": 12,
					# "aL": 1.2, # [fm]
					"a": 0.102, # [fm]
					"Q2": 1.617,
					"Q2Err": 0.006,
					"Q4": 8.15,
					"Q4Err": 0.07,
					"Q4C": 0.30,
					"Q4CErr": 0.04,
					"R": 0.187,
					"RErr": 0.024,
				},
				2: {
					"beta": 6.05,
					"t0": 3.78, # t0/a^2
					"L": 14,
					# "aL": 1.2, # [fm]
					"a": 0.087, # [fm]
					"Q2": 1.699,
					"Q2Err": 0.007,
					"Q4": 9.07,
					"Q4Err": 0.09,
					"Q4C": 0.41,
					"Q4CErr": 0.05,
					"R": 0.24,
					"RErr": 0.03,					
				},
				3: {
					"beta": 6.13,
					"t0": 4.87, # t0/a^2
					"L": 16,
					# "aL": 1.2, # [fm]
					"a": 0.077, # [fm]
					"Q2": 1.750,
					"Q2Err": 0.007,
					"Q4": 9.58,
					"Q4Err": 0.09,
					"Q4C": 0.39,
					"Q4CErr": 0.05,
					"R": 0.22,
					"RErr": 0.03,
				},
				4: {
					"beta": 6.21,
					"t0": 6.20, # t0/a^2
					"L": 18,
					# "aL": 1.2, # [fm]
					"a": 0.068, # [fm]
					"Q2": 1.741,
					"Q2Err": 0.007,
					"Q4": 9.44,
					"Q4Err": 0.09,
					"Q4C": 0.35,
					"Q4CErr": 0.05,
					"R": 0.20,
					"RErr": 0.03,
				},
			},

			"C": 
			{
				1: {
					"beta": 5.96,
					"t0": 2.79, # t0/a^2
					"L": 13,
					# "aL": 1.3, # [fm]
					"a": 0.102, # [fm]
					"Q2": 2.244,
					"Q2Err": 0.006,
					"Q4": 15.50,
					"Q4Err": 0.10,
					"Q4C": 0.40,
					"Q4CErr": 0.05,
					"R": 0.177,
					"RErr": 0.023,
				},
			},

			"D": {
				1: {
					"beta": 5.96,
					"t0": 2.79, # t0/a^2
					"L": 14,
					# "aL": 1.4, # [fm]
					"a": 0.102, # [fm]
					"Q2": 3.028,
					"Q2Err": 0.006,
					"Q4": 28.14,
					"Q4Err": 0.14,
					"Q4C": 0.63,
					"Q4CErr": 0.07,
					"R": 0.209,
					"RErr": 0.023,
				},
				2: {
					"beta": 6.05,
					"t0": 3.78, # t0/a^2
					"L": 17,
					# "aL": 1.5, # [fm]
					"a": 0.087, # [fm]
					"Q2": 3.686,
					"Q2Err": 0.014,
					"Q4": 41.6,
					"Q4Err": 0.4,
					"Q4C": 0.83,
					"Q4CErr": 0.19,
					"R": 0.22,
					"RErr": 0.05,
				},
				3: {
					"beta": 6.13,
					"t0": 4.87, # t0/a^2
					"L": 19,
					# "aL": 1.5, # [fm]
					"a": 0.077, # [fm]
					"Q2": 3.523,
					"Q2Err": 0.013,
					"Q4": 37.8,
					"Q4Err": 0.3,
					"Q4C": 0.56,
					"Q4CErr": 0.17,
					"R": 0.16,
					"RErr": 0.05,
				},
				4: {
					"beta": 6.21,
					"t0": 6.20, # t0/a^2
					"L": 21,
					# "aL": 1.4, # [fm]
					"a": 0.068, # [fm]
					"Q2": 3.266,
					"Q2Err": 0.012,
					"Q4": 32.7,
					"Q4Err": 0.3,
					"Q4C": 0.68,
					"Q4CErr": 0.15,
					"R": 0.21,
					"RErr": 0.05,
				},
			},

			"E": {
				1: {
					"beta": 5.96,
					"t0": 2.79, # t0/a^2
					"L": 15,
					# "aL": 1.5, # [fm]
					"a": 0.102, # [fm]
					"Q2": 3.982,
					"Q2Err": 0.006,
					"Q4": 48.38,
					"Q4Err": 0.18,
					"Q4C": 0.81,
					"Q4CErr": 0.09,
					"R": 0.202,
					"RErr": 0.023,
				},
			},

			"F": {
				1: {
					"beta": 5.96,
					"t0": 2.79, # t0/a^2
					"L": 16,
					# "aL": 1.6, # [fm]
					"a": 0.102, # [fm]
					"Q2": 5.167,
					"Q2Err": 0.006,
					"Q4": 80.90,
					"Q4Err": 0.22,
					"Q4C": 0.81,
					"Q4CErr": 0.11,
					"R": 0.157,
					"RErr": 0.022,
				},
			},
		}

def main():
	exit("Exit: TopcRPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()