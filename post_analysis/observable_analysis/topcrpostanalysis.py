from post_analysis.core.postcore import PostCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
from tools.sciprint import sciprint
from tools.table_printer import TablePrinter
from collections import OrderedDict
import numpy as np
import os

class TopcRPostAnalysis(PostCore):
	"""
	Post-analysis of the topc ratio with Q^4_C/Q^2. Requires that Q4 and Q2 
	has been imported.
	"""

	observable_name = r"$R=\frac{\langle Q^4 \rangle_C}{\langle Q^2 \rangle}$"
	observable_name_compact = "topcr"
	x_label = r"$\sqrt{8t_{f}}[fm]$"
	y_label = r"$R$"

	formula = r", $R = \langle Q^4_C \rangle = \langle Q^4 \rangle - $"
	formula += r"$3 \langle Q^2 \rangle^2$"

	print_latex = False

	dpi=400

	lattice_sizes = {}

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
		self.colors = data.colors
		self.lattice_sizes = data.lattice_volumes
		self.size_labels = data.labels
		self.reference_values = data.reference_values
		self.print_latex = data.print_latex
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
		vol = lambda b: self.lattice_sizes[b]*get_lattice_spacing(b)[0]**4
		self.V = {b: vol(b) for b in self.beta_values}

	def _normalize_Q(self):
		"""Normalizes Q4 and Q2"""
		for atype in self.analysis_types:
			for beta in self.beta_values:
				self.topc2[atype][beta]["y"] /= self.V[beta]
				self.topc2[atype][beta]["y_error"] /= self.V[beta]
				self.topc4[atype][beta]["y"] /= self.V[beta]**2
				self.topc4[atype][beta]["y_error"] /= self.V[beta]**2

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
			values["a"] = get_lattice_spacing(beta)[0]
			values["x"] = values["a"]* np.sqrt(8*data[beta]["x"])
			values["y"] = data[beta]["y"]
			values["y_err"] = data[beta]["y_error"]
			# values["y_raw"] = data_raw[beta][self.observable_name_compact]
			# values["tau_int"] = data[beta]["ac"]["tau_int"]
			values["label"] = r"%s $\beta=%2.2f$" % (
				self.size_labels[beta], beta)
			self.plot_values[beta] = values

	def plot(self, *args, **kwargs):
		"""Ensuring I am plotting with formule in title."""
		kwargs["plot_with_formula"] = True
		kwargs["error_shape"] = "band"
		kwargs["y_limits"] = [-1,1]
		# kwargs["x_limits"] = [0.5, 0.6]
		super(TopcRPostAnalysis, self).plot(*args, **kwargs)

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

	@staticmethod
	def ratio_error(x, xerr, y, yerr):
		"""Returns the ratio and and error between two quantities."""
		return x/y, np.sqrt((xerr/y)**2 + (x*yerr/y**2)**2)

	def _print_data(self, t0=None):
		"""Prints data."""

		article_param_header = [r"Lattice", r"$\beta$", r"$L/a$", r"$L[\fm]$",
			r"$a[\fm]$", r"$t_0/{a^2}$", r"$t_0/{r_0^2}$",
		]

		art_flat = self.article_flattened
		article_param_table = [
			art_flat.keys(),
			[art_flat[k]["beta"] for k in art_flat],
			[art_flat[k]["L"] for k in art_flat],
			[art_flat[k]["aL"] for k in art_flat],
			[art_flat[k]["a"] for k in art_flat],
			[sciprint(art_flat[k]["t0"], art_flat[k]["t0err"], prec=4)
				for k in art_flat],
			[sciprint(art_flat[k]["t0r02"], art_flat[k]["t0r02err"], prec=4, 
				force_prec=True) for k in art_flat],
		]

		art_param_table_printer = TablePrinter(article_param_header, 
			article_param_table)
		art_param_table_printer.print_table(width=15, 
			row_seperator_positions=[6, 8, 10])

		article_values_header = [r"Lattice", r"$\langle Q^2 \rangle$", 
			r"$\langle Q^4 \rangle$", r"$\langle Q^4 \rangle_C$", r"$R$"]
		article_values_table = [
			art_flat.keys(),
			[sciprint(art_flat[k]["Q2"], art_flat[k]["Q2Err"], prec=3) 
				for k in art_flat],
			[sciprint(art_flat[k]["Q4"], art_flat[k]["Q4Err"], prec=2)
				for k in art_flat],
			[sciprint(art_flat[k]["Q4C"], art_flat[k]["Q4CErr"], prec=3)
				for k in art_flat],
			[sciprint(art_flat[k]["R"], art_flat[k]["RErr"], prec=3)
				for k in art_flat],
		]

		art_values_table_printer = TablePrinter(article_values_header, 
			article_values_table)
		art_values_table_printer.print_table(width=15, 
			row_seperator_positions=[6, 8, 10])

		article_normed_header = [r"Lattice", 
			r"$\langle Q^2 \rangle_\text{normed}$", 
			r"$\langle Q^4 \rangle_\text{normed}$", 
			r"$\langle Q^4 \rangle_{C,\text{normed}}$", r"$R_\text{normed}$"]
		article_normed_table = [
			art_flat.keys(),
			[sciprint(art_flat[k]["Q2_norm"], art_flat[k]["Q2Err_norm"], prec=3) 
				for k in art_flat],
			[sciprint(art_flat[k]["Q4_norm"], art_flat[k]["Q4Err_norm"], prec=3)
				for k in art_flat],
			[sciprint(art_flat[k]["Q4C_norm"], art_flat[k]["Q4CErr_norm"], prec=3)
				for k in art_flat],
			[sciprint(art_flat[k]["R_norm"], art_flat[k]["RErr_norm"], prec=3)
				for k in art_flat],
		]

		art_normed_table_printer = TablePrinter(article_normed_header, 
			article_normed_table)
		art_normed_table_printer.print_table(width=15, 
			row_seperator_positions=[6, 8, 10])

		# TODO: Prints values I have generated
		ref_vals = self.reference_values
		values_header = [r"$\beta$", r"$L/a$", r"$\langle Q^2 \rangle$", 
			r"$\langle Q^4 \rangle$", r"$\langle Q^4 \rangle_C$", r"$R$"]
		values_table = [
			[b for b in self.beta_values],
			["{:.2f}".format(self.data_values[b]["aL"])
				for b in self.beta_values],
			[sciprint(self.data_values[b]["Q2"], self.data_values[b]["Q2Err"])
				for b in self.beta_values],
			[sciprint(self.data_values[b]["Q4"], self.data_values[b]["Q4Err"])
				for b in self.beta_values],
			[sciprint(self.data_values[b]["Q4C"], self.data_values[b]["Q4CErr"])
				for b in self.beta_values],
			[sciprint(self.data_values[b]["R"], self.data_values[b]["RErr"])
				for b in self.beta_values],
		]

		values_table_printer = TablePrinter(values_header, values_table)
		values_table_printer.print_table(width=15)

		# TODO: Prints ratio between me and article
		ratio_header = [r"Lattice", r"$\beta$", 
			r"$\text{Ratio}(\langle Q^2 \rangle)$",
			r"$\text{Ratio}(\langle Q^4 \rangle)$", 
			r"$\text{Ratio}(\langle Q^4 \rangle_C)$", r"$\text{Ratio}(R)$"]
		ratio_table = []
		for fk in self.article_flattened:
			for b in self.beta_values:
				sub_list = []
				
				sub_list.append(fk)
				sub_list.append(b)
				sub_list.append(sciprint(
					self.data_ratios[fk][b]["Q2"],
					self.data_ratios[fk][b]["Q2Err"]))
				sub_list.append(sciprint(
					self.data_ratios[fk][b]["Q4"],
					self.data_ratios[fk][b]["Q4Err"]))
				sub_list.append(sciprint(
					self.data_ratios[fk][b]["Q4C"],
					self.data_ratios[fk][b]["Q4CErr"]))
				sub_list.append(sciprint(
					self.data_ratios[fk][b]["R"],
					self.data_ratios[fk][b]["RErr"]))

				ratio_table.append(sub_list)

		ratio_table = np.asarray(ratio_table).T.tolist()
		ratio_tab_pos = (4*np.arange(len(self.article_flattened)))
		ratio_table_printer = TablePrinter(ratio_header, ratio_table)
		ratio_table_printer.print_table(width=15,
			row_seperator_positions=ratio_tab_pos)

		print "Reference scale t0(my data): %s" % self.t0

	def _setup_comparison_values(self, tf="t0beta_a2", atype="bootstrap"):
		"""
		Sets up a new dictionary with my own values for comparing with the 
		article values.
		"""
		self.data_values = {b: {} for b in self.beta_values}

		self._get_tf_value(tf, atype, None)

		try:
			ref_vals = self.reference_values[atype]["bootstrap"]
		except KeyError:
			fallback_key = self.reference_values[atype].keys()[0]
			print ("Bootstrap line extrapolation not found."
				" Falling back to %s" % fallback_key)
			ref_vals = self.reference_values[atype][fallback_key]

		t0_indexes = [
			np.argmin(np.abs(self.topc2[atype][b]["x"] - self.t0[b])) 
			for b in self.beta_values
		]

		for t0_index, b in zip(t0_indexes, self.beta_values):
			self.data_values[b]["aL"] = ref_vals[b]["aL"]

			self.data_values[b]["Q2"] = \
				self.topc2[atype][b]["y"][t0_index]
			self.data_values[b]["Q2Err"] = \
				self.topc2[atype][b]["y_error"][t0_index]

			self.data_values[b]["Q4"] = \
				self.topc4[atype][b]["y"][t0_index]
			self.data_values[b]["Q4Err"] = \
				self.topc4[atype][b]["y_error"][t0_index]

			self.data_values[b]["Q4C"] = \
				self.topc4C[atype][b]["y"][t0_index]
			self.data_values[b]["Q4CErr"] = \
				self.topc4C[atype][b]["y_error"][t0_index]

			self.data_values[b]["R"] = \
				self.topcR[atype][b]["y"][t0_index]
			self.data_values[b]["RErr"] = \
				self.topcR[atype][b]["y_error"][t0_index]

		self.data_ratios = {k: {b: {} for b in self.beta_values} 
			for k in self.article_flattened.keys()}

		for flat_key in self.article_flattened:
			beta_article = self.article_flattened[flat_key]["beta"]
			for t0_index, b in zip(t0_indexes, self.beta_values):
				[self.data_ratios[flat_key][b]["Q2"],
					self.data_ratios[flat_key][b]["Q2Err"]] = \
					self.ratio_error(self.data_values[b]["Q2"],
						self.data_values[b]["Q2Err"],
						self.article_flattened[flat_key]["Q2_norm"], 
						self.article_flattened[flat_key]["Q2Err_norm"])

				[self.data_ratios[flat_key][b]["Q4"], 
					self.data_ratios[flat_key][b]["Q4Err"]] = \
					self.ratio_error(self.data_values[b]["Q4"],
						self.data_values[b]["Q4Err"],
						self.article_flattened[flat_key]["Q4_norm"], 
						self.article_flattened[flat_key]["Q4Err_norm"])

				[self.data_ratios[flat_key][b]["Q4C"], 
					self.data_ratios[flat_key][b]["Q4CErr"]] = \
					self.ratio_error(self.data_values[b]["Q4C"],
						self.data_values[b]["Q4CErr"],
						self.article_flattened[flat_key]["Q4C_norm"], 
						self.article_flattened[flat_key]["Q4CErr_norm"])

				[self.data_ratios[flat_key][b]["R"],
					self.data_ratios[flat_key][b]["RErr"]] = \
					self.ratio_error(self.data_values[b]["R"],
						self.data_values[b]["RErr"],
						self.article_flattened[flat_key]["R_norm"], 
						self.article_flattened[flat_key]["RErr_norm"])

		# arr = np.asarray(self.data_ratios)
		# print arr
		# print arr.shape
		# print arr.reshape(arr.shape[0]*arr.shape[1])



	def compare_lattice_values(self, tf=None, atype="bootstrap"):
		"""
		Compares values at flow times given by the data we are comparing against
		"""

		x_pvals_article = []
		y_pvals_article = []

		x_pvals_me = []
		y_pvals_me = []

		def ratio_error(x, xerr, y, yerr):
			return x/y, np.sqrt((xerr/y)**2 + (x*yerr/y**2)**2)

		self._setup_comparison_values(tf=tf, atype=atype)
		self._print_data()

		# for size in sorted(self.article_name_size[data_set]):
		for size in sorted(self.article_size_name):

		# 	# Sets the t0 value to extract at.
		# 	if tf == "article":
		# 		self.t0 = {b: self.article_name_size["B"][size]["t0cont"] 
		# 			for b in self.beta_values}
		# 	else:
		# 		tf = "t0beta_a2"
		# 		self._get_tf_value(tf, atype, None)

		# 	print "="*150
		# 	print "Reference value type %s t0: %s" % (tf, self.t0)

		# 	print "\nMy data:"
		# 	for beta in self.beta_values:
		# 		# Gets the approximate same t0 ref. value
		# 		t0_index = np.argmin(np.abs(self.topc2[atype][beta]["x"] - self.t0[beta]))

		# 		print ("Beta: %4.2f t0: %4.2f Q2: %10.5f Q2_err: %10.5f Q4: %10.5f \
		# 			Q4_err: %10.5f Q4C: %10.5f Q4C_err: %10.5f R: %10.5f \
		# 			R_err: %10.5f" % (beta, self.t0[beta],
		# 			self.topc2[atype][beta]["y"][t0_index], 
		# 			self.topc2[atype][beta]["y_error"][t0_index],
		# 			self.topc4[atype][beta]["y"][t0_index], 
		# 			self.topc4[atype][beta]["y_error"][t0_index],
		# 			self.topc4C[atype][beta]["y"][t0_index], 
		# 			self.topc4C[atype][beta]["y_error"][t0_index],
		# 			self.topcR[atype][beta]["y"][t0_index], 
		# 			self.topcR[atype][beta]["y_error"][t0_index]))

		# 	print "\nArticle data(normalized by volume):"
		# 	for data_set in sorted(article_data2[size]):

		# 		print "Dataset: %s Beta: %2.2f Volume: %f t0: %s" % (
		# 			data_set, self.article_name_size[data_set][size]["beta"],
		# 			self.article_name_size[data_set][size]["V"],	self.t0)
		# 		print "Q2:  %10.5f Q2_err:  %10.5f" % (
		# 			self.article_name_size[data_set][size]["Q2_norm"],
		# 			self.article_name_size[data_set][size]["Q2Err_norm"])
		# 		print "Q4:  %10.5f Q4_err:  %10.5f" % (
		# 			self.article_name_size[data_set][size]["Q4_norm"],
		# 			self.article_name_size[data_set][size]["Q4Err_norm"])
		# 		print "Q4C: %10.5f Q4C_err: %10.5f" % (
		# 			self.article_name_size[data_set][size]["Q4C_norm"],
		# 			self.article_name_size[data_set][size]["Q4CErr_norm"])
		# 		print "R:   %10.5f R_err:   %10.5f" % (
		# 			self.article_name_size[data_set][size]["R_norm"],
		# 			self.article_name_size[data_set][size]["RErr_norm"])

		# 		if size==1:
		# 			x_pvals_article.append(self.t0)
		# 			y_pvals_article.append((
		# 				self.article_name_size[data_set][size]["R_norm"],
		# 				self.article_name_size[data_set][size]["RErr_norm"]))

			print "\nRatios between me and article"
			for data_set in sorted(self.article_size_name[size]):
				# Compares values by dividing my values by article values
				for beta in self.beta_values:
					beta_article = self.article_name_size[data_set][size]["beta"]

					# Gets the approximate same t0 ref. value
					t0_index = np.argmin(
						np.abs(self.topc2[atype][beta]["x"] - self.t0[beta]))
					print "Beta(me) %.2f Beta(article) %.2f Dataset %-s" % (
						beta, beta_article, data_set)

					print "Q2_me/Q2_article:   %10.5f +/- %10.5f" % (
						ratio_error(self.topc2[atype][beta]["y"][t0_index],
						self.topc2[atype][beta]["y_error"][t0_index],
						self.article_name_size[data_set][size]["Q2_norm"], 
						self.article_name_size[data_set][size]["Q2Err_norm"]))
					print "Q4_me/Q4_article:   %10.5f +/- %10.5f" % (
						ratio_error(self.topc4[atype][beta]["y"][t0_index], 
						self.topc4[atype][beta]["y_error"][t0_index],
						self.article_name_size[data_set][size]["Q4_norm"], 
						self.article_name_size[data_set][size]["Q4Err_norm"]))
					print "Q4C_me/Q4C_article: %10.5f +/- %10.5f" % (
						ratio_error(self.topc4C[atype][beta]["y"][t0_index], 
						self.topc4C[atype][beta]["y_error"][t0_index],
						self.article_name_size[data_set][size]["Q4C_norm"], 
						self.article_name_size[data_set][size]["Q4CErr_norm"]))
					print "R_me/R_article:     %10.5f +/- %10.5f" % (
						ratio_error(self.topcR[atype][beta]["y"][t0_index], 
						self.topcR[atype][beta]["y_error"][t0_index],
						self.article_name_size[data_set][size]["R_norm"], 
						self.article_name_size[data_set][size]["RErr_norm"]))
			print ""

		exit("Exits before exiting compare_lattice_values @ 565")

	def _normalize_article_values(self):
		"""
		Normalizes values from article based on physical volume.
		"""
		for data_set in self.article_name_size:
			for size in self.article_name_size[data_set]:
				# Set up volume in physical units
				L = self.article_name_size[data_set][size]["L"]
				a = self.article_name_size[data_set][size]["a"]
				aL = a*float(L)
				V = aL**4
				# self._add_article_dict_item(name, size, key, value)
				self._add_article_dict_item(data_set, size, "aL", aL)
				self._add_article_dict_item(data_set, size, "V", V)

				# Normalize Q^2 by V
				Q2 = self.article_name_size[data_set][size]["Q2"]
				Q2Err = self.article_name_size[data_set][size]["Q2Err"]
				Q2_norm = Q2/V
				Q2Err_norm = Q2Err/V
				self._add_article_dict_item(data_set, size, "Q2_norm", Q2_norm)
				self._add_article_dict_item(data_set, size, "Q2Err_norm", Q2Err_norm)

				# Normalize Q^4 by V
				Q4 = self.article_name_size[data_set][size]["Q4"]
				Q4Err = self.article_name_size[data_set][size]["Q4Err"]
				Q4_norm = Q4/V**2
				Q4Err_norm = Q4Err/V**2
				self._add_article_dict_item(data_set, size, "Q4_norm", Q4_norm)
				self._add_article_dict_item(data_set, size, "Q4Err_norm", Q4Err_norm)

				# Recalculates 4th cumulant
				Q4C_norm = self.Q4C(Q4_norm, Q2_norm)
				Q4CErr_norm = self.Q4C_error(Q4_norm, Q4Err_norm, Q2_norm, 
					Q2Err_norm)
				self._add_article_dict_item(data_set, size, "Q4C_norm", Q4C_norm)
				self._add_article_dict_item(data_set, size, "Q4CErr_norm", Q4CErr_norm)

				# Recalculates R
				R_norm = self.R(Q4C_norm, Q2_norm)
				RErr_norm = self.R_error(Q4C_norm, Q4CErr_norm, Q2_norm, 
					Q2Err_norm)
				self._add_article_dict_item(data_set, size, "R_norm", R_norm)
				self._add_article_dict_item(data_set, size, "RErr_norm", RErr_norm)

		# for data_set in sorted(self.article_name_size):
		# 	for size in sorted(self.article_name_size[data_set]):
		# 		print "="*50
		# 		print "Dataset: %s Size number: %s Volume: %f" % (
		# 			data_set, size, self.article_name_size[data_set][size]["V"])
		# 		print "Q2: %10.5f %10.5f" % (
		# 			self.article_name_size[data_set][size]["Q2_norm"],
		# 			self.article_name_size[data_set][size]["Q2Err_norm"])
		# 		print "Q4: %10.5f %10.5f" % (
		# 			self.article_name_size[data_set][size]["Q4_norm"],
		# 			self.article_name_size[data_set][size]["Q4Err_norm"])
		# 		print "Q4C: %10.5f %10.5f" % (
		# 			self.article_name_size[data_set][size]["Q4C_norm"],
		# 			self.article_name_size[data_set][size]["Q4CErr_norm"])
		# 		print "R: %10.5f %10.5f" % (
		# 			self.article_name_size[data_set][size]["R_norm"],
		# 			self.article_name_size[data_set][size]["RErr_norm"])

	def _add_article_dict_item(self, name, size, key, value):
		"""Small method for adding item to article data dictionaries."""
		self.article_size_name[size][name][key] = value
		self.article_name_size[name][size][key] = value
		flat_key = "{0:s}_{1:d}".format(name, size)
		self.article_flattened[flat_key][key] = value

	def _setup_article_values(self):
		"""
		Sets up the article values from https://arxiv.org/abs/1506.06052

		Format:
			{Lattice type}/{Beta value}/{all other stuff}
		"""

		self.article_name_size = {
			"A": 
			{
				1: {
					"beta": 5.96,
					"t0cont": 2.79, # t0/a^2
					"t0": 2.995, # t0/a^2
					"t0err": 0.004, # t0/a^2
					"t0r02": 0.1195,
					"t0r02err": 0.0009,
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
					"t0cont": 2.79, # t0/a^2
					"t0": 2.7984,
					"t0err":0.0009,
					"t0r02": 0.1117,
					"t0r02err": 0.0009,
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
					"t0cont": 3.78, # t0/a^2
					"t0": 3.7960,
					"t0err": 0.0012,
					"t0r02": 0.1114,
					"t0r02err": 0.0009,
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
					"t0cont": 4.87, # t0/a^2
					"t0": 4.8855,
					"t0err": 0.0015,
					"t0r02": 0.1113,
					"t0r02err": 0.0010,
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
					"t0cont": 6.20, # t0/a^2
					"t0": 6.2191,
					"t0err": 0.0020,
					"t0r02": 0.1115,
					"t0r02err": 0.0011,
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
					"t0cont": 2.79, # t0/a^2
					"t0": 2.7908,
					"t0err":0.0005,
					"t0r02": 0.1114,
					"t0r02err": 0.0009,
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
					"t0cont": 2.79, # t0/a^2
					"t0": 2.7889,
					"t0err":0.0003,
					"t0r02": 0.1113,
					"t0r02err": 0.0009,
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
					"t0cont": 3.78, # t0/a^2
					"t0": 3.7825,
					"t0err": 0.0008,
					"t0r02": 0.1110,
					"t0r02err": 0.0009,
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
					"t0cont": 4.87, # t0/a^2
					"t0": 4.8722,
					"t0err": 0.0011,
					"t0r02": 0.1110,
					"t0r02err": 0.0010,
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
					"t0cont": 6.20, # t0/a^2
					"t0": 6.1957,
					"t0err": 0.0014,
					"t0r02": 0.1111,
					"t0r02err": 0.0011,
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
					"t0cont": 2.79, # t0/a^2
					"t0": 2.78892,
					"t0err":0.00023,
					"t0r02": 0.1113,
					"t0r02err": 0.0009,
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
					"t0cont": 2.79, # t0/a^2
					"t0": 2.78867,
					"t0err":0.00016,
					"t0r02": 0.1113,
					"t0r02err": 0.0009,
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

		self.article_size_name = {}
		for data_set in sorted(self.article_name_size):
			for size in sorted(self.article_name_size[data_set]):
				self.article_size_name[size] = {}

		for data_set in sorted(self.article_name_size):
			for size in sorted(self.article_name_size[data_set]):
				self.article_size_name[size][data_set] = \
					self.article_name_size[data_set][size]

		# Usefull when printing out values. The two first layers of 
		# dictionaries are merged.
		self.article_flattened = {}
		for data_set in sorted(self.article_name_size):
			for size in sorted(self.article_name_size[data_set]):
				name = "{0:s}_{1:d}".format(data_set, size)
				self.article_flattened[name] = \
					self.article_name_size[data_set][size]

		self.article_flattened = OrderedDict(sorted(
			self.article_flattened.items(),
			key=lambda k: [k[0].split("_")[1], k[0].split("_")[0]]))


def main():
	exit("Exit: TopcRPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()