from post_analysis.core.postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from tools.table_printer import TablePrinter
import tools.sciprint as sciprint
from statistics.linefit import LineFit, extract_fit_target
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
import numpy as np
import os
import copy

class EnergyPostAnalysis(PostCore):
	"""Post analysis of the energy, <E>."""
	observable_name = "Energy"
	observable_name_compact = "energy"

	# Regular plot variables
	y_label = r"$t_f^2\langle E\rangle$"
	x_label = r"$t_f/r_0^2$"
	formula = r"$\langle E\rangle = -\frac{1}{64|\Lambda|}F_{\mu\nu}^a{F^a}^{\mu\nu}$"

	# Continuum plot variables
	x_label_continuum = r"$(a/r_0)^2$"
	y_label_continuum = r"$\frac{\sqrt{8t_{f,0}}}{r_0}$"

	@staticmethod
	def derivative(y, eps):
		"""
		First order derivative of O(n^2).

		Args:
			y, array, to derivate.
			eps, integration step length.
		Returns:
			y derivative.
		"""
		y_der = np.zeros(len(y) - 2)
		for i in xrange(1, len(y) - 1):
			y_der[i-1] = (y[i+1] - y[i-1]) / (2*eps)

		return y_der

	def calculateW(self, x, y, y_err, y_raw, feps, t_unscaled):
		"""
		Calculates the W(t) used in the scale setting definition given in
		http://xxx.lanl.gov/pdf/1203.4469v2
		"""
		t = t_unscaled[1:-1]
		W = np.zeros(len(x) - 2)
		W_err = np.zeros(len(x) - 2)
		W_raw = np.zeros((len(x) - 2, y_raw.shape[-1]))
		dE_raw = np.zeros((len(x) - 2, y_raw.shape[-1]))

		dE = self.derivative(y, feps)
		for iBoot in xrange(W_raw.shape[-1]):
			dE_raw[:,iBoot] = self.derivative(y_raw[:,iBoot], feps)

		# dE_err = np.zeros(len(x) - 2)
		# for i in xrange(1, len(y) - 1):
		# 	dE_err[i-1] = (y_err[+1] - y_err[-1])/(2*feps)

		# Calculates W(t) = t d/dt { t^2 E(t) }
		for i in xrange(1, len(y) - 1):
			W[i-1] = 2*t[i-1]**2*y[i] + t[i-1]**3*dE[i-1]
			W_raw[i-1] = 2*t[i-1]**2*y_raw[i] + t[i-1]**3*dE_raw[i-1]

		# Uncertainty propagation in derivative: 
		# https://physics.stackexchange.com/questions/200029/how-does-uncertainty-error-propagate-with-differentiation
		for i in xrange(1, len(y) - 1):
			# W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2)# + (t[i-1]**3*np.sqrt(2)*y_err[i]/feps)**2)
			# print np.std(W_raw[i-1])
			W_err[i-1] = np.std(W_raw[i-1])
			# W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2 + (t[i-1]**3*np.std(W_raw[i-1]))**2)
			# W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2)
			# W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2 + (y_err[i+1]/(2*feps))**2 + (y_err[i-1]/(2*feps))**2 )

		# # Sets up the x-axis value for t_f*a^2
		# ta2 = np.zeros(len(x) - 2)
		# for i in xrange(1, len(y) - 1):
		# 	ta2[i-1] = t_unscaled[i] 

		# W = np.mean(W_raw, axis=1)
		# plt.plot(x[1:-1], W, color="tab:red", alpha=0.5)
		# plt.fill_between(x[1:-1], W-W_err, W+W_err, alpha=0.5, edgecolor='',
		# 	facecolor="tab:red")
		# plt.grid(True)
		# plt.hlines(0.3,x[1], x[-2], linestyle=":", alpha=0.75, color="gray")
		# plt.xlabel(r"$t_f/{r_0^2}$")
		# plt.ylabel(r"$W(t_f)$")
		# plt.show()
		# exit(1)

		return t_unscaled, W, W_err, W_raw

	def _initiate_plot_values(self, data, data_raw):
		# Sorts data into a format specific for the plotting method
		for beta in sorted(data.keys()):
			values = {}
			values["beta"] = beta
			values["a"], values["a_err"] = get_lattice_spacing(beta)
			values["t"] = data[beta]["x"]*values["a"]**2
			values["sqrt8t"] = values["a"]*np.sqrt(8*data[beta]["x"])
			values["x"] = values["t"]/self.r0**2
			values["y"] = data[beta]["y"]*data[beta]["x"]**2
			values["y_err"] = data[beta]["y_error"]*data[beta]["x"]**2
			values["flow_epsilon"] = self.flow_epsilon[beta]
			values["tder"], values["W"], values["W_err"], values["W_raw"] = \
				self.calculateW(values["t"], data[beta]["y"], 
					data[beta]["y_error"], 
					data_raw[beta][self.observable_name_compact], 
					values["flow_epsilon"], data[beta]["x"])
			# print values["x"]
			# exit("Exits in plot values initiations.")

			if self.with_autocorr:
				values["tau_int"] = data[beta]["ac"]["tau_int"]
				values["tau_int_err"] = data[beta]["ac"]["tau_int_err"]
			else:
				values["tau_int"] = None
				values["tau_int_err"] = None

			values[self.analysis_data_type] = \
				(data_raw[beta][self.observable_name_compact].T \
					*(data[beta]["x"]**2)).T

			values["label"] = (r"%s $\beta=%2.2f$" %
				(self.size_labels[beta], beta))

			self.plot_values[beta] = values

	def get_t0_scale(self, extrapolation_method="plateau_mean", E0=0.3, **kwargs):
		"""
		Method for retrieveing reference value t0 based on Luscher(2010),
		Properties and uses of the Wilson flow in lattice QCD.
		t^2<E_t>|_{t=t_0} = 0.3
		Will return t0 values and make a plot of the continuum value 
		extrapolation.

		Args:
			extrapolation_method: str, optional. Method of t0 extraction. 
				Default is plateau_mean.
			E0: float, optional. Default is 0.3.

		Returns:
			t0: dictionary of t0 values for each of the betas, and a continuum
				value extrapolation.
		"""
		if self.verbose:
			print "Scale t0 extraction method:      " + extrapolation_method
			print "Scale t0 extraction data:        " + self.analysis_data_type

		# Retrieves t0 values from data
		a_values = []
		a_values_err = []
		t0_values = []
		t0err_values = []

		for beta, bval in sorted(self.plot_values.items(), key=lambda i: i[0]):
			y0, t0, t0_err, _, _ = extract_fit_target(E0, bval["t"], bval["y"],
				y_err=bval["y_err"], y_raw=bval[self.analysis_data_type], 
				tau_int=bval["tau_int"], tau_int_err=bval["tau_int_err"],
				extrapolation_method=extrapolation_method, plateau_size=10,
				inverse_fit=True, **kwargs)

			a_values.append(bval["a"]**2/t0)
			a_values_err.append(np.sqrt((2*bval["a_err"]*bval["a"]/t0)**2 \
				+ (bval["a"]**2*t0_err/t0**2)**2))

			t0_values.append(t0)
			t0err_values.append(t0_err)

		a_values = np.asarray(a_values[::-1])
		a_values_err = np.asarray(a_values_err[::-1])
		t0_values = np.asarray(t0_values[::-1])
		t0err_values = np.asarray(t0err_values[::-1])		

		# Functions for t0 and propagating uncertainty
		t0_func = lambda _t0: np.sqrt(8*_t0)/self.r0
		t0err_func = lambda _t0, _t0_err: _t0_err*np.sqrt(8/_t0)/(2.0*self.r0)

		# Sets up t0 and t0_error values to plot
		y = t0_func(t0_values)
		yerr = t0err_func(t0_values, t0err_values)

		# Extrapolates t0 to continuum
		N_cont = 1000
		a_squared_cont = np.linspace(-0.025, a_values[-1]*1.1, N_cont)

		# Fits to continuum and retrieves values to be plotted
		continuum_fit = LineFit(a_values, y, y_err=yerr)
		y_cont, y_cont_err, fit_params, chi_squared = \
			continuum_fit.fit_weighted(a_squared_cont)

		res = continuum_fit(0, weighted=True)
		self.sqrt_8t0_cont = res[0][0]
		self.sqrt_8t0_cont_error = (res[1][-1][0] - res[1][0][0])/2
		self.t0_cont = self.sqrt_8t0_cont**2/8
		self.t0_cont_error = self.sqrt_8t0_cont_error*np.sqrt(self.t0_cont/2.0)

		# Creates figure and plot window
		fig = plt.figure()
		ax = fig.add_subplot(111)
		# Plots linefit with errorband
		ax.plot(a_squared_cont, y_cont, color="tab:red", alpha=0.5,
			label=r"$\chi=%.2f$" % chi_squared)
		ax.fill_between(a_squared_cont, y_cont_err[0], 
			y_cont_err[1], alpha=0.5, edgecolor='',
			facecolor="tab:red")
		ax.axvline(0, linestyle="dashed", color="tab:red")
		ax.errorbar(a_values, y, xerr=a_values_err, yerr=yerr, fmt="o", capsize=5,
			capthick=1, color="#000000", ecolor="#000000")
		ax.set_ylabel(r"$\sqrt{8t_0}/r_0$")
		ax.set_xlabel(r"$a^2/t_0$")
		ax.set_xlim(a_squared_cont[0], a_squared_cont[-1])
		ax.legend()
		ax.grid(True)

		# Saves figure
		fname = os.path.join(self.output_folder_path, 
			"post_analysis_extrapmethod%s_t0reference_continuum_%s.pdf" % (
				extrapolation_method, self.analysis_data_type))
		fig.savefig(fname, dpi=self.dpi)
		if self.verbose:
			print "Figure saved in %s" % fname

		plt.close(fig)

		self.extrapolation_method = extrapolation_method

		_tmp_beta_dict = {
			b: {
				"t0": t0_values[i],
				"t0err": t0err_values[i],
				"t0a2": t0_values[i]/self.plot_values[b]["a"]**2,
				# Including error term in lattice spacing, a
				"t0a2err": np.sqrt((t0err_values[i]/self.plot_values[b]["a"]**2)**2 \
					+ (2*self.plot_values[b]["a_err"]*t0_values[i]/self.plot_values[b]["a"]**3)**2),
				"t0r02": t0_values[i]/self.r0**2,
				"t0r02err": t0err_values[i]/self.r0**2,
				"aL": self.plot_values[b]["a"]*self.lattice_sizes[b][0],
				"aLerr": (self.plot_values[b]["a_err"] \
					* self.lattice_sizes[b][0]),
				"L": self.lattice_sizes[b][0],
				"a": self.plot_values[beta]["a"],
				"a_err": self.plot_values[b]["a_err"],
			}
			for i, b in enumerate(self.beta_values)
		}

		t0_dict = {"t0cont": self.t0_cont, "t0cont_err": self.t0_cont_error}
		t0_dict.update(_tmp_beta_dict)

		if self.verbose:
			print "t0 reference values table: "
			print "sqrt(8t0)/r0 = %.16f +/- %.16f" % (self.sqrt_8t0_cont,
				self.sqrt_8t0_cont_error)
			print "t0 = %.16f +/- %.16f" % (self.t0_cont,
				self.t0_cont_error)
			for b in self.beta_values:
				msg = "beta = %.2f || t0 = %10f +/- %-10f" % (b, 
					t0_dict[b]["t0"], t0_dict[b]["t0err"])
				msg += " || t0/a^2 = %10f +/- %-10f" % (t0_dict[b]["t0a2"], 
					t0_dict[b]["t0a2err"])
				msg += " || t0/a^2 = %10f +/- %-10f" % (t0_dict[b]["t0a2"], 
					t0_dict[b]["t0a2err"])
				msg += " || t0/r0^2 = %10f +/- %-10f" % (t0_dict[b]["t0r02"],
					t0_dict[b]["t0r02err"])
				print msg

		if self.print_latex:
			# Header:
			# beta   t0a2   t0r02   L/a   L   a

			header = [r"$\beta$", r"$t_0/a^2$", r"$t_0/{r_0^2}$", r"$L/a$",
				r"$L[\fm]$", r"$a[\fm]$"]

			bvals = self.beta_values
			tab = [
				[r"{0:.2f}".format(b) for b in bvals],
				[r"{0:s}".format(sciprint.sciprint(t0_dict[b]["t0a2"], 
					t0_dict[b]["t0a2err"])) for b in bvals],
				[r"{0:s}".format(sciprint.sciprint(t0_dict[b]["t0r02"], 
					t0_dict[b]["t0r02err"])) for b in bvals],
				[r"{0:d}".format(self.lattice_sizes[b][0]) for b in bvals],
				[r"{0:s}".format(sciprint.sciprint(
					self.lattice_sizes[b][0]*self.plot_values[b]["a"], 
					self.lattice_sizes[b][0]*self.plot_values[b]["a_err"])) 
					for b in bvals],
				[r"{0:s}".format(sciprint.sciprint(
					self.plot_values[b]["a"],
					self.plot_values[b]["a_err"])) for b in bvals],
			]

			ptab = TablePrinter(header, tab)			
			ptab.print_table(latex=True, width=15)

		return t0_dict


	def get_w0_scale(self, extrapolation_method="plateau_mean", W0=0.3, 
		**kwargs):
		"""
		Method for retrieving the w0 reference scale setting, based on paper:
		http://xxx.lanl.gov/pdf/1203.4469v2
		"""
		if self.verbose:
			print "Scale w0 extraction method:      " + extrapolation_method
			print "Scale w0 extraction data:        " + self.analysis_data_type

		# Retrieves t0 values from data
		a_values = []
		a_values_err = []
		w0_values = []
		w0err_values = []

		for beta, bval in sorted(self.plot_values.items(), key=lambda i: i[0]):
			y0, w0, w0_err, _, _ = extract_fit_target(W0, bval["tder"], bval["W"],
				y_err=bval["W_err"], y_raw=bval["W_raw"], 
				tau_int=bval["tau_int"][1:-1], 
				tau_int_err=bval["tau_int_err"][1:-1],
				extrapolation_method=extrapolation_method, plateau_size=10,
				inverse_fit=True, **kwargs)

			# TODO: fix a lattice spacing error here @ energy
			a_values.append(bval["a"]**2)
			a_values_err.append(2*bval["a_err"]*bval["a"])
			w0_values.append(np.sqrt(w0)*bval["a"])
			w0err_values.append(np.sqrt((0.5*w0_err/np.sqrt(w0))**2 \
				+ (np.sqrt(w0)*bval["a_err"])**2))

		a_values = np.asarray(a_values[::-1])
		a_values_err = np.asarray(a_values_err[::-1])
		w0_values = np.asarray(w0_values[::-1])
		w0err_values = np.asarray(w0err_values[::-1])

		# # Functions for t0 and propagating uncertainty
		# t0_func = lambda _t0: np.sqrt(8*_t0)/self.r0
		# t0err_func = lambda _t0, _t0_err: _t0_err*np.sqrt(8/_t0)/(2.0*self.r0)


		# # Sets up t0 and t0_error values to plot
		# y = t0_func(t0_values)
		# yerr = t0err_func(t0_values, t0err_values)

		# Extrapolates t0 to continuum
		N_cont = 1000
		a_squared_cont = np.linspace(-0.00025, a_values[-1]*1.1, N_cont)

		# Fits to continuum and retrieves values to be plotted
		continuum_fit = LineFit(a_values, w0_values, y_err=w0err_values)
		y_cont, y_cont_err, fit_params, chi_squared = \
			continuum_fit.fit_weighted(a_squared_cont)

		res = continuum_fit(0, weighted=True)
		self.w0_cont = res[0][0]
		self.w0_cont_error = (res[1][-1][0] - res[1][0][0])/2
		# self.t0_cont = self.sqrt_8t0_cont**2/8
		# self.t0_cont_error = self.sqrt_8t0_cont_error*np.sqrt(self.t0_cont/2.0)

		# Creates figure and plot window
		fig = plt.figure()
		ax = fig.add_subplot(111)
		# Plots linefit with errorband
		ax.plot(a_squared_cont, y_cont, color="tab:red", alpha=0.5,
			label=r"$\chi=%.2f$" % chi_squared)
		ax.fill_between(a_squared_cont, y_cont_err[0], 
			y_cont_err[1], alpha=0.5, edgecolor='',
			facecolor="tab:red")
		ax.axvline(0, linestyle="dashed", color="tab:red")
		ax.errorbar(a_values, w0_values, xerr=a_values_err, yerr=w0err_values, fmt="o", capsize=5,
			capthick=1, color="#000000", ecolor="#000000")
		ax.set_ylabel(r"$w_0[\mathrm{fm}]$")
		ax.set_xlabel(r"$a^2[\mathrm{GeV}^{-2}]$")
		ax.set_xlim(a_squared_cont[0], a_squared_cont[-1])
		ax.legend()
		ax.grid(True)

		# Saves figure
		fname = os.path.join(self.output_folder_path, 
			"post_analysis_extrapmethod%s_w0reference_continuum_%s.pdf" % (
				extrapolation_method, self.analysis_data_type))
		fig.savefig(fname, dpi=self.dpi)
		if self.verbose:
			print "Figure saved in %s" % fname

		plt.close(fig)

		w0_values = w0_values[::-1]
		w0err_values = w0err_values[::-1]

		_tmp_beta_dict = {
			b: {
				"w0": w0_values[i],
				"w0err": w0err_values[i],
				"aL": self.plot_values[b]["a"]*self.lattice_sizes[b][0],
				"aLerr": (self.plot_values[b]["a_err"] \
					* self.lattice_sizes[b][0]),
				"L": self.lattice_sizes[b][0],
				"a": self.plot_values[beta]["a"],
				"a_err": self.plot_values[b]["a_err"],
			}
			for i, b in enumerate(self.beta_values)
		}

		w0_dict = {"w0cont": self.w0_cont, "w0cont_err": self.w0_cont_error}
		w0_dict.update(_tmp_beta_dict)

		if self.verbose:
			print "w0 reference values table: "
			print "w0 = %.16f +/- %.16f" % (self.w0_cont, self.w0_cont_error)
			for b in self.beta_values:
				msg = "beta = %.2f || w0 = %10f +/- %-10f" % (b, 
					w0_dict[b]["w0"], w0_dict[b]["w0err"])
				print msg

		if self.print_latex:
			# Header:
			# beta  w0  a^2  L/a  L  a

			header = [r"$\beta$", r"$w_0[\fm]$", 
				r"$a^2[\mathrm{GeV}^{-2}]$", r"$L/a$", r"$L[\fm]$", 
				r"$a[\fm]$"]

			bvals = self.beta_values
			tab = [
				[r"{0:.2f}".format(b) for b in bvals],
				[r"{0:s}".format(sciprint.sciprint(w0_dict[b]["w0"], 
					w0_dict[b]["w0err"])) for b in bvals],
				[r"{0:s}".format(sciprint.sciprint(self.plot_values[b]["a"]**2,
					self.plot_values[b]["a_err"]*2*self.plot_values[b]["a"])) 
					for b in bvals],
				[r"{0:d}".format(self.lattice_sizes[b][0]) for b in bvals],
				[r"{0:s}".format(sciprint.sciprint(
					self.lattice_sizes[b][0]*self.plot_values[b]["a"], 
					self.lattice_sizes[b][0]*self.plot_values[b]["a_err"])) 
					for b in bvals],
				[r"{0:s}".format(sciprint.sciprint(
					self.plot_values[b]["a"],
					self.plot_values[b]["a_err"])) for b in bvals],
			]

			ptab = TablePrinter(header, tab)			
			ptab.print_table(latex=True, width=15)

	# def _energy_continuum(self, t):
	# 	"""
	# 	Second order approximation of the energy.
	# 	"""
	# 	coupling = self._coupling_alpha(t)
	# 	k1 = 1.0978
	# 	mean_E = 3.0/(4.0*np.pi*t**2) * coupling * (1 + k1*coupling)
	# 	mean_E_error = 3.0/(4.0*np.pi*t**2)*coupling**3
	# 	return mean_E, mean_E_error

	# @staticmethod
	# def _coupling_alpha(t):
	# 	"""
	# 	Running coupling constant.
	# 	"""
	# 	q = 1.0/np.sqrt(8*t)
	# 	beta0 = 11.0
	# 	LAMBDA = 0.34 # [GeV]
	# 	alpha = 4*np.pi / (beta0 * np.log(q/LAMBDA**2))
	# 	return alpha

	# def _get_fit_interval(self, fit_target, fit_interval, y_mean):
	# 	"""Function for finding the fit interval."""
	# 	start_index = np.argmin(np.abs(y_mean - (fit_target - fit_interval)))
	# 	end_index = np.argmin(np.abs(y_mean - (fit_target + fit_interval)))
	# 	return start_index, end_index

	# def _inverse_beta_fit(self, fit_target, fit_interval):
	# 	"""
	# 	Perform an inverse fit on the observable susceptibility and extracts 
	# 	extracting x0 with x0 error.
	# 	"""
	# 	for beta in sorted(self.plot_values):
	# 		x = self.plot_values[beta]["x"]
	# 		y = self.plot_values[beta]["y"]
	# 		y_err = self.plot_values[beta]["y_err"]

	# 		index_low, index_high = self._get_fit_interval(fit_target, fit_interval, x)

	# 		x = x[index_low:index_high]
	# 		y = y[index_low:index_high]
	# 		y_err = y_err[index_low:index_high]

	# 		fit = LineFit(x, y, y_err)
	# 		y_hat, y_hat_err, fit_params, chi_squared = fit.fit_weighted()
	# 		b0, b0_err, b1, b1_err = fit_params

	# 		self.plot_values[beta]["fit"] = {
	# 			"y_hat": y_hat,
	# 			"y_hat_err": y_hat_err,
	# 			"b0": b0,
	# 			"b0_err": b0_err,
	# 			"b1": b1,
	# 			"b1_err": b1_err,
	# 			"chi_squared": chi_squared,
	# 		}

	# 		x0, x0_err = fit.inverse_fit(fit_target, weighted=True)

	# 		fit.plot(True)

	# 		self.plot_values[beta]["fit"]["inverse"] = {
	# 				"x0": x0,
	# 				"x0_err": x0_err, 
	# 		}


	# def _linefit_to_continuum(self, x_points, y_points, y_points_error,
	# 	fit_type="least_squares"):
	# 	"""
	# 	Fits a a set of values to continuum.
	# 	Args:
	# 		x_points (numpy float array) : x points to data fit
	# 		y_points (numpy float array) : y points to data fit
	# 		y_points_error (numpy float array) : error of y points to data fit
	# 		[optional] fit_type (str) : type of fit to perform.
	# 			Options: 'curve_fit' (default), 'polyfit'
	# 	"""

	# 	# Fitting data
	# 	if fit_type == "least_squares":
	# 		pol, polcov = sciopt.curve_fit(lambda x, a, b: x*a + b, x_points,
	# 			y_points, sigma=y_points_error)
	# 	elif fit_type == "polynomial":
	# 		pol, polcov = np.polyfit(x_points, y_points, 1, rcond=None, 
	# 			full=False, w=1.0/y_points_error, cov=True)
	# 	else:
	# 		raise KeyError("fit_type '%s' not recognized." % fit_type)

	# 	# Gets line properties
	# 	a = pol[0]
	# 	b = pol[1]
	# 	a_err, b_err = np.sqrt(np.diag(polcov))

	# 	# Sets up line fitted variables		
	# 	x = np.linspace(0, x_points[-1]*1.03, 1000)
	# 	y = a * x + b
	# 	y_std = a_err*x + b_err

	# 	return x, y, y_std, a, b, a_err, b_err

	# def plot_continuum(self, fit_target, fit_interval, fit_type, 
	# 	plot_arrows=[0.05, 0.07, 0.1], legend_location="best",
	# 	line_fit_type="least_squares"):
	# 	"""
	# 	Continuum plotter for the energy.
	# 	"""

	# 	# Retrieves t0 values used to be used for continium fitting
	# 	self._get_beta_to_fit(
	# 		fit_target, fit_interval, axis="y",
	# 		fit_type=fit_type, 
	# 		fit_function_modifier=lambda x: x*self.r0**2,
	# 		plot_fit_window=False)

	# 	a_lattice_spacings = \
	# 		np.asarray([val["a"] for val in self.beta_fit])[::-1]
	# 	t_fit_points = np.asarray([val["t0"] for val in self.beta_fit])[::-1]
	# 	t_fit_points_errors = \
	# 		np.asarray([val["t0_err"] for val in self.beta_fit])[::-1]

	# 	# Initiates empty arrays for
	# 	x_points = np.zeros(len(a_lattice_spacings) + 1)
	# 	y_points = np.zeros(len(a_lattice_spacings) + 1)
	# 	y_points_err = np.zeros(len(a_lattice_spacings) + 1)

	# 	# Populates with fit data
	# 	x_points[1:] = (a_lattice_spacings / self.r0)**2 
	# 	y_points[1:] = np.sqrt(8*(t_fit_points)) / self.r0
	# 	y_points_err[1:] = \
	# 		(8*t_fit_points_errors) / (np.sqrt(8*t_fit_points)) / self.r0

	# 	# Fits to continuum and retrieves values to be plotted
	# 	x_line, y_line, y_line_std, a, b, a_err, b_err = \
	# 		self._linefit_to_continuum(x_points[1:], y_points[1:], 
	# 			y_points_err[1:], fit_type=line_fit_type)

	# 	# Populates arrays with first fitted element
	# 	x_points[0] = x_line[0]
	# 	y_points[0] = y_line[0]
	# 	y_points_err[0] = y_line_std[0]

	# 	# Creates figure and plot window
	# 	fig = plt.figure(self.dpi)
	# 	ax = fig.add_subplot(111)

	# 	# ax.axvline(0,linestyle="--",color="0",alpha=0.5)

	# 	ax.errorbar(x_points[1:], y_points[1:], yerr=y_points_err[1:],
	# 		fmt="o", color="0", ecolor="0")
	# 	ax.errorbar(x_points[0], y_points[0], yerr=y_points_err[0], fmt="o",
	# 		capthick=4, color="r", ecolor="r")
	# 	ax.plot(x_line, y_line, color="0", 
	# 		label=r"$y=(%.3f\pm%.3f)x + %.4f\pm%.4f$" % (a, a_err, b, b_err))
	# 	ax.fill_between(x_line, y_line-y_line_std, y_line + y_line_std, 
	# 		alpha=0.2, edgecolor='', facecolor="0")
	# 	ax.set_ylabel(self.y_label_continuum)
	# 	ax.set_xlabel(self.x_label_continuum)

	# 	ax.set_title((r"Continuum limit reference scale: $t_{0,cont}=%2.4f\pm%g$"
	# 		% ((self.r0*y_points[0])**2/8,(self.r0*y_points_err[0])**2/8)))

	# 	ax.set_xlim(-0.005, 0.045)
	# 	ax.set_ylim(0.92, 0.98)
	# 	ax.legend()
	# 	ax.grid(True)

	# 	# Fixes axis tick intervals
	# 	start, end = ax.get_ylim()
	# 	ax.yaxis.set_ticks(np.arange(start, end, 0.01))

	# 	# Puts on some arrows at relevant points
	# 	for arrow in plot_arrows:
	# 		ax.annotate(r"$a=%.2g$fm" % arrow, xy=((arrow/self.r0)**2, end), 
	# 			xytext=((arrow/self.r0)**2, end+0.005), 
	# 			arrowprops=dict(arrowstyle="->"), ha="center")
		
	# 	ax.legend(loc=legend_location) # "lower left"

	# 	# Saves figure
	# 	fname = os.path.join(self.output_folder_path, 
	# 		("post_analysis_%s_continuum%s_%s.pdf" % 
	# 			(self.observable_name_compact, str(fit_target).replace(".",""),
	# 				fit_type.strip("_"))))

	# 	fig.savefig(fname, dpi=self.dpi)

	# 	print ("Continuum plot of %s created in %s" 
	# 		% (self.observable_name.lower(), fname))

	# 	plt.close(fig)

	# def coupling_fit(self):
	# 	print "Finding Lambda"

	# 	pass

	def plot_w(self, *args, **kwargs):
		"""Plots the W(t)."""
		w_plot_values = copy.deepcopy(self.plot_values)
		for beta in sorted(self.beta_values):
			w_plot_values[beta]["x"] = self.plot_values[beta]["x"][1:-1]
			w_plot_values[beta]["y"] = self.plot_values[beta]["W"]
			w_plot_values[beta]["y_err"] = self.plot_values[beta]["W_err"]

		kwargs["observable_name_compact"] = "energyW"
		kwargs["x_label"] = r"$t_f/r_0^2$"
		kwargs["y_label"] = r"$W(t)$"
		# kwargs["show_plot"] = True

		self._plot_core(w_plot_values, *args, **kwargs)

	def __str__(self):
		"""Class string representation method."""
		msg = "\n" + self.section_seperator
		msg += "\nPost analaysis for:        " + self.observable_name_compact
		msg += "\n" + self.__doc__
		msg += "\nAnalysis-type:             " + self.analysis_data_type
		# msg += "\nE0 extraction method:      " + self.extrapolation_method
		msg += "\nIncluding autocorrelation: " + self.ac
		msg += "\nOutput folder:             " + self.output_folder_path
		msg += "\n" + self.section_seperator
		return msg

def main():
	exit("Exit: EnergyPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()