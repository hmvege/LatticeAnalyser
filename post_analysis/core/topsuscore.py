from postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from tools.latticefunctions import witten_veneziano
from statistics.linefit import LineFit, extract_fit_target
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import types

class TopsusCore(PostCore):
	observable_name = "topsus_core"
	observable_name_compact = "topsus_core"

	# Regular plot variables
	x_label = r"$\sqrt{8t_{flow}}[fm]$"

	# Continuum plot variables
	y_label_continuum = r"$\chi^{1/4}[GeV]$"
	# x_label_continuum = r"$a/{{r_0}^2}$"
	x_label_continuum = r"$a^2/t_0$"

	# For specialized observables
	extra_continuum_msg = ""

	# For topsus function
	hbarc = 0.19732697 #eV micro m

	chi_const = {}
	chi = {}
	chi_der = {}

	# For description in printing the different parameters from fit 
	descr = ""

	# Description for the extrapolation method in fit
	extrapolation_method = ""

	# String for intervals
	intervals_str = ""

	def __init__(self, *args, **kwargs):
		super(TopsusCore, self).__init__(*args, **kwargs)
		self._initialize_topsus_func()

	def _initialize_topsus_func_const(self):
		"""Sets the constant in the topsus function for found beta values."""
		for beta in self.beta_values:
			V = self.lattice_sizes[beta][0]**3 * self.lattice_sizes[beta][1]
			self.chi_const[beta] = self.hbarc/get_lattice_spacing(beta)[0]\
				/float(V)**(0.25)
			# self.chi[beta] = lambda qq: self.chi_const[beta]*qq**(0.25)

	def _initialize_topsus_func(self):
		"""Sets the topsus function for all found beta values."""

		self._initialize_topsus_func_const()

		# Bad hardcoding due to functions stored in a dictionary in a loop is
		# not possible.
		if 6.0 in self.beta_values:
			self.chi[6.0] = lambda qq: self.chi_const[6.0]*qq**(0.25)
			self.chi_der[6.0] = lambda qq, qqerr: \
				0.25*self.chi_const[6.0]*qqerr/qq**(0.75)

		if 6.1 in self.beta_values:
			self.chi[6.1] = lambda qq: self.chi_const[6.1]*qq**(0.25)
			self.chi_der[6.1] = lambda qq, qqerr: \
				0.25*self.chi_const[6.1]*qqerr/qq**(0.75)

		if 6.2 in self.beta_values:
			self.chi[6.2] = lambda qq: self.chi_const[6.2]*qq**(0.25)
			self.chi_der[6.2] = lambda qq, qqerr: \
				0.25*self.chi_const[6.2]*qqerr/qq**(0.75)

		if 6.45 in self.beta_values:
			self.chi[6.45] = lambda qq: self.chi_const[6.45]*qq**(0.25)
			self.chi_der[6.45] = lambda qq, qqerr: \
				0.25*self.chi_const[6.45]*qqerr/qq**(0.75)

	def plot_continuum(self, fit_target, title_addendum="",
		extrapolation_method="plateau", plateau_fit_size=20,
		interpolation_rank=3, plot_continuum_fit=False):
		"""
		Method for plotting the continuum limit of topsus at a given 
		fit_target.

		Args:
			fit_target: float, value of where we extrapolate from.
			title_addendum: str, optional, default is an empty string, ''. 
				Adds string to end of title.
			extrapolation_method: str, optional, method of selecting the 
				extrapolation point to do the continuum limit. Method will be
				used on y values and tau int. Choices:
					- plateau: line fits points neighbouring point in order to 
						reduce the error bars using y_raw for covariance matrix.
					- plateau_mean: line fits points neighbouring point in order to
						reduce the error bars. Line will be weighted by the y_err.
					- nearest: line fit from the point nearest to what we seek
					- interpolate: linear interpolation in order to retrieve value
						and error. Does not work in conjecture with use_raw_values.
					- bootstrap: will create multiple line fits, and take average. 
						Assumes y_raw is the bootstrapped or jackknifed samples.
			plateau_size: int, optional. Number of points in positive and 
				negative direction to extrapolate fit target value from. This 
				value also applies to the interpolation interval. Default is 20.
			interpolation_rank: int, optional. Interpolation rank to use if 
				extrapolation method is interpolation Default is 3.
			raw_func: function, optional, will modify the bootstrap data after 
				samples has been taken by this function.
			raw_func_err: function, optional, will propagate the error of the 
				bootstrapped line fitted data, raw_func_err(y, yerr). Calculated
				by regular error propagation.
		"""
		self.extrapolation_method = extrapolation_method
		if not isinstance(self.reference_values, types.NoneType):
			if extrapolation_method in self.reference_values.keys():
				# Checking that the extrapolation method selected can be used.
				t0_values = self.reference_values[self.extrapolation_method]\
					[self.analysis_data_type]
			else:
				# If the extrapolation method is not among the used methods.
				t0_values = self.reference_values.values()[0]\
					[self.analysis_data_type]
		else:
			t0_values = None

		# Retrieves data for analysis.
		if fit_target == -1:
			fit_target = self.plot_values[max(self.plot_values)]["x"][-1]

		a_squared, obs, obs_raw, obs_err, tau_int_corr = [], [], [], [], []
		for beta in sorted(self.plot_values):
			x = self.plot_values[beta]["x"]
			y = self.plot_values[beta]["y"]
			y_err = self.plot_values[beta]["y_err"]
			y_raw = self.plot_values[beta]["y_raw"]

			if self.with_autocorr:
				tau_int = self.plot_values[beta]["tau_int"]
				tau_int_err = self.plot_values[beta]["tau_int_err"]
			else:
				tau_int = None
				tau_int_err = None

			# Extrapolation of point to use in continuum extrapolation
			res = extract_fit_target(fit_target, x, y, y_err, y_raw=y_raw,
				tau_int=tau_int, tau_int_err=tau_int_err, 
				extrapolation_method=extrapolation_method, 
				plateau_size=plateau_fit_size, interpolation_rank=3, 
				plot_fit=plot_continuum_fit, raw_func=self.chi[beta],
				raw_func_err=self.chi_der[beta], plot_samples=False, 
				verbose=False)

			_x0, _y0, _y0_error, _y0_raw, _tau_int0 = res
			
			if self.verbose:
				msg = "Beta = %4.2f Topsus = %14.12f +/- %14.12f" % (
					beta, _y0, _y0_error)
			
			if isinstance(t0_values, types.NoneType):
				a_squared.append(self.plot_values[beta]["a"]**2/_x0)
			else:
				a_squared.append(
					self.plot_values[beta]["a"]**2/t0_values["t0_cont"])

				if self.verbose:
					msg += " t0 = %14.12f" % (t0_values["t0_cont"]\
						/ (self.plot_values[beta]["a"]**2))

			if self.verbose:
				print msg

			obs.append(_y0)
			obs_err.append(_y0_error)
			obs_raw.append(_y0_raw)
			tau_int_corr.append(_tau_int0)

		# Makes lists into arrays
		a_squared = np.asarray(a_squared)[::-1]
		obs = np.asarray(obs)[::-1]
		obs_err = np.asarray(obs_err)[::-1]

		# Continuum limit arrays
		N_cont = 1000
		a_squared_cont = np.linspace(-0.0025, a_squared[-1]*1.1, N_cont)

		# Fits to continuum and retrieves values to be plotted
		continuum_fit = LineFit(a_squared, obs, obs_err)

		y_cont, y_cont_err, fit_params, chi_squared = \
			continuum_fit.fit_weighted(a_squared_cont)
		self.chi_squared = chi_squared
		self.fit_params = fit_params
		self.fit_target = fit_target

		# continuum_fit.plot(True)

		# Gets the continium value and its error
		y0_cont, y0_cont_err, _, _, = \
			continuum_fit.fit_weighted(0.0)

		# Matplotlib requires 2 point to plot error bars at
		a0_squared = [0, 0]
		y0 = [y0_cont[0], y0_cont[0]]
		y0_err = [y0_cont_err[0][0], y0_cont_err[1][0]]

		# Stores the chi continuum
		self.topsus_continuum = y0[0]
		self.topsus_continuum_error = (y0_err[1] - y0_err[0])/2.0

		y0_err = [self.topsus_continuum_error, self.topsus_continuum_error]

		# Sets of title string with the chi squared and fit target
		title_string = r"$t_{f,0} = %.2f[fm], \chi^2 = %.2f$" % (
			self.fit_target, self.chi_squared)
		title_string += title_addendum

		# Creates figure and plot window
		fig = plt.figure()
		ax = fig.add_subplot(111)

		# Plots linefit with errorband
		ax.plot(a_squared_cont, y_cont, color="tab:blue", alpha=0.5)
		ax.fill_between(a_squared_cont, y_cont_err[0], y_cont_err[1],
			alpha=0.5, edgecolor='', facecolor="tab:blue")

		# Plot lattice points
		ax.errorbar(a_squared, obs, yerr=obs_err, fmt="o",
			color="tab:orange", ecolor="tab:orange")

		# plots continuum limit, 5 is a good value for cap size
		ax.errorbar(a0_squared, y0,
			yerr=y0_err, fmt="o", capsize=None,
			capthick=1, color="tab:red", ecolor="tab:red",
			label=r"$\chi^{1/4}=%.3f\pm%.3f$" % (self.topsus_continuum,
				self.topsus_continuum_error))

		ax.set_ylabel(self.y_label_continuum)
		ax.set_xlabel(self.x_label_continuum)
		ax.set_title(title_string)
		ax.set_xlim(a_squared_cont[0], a_squared_cont[-1])
		ax.legend()
		ax.grid(True)

		if self.verbose:
			print "Target: %.16f +/- %.16f" % (self.topsus_continuum,
				self.topsus_continuum_error)

		# Saves figure
		fname = os.path.join(self.output_folder_path, 
			"post_analysis_extrapmethod%s_%s_continuum%s_%s.png" % (
				extrapolation_method, self.observable_name_compact,
				str(fit_target).replace(".",""), self.analysis_data_type))
		fig.savefig(fname, dpi=self.dpi)

		if self.verbose:
			print "Continuum plot of %s created in %s" % (
				self.observable_name.lower(), fname)

		# plt.show()
		plt.close(fig)

		self.print_continuum_estimate()

	def get_linefit_parameters(self):
		"""Returns the chi^2, a, a_err, b, b_err."""
		return self.chi_squared, self.fit_params, self.topsus_continuum, \
			self.topsus_continuum_error, self.NF, self.NF_error, \
			self.fit_target, self.intervals_str, self.descr, \
			self.extrapolation_method

	def print_continuum_estimate(self):
		"""Prints the NF from the Witten-Veneziano formula."""
		self.NF, self.NF_error = witten_veneziano(self.topsus_continuum, 
			self.topsus_continuum_error)
		msg = "Observable: %s" % self.observable_name_compact
		msg += "\n    Fit target: %.4f" % self.fit_target
		msg += "\n    Topsus = %.16f" % self.topsus_continuum
		msg += "\n    Topsus_error = %.16f" % self.topsus_continuum_error
		msg += "\n    N_f = %.16f" % self.NF
		msg += "\n    N_f_error = %.16f" % self.NF_error
		msg += "\n    Chi^2 = %.16f" % self.chi_squared
		msg += self.extra_continuum_msg
		msg += "\n"
		print msg 

def main():
	exit("Exit: TopsusCore not intended to be a standalone module.")

if __name__ == '__main__':
	main()