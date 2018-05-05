from postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from tools.latticefunctions import witten_veneziano
from statistics.linefit import LineFit, extract_fit_target
import matplotlib.pyplot as plt
import numpy as np
import os

class TopsusCore(PostCore):
	observable_name = "topsus_core"
	observable_name_compact = "topsus_core"

	# Regular plot variables
	x_label = r"$\sqrt{8t_{flow}}[fm]$"

	# Continuum plot variables
	y_label_continuum = r"$\chi^{1/4}[GeV]$"
	# x_label_continuum = r"$a/{{r_0}^2}$"
	x_label_continuum = r"$a^2/t_0$"

	def plot_continuum(self, fit_target, title_addendum="",
		extrapolation_method="platou", use_raw_values=False, 
		platou_fit_size=20, interpolation_rank=3):
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
					- platou: line fits points neighbouring point in order to 
						reduce the error bars.
					- nearest: line fit from the point nearest to what we seek
					- interpolate: linear interpolation in order to retrieve 
						value and error. Does not work in conjecture with 
						use_raw_values.
			use_raw_values: bool, optional, if true, will use bootstrap, 
				jackknifed or unanalyzed samples directly. Default is False.
			platou_size: int, optional. Number of points in positive and 
				negative direction to extrapolate fit target value from. This 
				value also applies to the interpolation interval. Default is 20.
			interpolation_rank: int, optional. Interpolation rank to use if 
				extrapolation method is interpolation Default is 3.
		"""

		#################################
		#### Continuum extrapolation ####
		#################################
		# Retrieves data for analysis.
		if fit_target == -1:
			fit_target = self.plot_values[max(self.plot_values)]["x"][-1]

		a_squared, obs, obs_raw, obs_err, tau_int_corr = [], [], [], [], []
		for beta in sorted(self.plot_values):
			x = self.plot_values[beta]["x"]
			y = self.plot_values[beta]["y"]
			y_err = self.plot_values[beta]["y_err"]
			y_raw = self.plot_values[beta][self.analysis_data_type]
			if self.with_autocorr:
				tau_int = self.plot_values[beta]["tau_int"]
				tau_int_err = self.plot_values[beta]["tau_int_err"]
			else:
				tau_int = None
				tau_int_err = None

			res = extract_fit_target(fit_target, x, y, y_err, y_raw=y_raw,
				tau_int=tau_int, tau_int_err=tau_int_err, 
				extrapolation_method="bootstrap", use_raw_values=True, 
				platou_size=20, interpolation_rank=3, plot_fit=True,
				verbose=self.verbose)

			# THIS IS WHERE WE IMPLEMENT DIFFERENT METHODS!!!
			fit_index = np.argmin(np.abs(x - fit_target))
			a_squared.append(self.plot_values[beta]["a"]**2/fit_target)
			obs.append(y[fit_index])
			obs_err.append(y_err[fit_index])
			obs_raw.append(y_raw[fit_index])
			tau_int_corr.append(tau_int[fit_index])


		# Makes lists into arrays
		a_squared = np.asarray(a_squared)[::-1]
		obs = np.asarray(obs)[::-1]
		obs_err = np.asarray(obs_err)[::-1]
		#################################
		#################################
		#################################

		# Continuum limit arrays
		N_cont = 1000
		a_squared_cont = np.linspace(0, a_squared[-1]*1.1, N_cont)

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
		y0_err = [y0_err, y0_err]

		# Stores the chi continuum
		self.topsus_continuum = y0[0]
		self.topsus_continuum_error = (y0_err[0][1] - y0_err[0][0])/2.0

		# Sets of title string with the chi squared and fit target
		title_string = r"$\sqrt{8t_{flow,0}} = %.2f[fm], \chi^2 = %.2g$" % (
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
		ax.errorbar(a0_squared, y0, yerr=y0_err, fmt="o", capsize=None,
			capthick=1, color="tab:red", ecolor="tab:red",
			label=r"$\chi^{1/4}=%.3f\pm%.3f$" % (self.topsus_continuum,
				self.topsus_continuum_error))

		ax.set_ylabel(self.y_label_continuum)
		ax.set_xlabel(self.x_label_continuum)
		ax.set_title(title_string)
		ax.set_xlim(0, a_squared[-1]*1.1)
		ax.legend()
		ax.grid(True)

		if self.verbose:
			print "Target: %.16f +/- %.16f" % (self.topsus_continuum,
				self.topsus_continuum_error)

		# Saves figure
		fname = os.path.join(self.output_folder_path, 
			"post_analysis_%s_continuum%s_%s.png" % (self.observable_name_compact,
				str(fit_target).replace(".",""), self.analysis_data_type))
		fig.savefig(fname, dpi=self.dpi)

		if self.verbose:
			print "Continuum plot of %s created in %s" % (
				self.observable_name.lower(), fname)
		plt.close(fig)

		self.print_continuum_estimate()

	def get_linefit_parameters(self):
		"""Returns the chi^2, a, a_err, b, b_err."""
		return self.chi_squared, self.fit_params, self.topsus_continuum, \
			self.topsus_continuum_error, self.NF, self.NF_error, \
			self.fit_target, self.interval

	def print_continuum_estimate(self):
		"""Prints the NF from the Witten-Veneziano formula."""
		self.NF, self.NF_error = witten_veneziano(self.topsus_continuum, 
			self.topsus_continuum_error)
		msg =  "\n    Topsus = %.16f" % self.topsus_continuum
		msg += "\n    Topsus_error = %.16f" % self.topsus_continuum_error
		msg += "\n    N_f = %.16f" % self.NF
		msg += "\n    N_f_error = %.16f" % self.NF_error
		msg += "\n    Chi^2 = %.16f" % self.chi_squared
		msg += "\n"
		print msg 

def main():
	exit("Exit: TopsusCore not intended to be a standalone module.")

if __name__ == '__main__':
	main()