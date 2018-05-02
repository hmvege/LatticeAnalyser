from postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from tools.latticefunctions import witten_veneziano
from statistics.linefit import LineFit
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

	def plot_continuum(self, fit_target, title_addendum=""):
		"""
		Method for plotting the continuum limit of topsus at a given 
		fit_target.

		Args:
			fit_target: float, value of where we extrapolate from.
			title_addendum: str, optional, default is an empty string, ''. 
				Adds string to end of title.
		"""
		if fit_target == -1:
			fit_target = self.plot_values[max(self.plot_values)]["x"][-1]

		a_squared, obs, obs_err = [], [], []
		for beta in sorted(self.plot_values):
			x = self.plot_values[beta]["x"]
			y = self.plot_values[beta]["y"]
			y_err = self.plot_values[beta]["y_err"]

			fit_index = np.argmin(np.abs(x - fit_target))

			a_squared.append(self.plot_values[beta]["a"]**2/fit_target)
			obs.append(y[fit_index])
			obs_err.append(y_err[fit_index])

		# Initiates empty arrays for the continuum limit
		a_squared = np.asarray(a_squared)[::-1]
		obs = np.asarray(obs)[::-1]
		obs_err = np.asarray(obs_err)[::-1]

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

		title_string = r"$\sqrt{8t_{flow,0}} = %.2f[fm], \chi^2 = %.2g$" % (
			fit_target, chi_squared)
		title_string += title_addendum
		
		# Gets the continium value and its error
		cont_index = np.argmin(np.abs(a_squared_cont))
		a0_squared = [a_squared_cont[cont_index], a_squared_cont[cont_index]]
		y0 = [y_cont[cont_index], y_cont[cont_index]]

		if y_cont_err[1][cont_index] < y_cont_err[0][cont_index]:
			y0_err_lower = y0[0] - y_cont_err[1][cont_index]
			y0_err_upper = y_cont_err[0][cont_index] - y0[0]
		else:
			y0_err_lower = y0[0] - y_cont_err[0][cont_index]
			y0_err_upper = y_cont_err[1][cont_index] - y0[0]

		# Stores the chi continuum
		self.topsus_continuum = y_cont[cont_index]
		self.topsus_continuum_error = [y0_err_lower, y0_err_upper]

		y0_err = [[y0_err_lower, 0], [y0_err_upper, 0]]

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
			label=r"$\chi^{1/4}=%.3f\pm%.3f$" % (y0[0],
				(y0_err_lower + y0_err_upper)/2.0))

		ax.set_ylabel(self.y_label_continuum)
		ax.set_xlabel(self.x_label_continuum)
		ax.set_title(title_string)
		ax.set_xlim(0, a_squared[-1]*1.1)
		ax.legend()
		ax.grid(True)

		if self.verbose:
			print "Target: %.16f +/- %.16f" % (y0[0],
				(y0_err_lower + y0_err_upper)/2.0)

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
			self.topsus_continuum_error[0], self.NF, self.NF_error, \
			self.fit_target, self.interval

	def print_continuum_estimate(self):
		"""Prints the NF from the Witten-Veneziano formula."""
		self.NF, self.NF_error = witten_veneziano(self.topsus_continuum, 
			self.topsus_continuum_error[0])
		msg =  "\n    Topsus = %.16f" % self.topsus_continuum
		msg += "\n    Topsus_error = %.16f" % self.topsus_continuum_error[0]
		msg += "\n    N_f = %.16f" % self.NF
		msg += "\n    N_f_error = %.16f" % self.NF_error
		msg += "\n    Chi^2 = %.16f" % self.chi_squared
		msg += "\n"
		print msg 

def main():
	exit("Exit: TopsusCore not intended to be a standalone module.")

if __name__ == '__main__':
	main()