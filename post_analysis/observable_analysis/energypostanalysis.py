from post_analysis.core.postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from statistics.linefit import LineFit
import matplotlib.pyplot as plt
import numpy as np
import os

class EnergyPostAnalysis(PostCore):
	"""Post analysis of the energy, <E>."""
	observable_name = "Energy"
	observable_name_compact = "energy"

	# Regular plot variables
	y_label = r"$t^2\langle E\rangle$"
	x_label = r"$t/r_0^2$"
	formula = r"$\langle E\rangle = -\frac{1}{64V}F_{\mu\nu}^a{F^a}^{\mu\nu}$"

	# Continuum plot variables
	x_label_continuum = r"$(a/r_0)^2$"
	y_label_continuum = r"$\frac{\sqrt{8t_0}}{r_0}$"

	def _get_fit_interval(self, fit_target, fit_interval, y_mean):
		"""Function for finding the fit interval."""
		start_index = np.argmin(np.abs(y_mean - (fit_target - fit_interval)))
		end_index = np.argmin(np.abs(y_mean - (fit_target + fit_interval)))
		return start_index, end_index

	def _inverse_beta_fit(self, fit_target, fit_interval):
		"""
		Perform an inverse fit on the observable susceptibility and extracts 
		extracting x0 with x0 error.
		"""
		for beta in sorted(self.plot_values):
			x = self.plot_values[beta]["x"]
			y = self.plot_values[beta]["y"]
			y_err = self.plot_values[beta]["y_err"]

			index_low, index_high = self._get_fit_interval(fit_target, fit_interval, x)

			x = x[index_low:index_high]
			y = y[index_low:index_high]
			y_err = y_err[index_low:index_high]

			fit = LineFit(x, y, y_err)
			y_hat, y_hat_err, fit_params, chi_squared = fit.fit_weighted()
			b0, b0_err, b1, b1_err = fit_params

			self.plot_values[beta]["fit"] = {
				"y_hat": y_hat,
				"y_hat_err": y_hat_err,
				"b0": b0,
				"b0_err": b0_err,
				"b1": b1,
				"b1_err": b1_err,
				"chi_squared": chi_squared,
			}

			x0, x0_err = fit.inverse_fit(fit_target, weigthed=True)

			fit.plot(True)

			self.plot_values[beta]["fit"]["inverse"] = {
					"x0": x0,
					"x0_err": x0_err, 
			}

	def _energy_continuum(self, t):
		"""
		Second order approximation of the energy.
		"""
		coupling = self._coupling_alpha(t)
		k1 = 1.0978
		mean_E = 3.0/(4.0*np.pi*t**2) * coupling * (1 + k1*coupling)
		mean_E_error = 3.0/(4.0*np.pi*t**2)*coupling**3
		return mean_E, mean_E_error

	@staticmethod
	def _coupling_alpha(t):
		"""
		Running coupling constant.
		"""
		q = 1.0/np.sqrt(8*t)
		beta0 = 11.0
		LAMBDA = 0.34 # [GeV]
		alpha = 4*np.pi / (beta0 * np.log(q/LAMBDA**2))
		return alpha

	def _initiate_plot_values(self, data, data_raw):
		# Sorts data into a format specific for the plotting method
		for beta in sorted(data.keys()):
			values = {}
			values["beta"] = beta
			values["a"] = get_lattice_spacing(beta)

			values["x"] = data[beta]["x"]/self.r0**2*values["a"]**2
			values["y"] = data[beta]["y"]*data[beta]["x"]**2
			values["y_err"] = data[beta]["y_error"]*data[beta]["x"]**2

			if self.with_autocorr:
				values["tau_int"] = data[beta]["ac"]["tau_int"]
				values["tau_int_err"] = data[beta]["ac"]["tau_int_err"]

			values[self.analysis_data_type] = \
				data_raw[beta][self.observable_name_compact]

			values["label"] = (r"%s $\beta=%2.2f$" %
				(self.size_labels[beta], beta))

			self.plot_values[beta] = values

	def _linefit_to_continuum(self, x_points, y_points, y_points_error,
		fit_type="least_squares"):
		"""
		Fits a a set of values to continuum.
		Args:
			x_points (numpy float array) : x points to data fit
			y_points (numpy float array) : y points to data fit
			y_points_error (numpy float array) : error of y points to data fit
			[optional] fit_type (str) : type of fit to perform.
				Options: 'curve_fit' (default), 'polyfit'
		"""

		# Fitting data
		if fit_type == "least_squares":
			pol, polcov = sciopt.curve_fit(lambda x, a, b: x*a + b, x_points,
				y_points, sigma=y_points_error)
		elif fit_type == "polynomial":
			pol, polcov = np.polyfit(x_points, y_points, 1, rcond=None, 
				full=False, w=1.0/y_points_error, cov=True)
		else:
			raise KeyError("fit_type '%s' not recognized." % fit_type)

		# Gets line properties
		a = pol[0]
		b = pol[1]
		a_err, b_err = np.sqrt(np.diag(polcov))

		# Sets up line fitted variables		
		x = np.linspace(0, x_points[-1]*1.03, 1000)
		y = a * x + b
		y_std = a_err*x + b_err

		return x, y, y_std, a, b, a_err, b_err

	def plot_continuum(self, fit_target, fit_interval, fit_type, 
			plot_arrows=[0.05, 0.07, 0.1], legend_location="best",
			line_fit_type="least_squares"):
		# Retrieves t0 values used to be used for continium fitting
		self._get_beta_values_to_fit(
			fit_target, fit_interval, axis="y",
			fit_type=fit_type, 
			fit_function_modifier=lambda x: x*self.r0**2,
			plot_fit_window=False)

		a_lattice_spacings = \
			np.asarray([val["a"] for val in self.beta_fit])[::-1]
		t_fit_points = np.asarray([val["t0"] for val in self.beta_fit])[::-1]
		t_fit_points_errors = \
			np.asarray([val["t0_err"] for val in self.beta_fit])[::-1]

		# Initiates empty arrays for
		x_points = np.zeros(len(a_lattice_spacings) + 1)
		y_points = np.zeros(len(a_lattice_spacings) + 1)
		y_points_err = np.zeros(len(a_lattice_spacings) + 1)

		# Populates with fit data
		x_points[1:] = (a_lattice_spacings / self.r0)**2 
		y_points[1:] = np.sqrt(8*(t_fit_points)) / self.r0
		y_points_err[1:] = \
			(8*t_fit_points_errors) / (np.sqrt(8*t_fit_points)) / self.r0

		# Fits to continuum and retrieves values to be plotted
		x_line, y_line, y_line_std, a, b, a_err, b_err = \
			self._linefit_to_continuum(x_points[1:], y_points[1:], 
				y_points_err[1:], fit_type=line_fit_type)

		# Populates arrays with first fitted element
		x_points[0] = x_line[0]
		y_points[0] = y_line[0]
		y_points_err[0] = y_line_std[0]

		# Creates figure and plot window
		fig = plt.figure(self.dpi)
		ax = fig.add_subplot(111)

		# ax.axvline(0,linestyle="--",color="0",alpha=0.5)

		ax.errorbar(x_points[1:], y_points[1:], yerr=y_points_err[1:],
			fmt="o", color="0", ecolor="0")
		ax.errorbar(x_points[0], y_points[0], yerr=y_points_err[0], fmt="o",
			capthick=4, color="r", ecolor="r")
		ax.plot(x_line, y_line, color="0", 
			label=r"$y=(%.3f\pm%.3f)x + %.4f\pm%.4f$" % (a, a_err, b, b_err))
		ax.fill_between(x_line, y_line-y_line_std, y_line + y_line_std, 
			alpha=0.2, edgecolor='', facecolor="0")
		ax.set_ylabel(self.y_label_continuum)
		ax.set_xlabel(self.x_label_continuum)

		ax.set_title((r"Continuum limit reference scale: $t_{0,cont}=%2.4f\pm%g$"
			% ((self.r0*y_points[0])**2/8,(self.r0*y_points_err[0])**2/8)))

		ax.set_xlim(-0.005, 0.045)
		ax.set_ylim(0.92, 0.98)
		ax.legend()
		ax.grid(True)

		# Fixes axis tick intervals
		start, end = ax.get_ylim()
		ax.yaxis.set_ticks(np.arange(start, end, 0.01))

		# Puts on some arrows at relevant points
		for arrow in plot_arrows:
			ax.annotate(r"$a=%.2g$fm" % arrow, xy=((arrow/self.r0)**2, end), 
				xytext=((arrow/self.r0)**2, end+0.005), 
				arrowprops=dict(arrowstyle="->"), ha="center")
		
		ax.legend(loc=legend_location) # "lower left"

		# Saves figure
		fname = os.path.join(self.output_folder_path, 
			("post_analysis_%s_continuum%s_%s.png" % 
				(self.observable_name_compact, str(fit_target).replace(".",""),
					fit_type.strip("_"))))

		fig.savefig(fname, dpi=self.dpi)

		print ("Continuum plot of %s created in %s" 
			% (self.observable_name.lower(), fname))

		plt.close(fig)

	def coupling_fit(self):
		print "Finding Lambda"

		pass

def main():
	exit("Exit: EnergyPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()