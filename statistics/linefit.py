#!/usr/bin/env python2

from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.optimize as sciopt
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import types
from tqdm import tqdm

__all__ = ["LineFit", "extract_fit_target"]

class LineFit:
	"""
	Line fit based on article by Lavagini et al (2007).
	"""
	def __init__(self, x, y, y_err=None):
		assert len(x) == len(y), "array lengths to not match"
		self.x = x
		self.y = y
		self.y_err = y_err
		self.n = len(y)
		self.x_lower = np.min(self.x)
		self.x_upper = np.max(self.x)

		# Sets weigths regardless if they are to be used or not.
		if not isinstance(y_err, types.NoneType):
			self.w = self._get_weights(self.y_err)

		self.inverse_fit_performed = False

	def fit(self, x_arr=None):
		"""
		Fits a linear function,
			y_hat = b0 + b1 * x
		from a provided dataset, x, y and y_err from initialization.

		Args:
			x_arr: optional array for which return as a fit. Default is None.

		Returns:
			y_hat: returns fit(x)
			y_hat_err: returns fit_error(x)
			fit_params: list containing b0, b0_err, b1, b1_err
			chi_squared: goodness of fit
		"""

		if isinstance(x_arr, types.NoneType):
			x_arr = np.linspace(self.x[self.x_lower], self.x[self.x_upper], 100)
		else:
			x_arr = np.atleast_1d(x_arr)

		# self.x_mean = np.mean(self.x)
		# self.y_mean = np.mean(self.y)

		# # Temporary sum contained in both eq. 4, eq. 6 and eq. 7
		# self.xi_xmean_sum = np.sum((self.x - self.x_mean)**2)
		
		self.x_mean, self.y_mean, self.xi_xmean_sum = self._get_means()

		# self.s_xy_err = np.sum((self.y - self._y_hat(self.x))**2) / (self.n - 2)

		# Eq. 4
		self.b1 = np.sum((self.x - self.x_mean) * self.y) / self.xi_xmean_sum

		# Eq. 3
		self.b0 = self.y_mean - self.b1 * self.x_mean

		# Eq. 5
		self.s_xy_err = self._get_s_xy()

		# Eq. 6
		self.b0_err = self.s_xy_err
		self.b0_err *= (1.0 / self.n + self.x_mean ** 2 / self.xi_xmean_sum)

		# Eq. 7
		self.b1_err = self.s_xy_err / self.xi_xmean_sum

		fit_params = [self.b0, self.b0_err, self.b1, self.b1_err]

		# Goodness of fit
		if not isinstance(self.y_err, types.NoneType):
			self.chi = self.chi_squared(self.y, self.y_err, self._y_hat(self.x))
		else:
			self.chi = None

		self.x_fit = x_arr
		self.y_fit = self._y_hat(x_arr)
		self.y_fit_err = self._y_hat_err(x_arr)

		return self.y_fit, self.y_fit_err, fit_params, self.chi

	def _y_hat(self, x):		
		"""Unweigthed y(x), eq. 1."""
		return self.b0 + self.b1 * x

	def _y_hat_err(self, x):
		"""Unweigthed y(x) error, eq. 8b."""
		_pt1 = self.b0 + self.b1 * x
		_pt2 = scipy.stats.t.isf(0.32, self.n - 2) * np.sqrt(self.s_xy_err) 
		_pt2 *= np.sqrt(1.0 / self.n + (x - self.x_mean)**2 / self.xi_xmean_sum)
		return [_pt1 - _pt2, _pt1 + _pt2]

	def fit_weighted(self, x_arr=None):
		"""
		Performs a weighted fit based on x, y, and y_err provided in 
		initialization.

		Args:
			x_arr: optional array for which return as a fit. Default is None.

		Returns:
			y_hat: returns fit(x)
			y_hat_err: returns fit_error(x)
			fit_params: list containing b0, b0_err, b1, b1_err
			chi_squared: goodness of fit
		"""

		if isinstance(x_arr, types.NoneType):
			x_arr = np.linspace(self.x[0], self.x[-1], 100)
		else:
			x_arr = np.atleast_1d(x_arr)

		assert not isinstance(self.y_err, types.NoneType), "Missing y_err."

		# self.xw_mean = np.sum(self.w * self.x) / np.sum(self.w)
		# self.yw_mean = np.sum(self.w * self.y) / np.sum(self.w)

		# # Temporary som contained in both eq. 4, eq. 6 and eq. 7
		# self.xwi_xmean_sum = np.sum(self.w * (self.x - self.xw_mean)**2)
		self.xw_mean, self.yw_mean, self.xwi_xmean_sum = self._get_means_weigthed()


		# Eq. 18
		self.b1w = np.sum(self.w * (self.x - self.xw_mean) * self.y)
		self.b1w /= self.xwi_xmean_sum

		# Eq. 17
		self.b0w = self.yw_mean - self.b1w * self.xw_mean

		# # Eq. 21
		# self.s_xyw_err = np.sum(self.w * (self.y - self._yw_hat(self.x))**2)
		# self.s_xyw_err /= (self.n - 2.0) 
		self.s_xyw_err = self._get_s_xyw()

		# Eq. 19
		self.b0w_err = (1.0/np.sum(self.w) + self.xw_mean**2 / self.xwi_xmean_sum) 
		self.b0w_err *= self.s_xyw_err

		# Eq. 20
		self.b1w_err = self.s_xyw_err / self.xwi_xmean_sum

		fit_params = [self.b0w, self.b0w_err, self.b1w, self.b1w_err]

		# Goodness of fit
		self.chi_w = self.chi_squared(self.y, self.y_err, self._yw_hat(self.x))

		# Stores variables for later possible use
		self.xw_fit = x_arr
		self.yw_fit = self._yw_hat(x_arr)
		self.yw_fit_err = self._yw_hat_err(x_arr)

		return self.yw_fit, self.yw_fit_err, fit_params, self.chi_w

	def _yw_hat(self, x):
		"""weigthed y(x), eq. 1"""
		return self.b0w + self.b1w * x

	def _yw_hat_err(self, x):
		"""Weigthed y(x) errors, eq. 22."""
		_pt1 = self.b0w + self.b1w * x
		_pt2 = scipy.stats.t.isf(1.0 - 0.32/2.0, self.n - 2) * np.sqrt(self.s_xyw_err) 
		# _pt2 = scipy.stats.t.isf(0.32, self.n - 2) * np.sqrt(self.s_xyw_err) 
		_pt2 *= np.sqrt(1.0 / np.sum(self.w) + \
			(x - self.xw_mean)**2 / self.xwi_xmean_sum)

		return [_pt1 - np.abs(_pt2), _pt1 + np.abs(_pt2)]

	def _get_s_xy(self):
		"""Eq. 5."""
		return np.sum((self.y - self._y_hat(self.x))**2) / (self.n - 2)

	def _get_s_xyw(self):
		"""Eq. 21."""
		s_xyw_err = np.sum(self.w * (self.y - self._yw_hat(self.x))**2)
		return s_xyw_err/(self.n - 2.0)

	def _get_means(self):
		""" Returns non-weigthed means."""
		# Eq. 4, eq. 6 and eq. 7
		x_mean = np.mean(self.x)
		xi_xmean_sum = np.sum((self.x - x_mean)**2)
		return x_mean, np.mean(self.y), xi_xmean_sum

	def _get_means_weigthed(self):
		"""Sets weighted means."""
		xw_mean = np.sum(self.w * self.x) / np.sum(self.w)
		yw_mean = np.sum(self.w * self.y) / np.sum(self.w)

		# Temporary som contained in both eq. 4, eq. 6 and eq. 7
		xwi_xmean_sum = np.sum(self.w * (self.x - xw_mean)**2)
		return xw_mean, yw_mean, xwi_xmean_sum

	def _get_weights(self, y_err):
		"""Sets weights based on error in fit."""
		return 1.0/y_err**2

	def set_fit_parameters(self, b0, b0_err, b1, b1_err, weighted=False):
		"""
		Method for setting fit parameters if they have been retrieved by 
		another method, such as scipy.optimize.curve_fit.

		Args:
			b0: constant term in line fit.
			b0: error of constant term in line fit.
			b1: slope term in line fit.
			b1: error of slope term in line fit.
		"""
		if weighted:
			self.b0w = b0
			self.b0w_err = b0_err
			self.b1w = b1
			self.b1w_err = b1_err
			self.xw_mean, self.yw_mean, self.xwi_xmean_sum \
				= self._get_means_weigthed()
			self.s_xyw_err = self._get_s_xyw()
		else:
			self.b0 = b0
			self.b0_err = b0_err
			self.b1 = b1
			self.b1_err = b1_err

			self.x_mean, self.y_mean, self.xi_xmean_sum \
				= self._get_means()
			self.s_xy_err = self._get_s_xy()

	def __call__(self, x, weighted=False):
		"""Returns the fitted function at x."""
		x = np.atleast_1d(x)
		if weighted:
			y_fit, y_fit_err = self._yw_hat(x), self._yw_hat_err(x)
			return y_fit, y_fit_err, self.chi_squared(self.y, self.y_err, 
				self._yw_hat(self.x))
		else:
			y_fit, y_fit_err = self._y_hat(x), self._y_hat_err(x)
			return y_fit, y_fit_err


	def inverse_fit(self, y0, weigthed=False):
		"""
		Inverse fiting on the values we have performed a fit one.

		Args:
			y0: target fit at y-axis, float.
			weigthed: bool, if we are to use weighted fit or not.

		Returns:
			x0: targeted y0 fit
			x0_err: errorband at y0 fit
		"""
		n = 100000
		x = np.linspace(self.x_lower, self.x_upper, n)

		if weigthed:
			# Finds the target value
			x0_index = np.argmin(np.abs(y0 - self._yw_hat(x)))
			x0 = x[x0_index]

			# Finds the error bands
			x_err_neg, x_err_pos = self._yw_hat_err(x)
			x0_err_index_neg = np.argmin(np.abs(y0 - x_err_pos))
			x0_err_index_pos = np.argmin(np.abs(y0 - x_err_neg))
			x0_err = [x[x0_err_index_neg], x[x0_err_index_pos]]

		else:
			# Finds the target value
			min_index = np.argmin(np.abs(y0 - self._y_hat(x)))
			x0 = x[min_index]

			# Finds the error bands
			x_err_neg, x_err_pos = self._y_hat_err(x)
			x0_err_index_neg = np.argmin(np.abs(y0 - x_err_pos))
			x0_err_index_pos = np.argmin(np.abs(y0 - x_err_neg))
			x0_err = [x[x0_err_index_neg], x[x0_err_index_pos]]

		self.y0 = y0
		self.x0 = x0
		self.x0_err = x0_err

		self.inverse_fit_performed = True

		return x0, x0_err

	@staticmethod
	def chi_squared(y, y_err, y_fit):
		"""
		Goodness test of a linear fit. 

		D.o.f. is 2. n is the number of observations.

		X^2 = 1/(n-2) sum^n_{i=1} (y_i - y_fit_i)^2/y_err_i^2

		Args:
			y: numpy float array, observations.
			y_err: numpy float array, standard deviations of the observations.
			y_fit: numpy float array, line fitted through n points.
		"""
		n = len(y)
		assert n == len(y_fit) == len(y_err)
		return np.sum((y - y_fit)**2 / y_err**2) / (n - 2.0)

	def plot(self, weighted=False):
		"""Use full function for quickly checking the fit."""

		# Gets the line fitted and its errors
		if self.inverse_fit_performed:
			fit_target = self.y0
			x_fit = self.x0
			x_fit_err = self.x0_err

		# Gets the signal
		x = self.x
		signal = self.y
		signal_err = self.y_err

		# Gets the fitted line
		if weighted:
			x_hat = self.xw_fit
			y_hat = self.yw_fit
			y_hat_err = self.yw_fit_err
			chi = self.chi_w
			fit_label = "Weigthed fit"
			fit_target_label = r"$x_{0,w}\pm\sigma_{x_0,w}$"
		else:
			x_hat = self.x_fit
			y_hat = self.y_fit
			y_hat_err = self.y_fit_err
			chi = self.chi
			fit_label = "Unweigthed fit"
			fit_target_label = r"$x_0\pm\sigma_{x_0}$"

		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
		ax1.plot(x_hat, y_hat, label=fit_label, color="tab:blue")
		ax1.fill_between(x_hat, y_hat_err[0], y_hat_err[1], alpha=0.5, 
			color="tab:blue")
		ax1.errorbar(x, signal, yerr=signal_err, marker="o", label="Signal", 
			linestyle="none", color="tab:orange")
		title_string = r"$\chi^2 = %.2f" % chi 

		if self.inverse_fit_performed:
			ax1.axhline(fit_target, linestyle="dashed", color="tab:grey")
			ax1.axvline(x_fit, color="tab:orange")
			ax1.fill_betweenx(np.linspace(np.min(y_hat),np.max(y_hat),100),
				x_fit_err[0], x_fit_err[1], label=fit_target_label, alpha=0.5, 
				color="tab:orange")
			title_string += r", x_0 = %.2f \pm%.2f$" % (x_fit,
			(x_fit_err[1] - x_fit_err[0]) / 2.0)

		ax1.legend(loc="best", prop={"size":8})
		ax1.set_title(title_string)
		plt.show()
		plt.close(fig1)

class ErrorPropagationSpline(object):
	"""
	Does a spline fit, but returns both the spline value and associated 
	uncertainty.

	https://gist.github.com/kgullikson88/147f6beb6256307d1360
	"""
	def __init__(self, x, y, yerr, N=10000, *args, **kwargs):
		"""
		See docstring for InterpolatedUnivariateSpline
		"""
		yy = np.vstack([y + np.random.normal(loc=0, scale=yerr) \
			for i in range(N)]).T
		self._splines = [spline(x, yy[:, i], *args, **kwargs) \
			for i in range(N)]

	def __call__(self, x, *args, **kwargs):
		"""
		Get the spline value and uncertainty at point(s) x. args and kwargs 
		are passed to spline.__call__.

		Args:
			x: array or float of points to interpolate at.
		Returns: 
			a tuple with the mean value at x and the standard deviation.
		"""
		x = np.atleast_1d(x) # Converts to at least one dimension
		s = np.vstack([curve(x, *args, **kwargs) for curve in self._splines])
		return (np.mean(s, axis=0), np.std(s, axis=0))

def extract_fit_target(fit_target, x, y, y_err, y_raw=None, tau_int=None,
	tau_int_err=None, extrapolation_method="bootstrap", plateau_size=20, 
	interpolation_rank=3, plot_fit=False, raw_func=lambda y: y, 
	raw_func_der=lambda y, yerr: yerr, verbose=False, **kwargs):
	"""
	Function for extracting a value at a specific point.

	Args:
		fit_target: float, value of where we extrapolate from.
		x: numpy float array
		y: numpy float array
		y_err: numpy float array, errors of y.
		y_raw: optional, numpy float array, raw values of y. E.g. unanalyzed,
			bootstrapped or jackknifed values.
		tau_int: numpy optional, float array, tau_int from autocorrelation.
		tau_int_err: numpy optional, float array, tau_int error from full
			autocorrelation.
		extrapolation_method: str, optional, method of selecting the 
			extrapolation point to do the continuum limit. Method will be used
			on y values and tau int. Choices:
			- plateau: line fits points neighbouring point in order to 
				reduce the error bars. Covariance matrix will be automatically
				included.
			- plateau_mean: line fits points neighbouring point in order to 
				reduce the error bars. Line will be weighted by the y_err.
			- nearest: line fit from the point nearest to what we seek
			- interpolate: linear interpolation in order to retrieve value
				and error. Does not work in conjecture with use_raw_values.
			- bootstrap: will create multiple line fits, and take average. 
				Assumes y_raw is the bootstrapped or jackknifed samples.
		plateau_size: int, optional. Number of points in positive and 
			negative direction to extrapolate fit target value from. This value
			also applies to the interpolation interval.	Default is 20.
		interpolation_rank: int, optional. Interpolation rank to use if 
			extrapolation method is interpolation Default is 3, cubic spline.
		raw_func: function, optional, will modify the bootstrap data after 
			samples has been taken by this function.
		raw_func_err: function, optional, will propagate the error of the 
			bootstrapped line fitted data, raw_func_err(y, yerr). Calculated
			by regular error propagation.
		plot_fit: bool, optional. Will plot and show the extrapolation window.
			Default is false.

	Raises:
		AssertionError: if extrapolation_method, extrapolation_data or
			ac_correction_method is not recognized among built in methods. 

	Returns:
		x0: x axis value at fit target
		y0: y axis value at fit target
		y0_error: y axis error at fit target, not corrected by tau_int
		y0_raw: raw value at y axis fit target
		tau_int0: tau int value at the fit target
	"""

	# Default values
	x0 = fit_target
	y0_raw = None
	tau_int0 = None
	chi_squared = None

	extrap_method_list = ["plateau", "plateau_mean", "nearest", "interpolate", 
		"bootstrap"]

	extrap_method_err = ("%s not an available extrapolation type: %s" % (
			(extrapolation_method, ", ".join(extrap_method_list))))
	assert extrapolation_method in extrap_method_list, extrap_method_err

	fit_index = np.argmin(np.abs(x - fit_target))
	ilow = fit_index - plateau_size
	ihigh = fit_index + plateau_size

	def _f(_x, a, b):
		return _x*a + b

	if extrapolation_method == "plateau":
		y0, y0_error, tau_int0, chi_squared = _extract_plateau_fit(x0, _f, 
			x[ilow:ihigh], y[ilow:ihigh], y_err[ilow:ihigh], 
			y_raw[ilow:ihigh], tau_int[ilow:ihigh], tau_int_err[ilow:ihigh])

	elif extrapolation_method == "plateau_mean":
		y0, y0_error, chi_squared = _extract_plateau_mean_fit(x0, _f, 
			x[ilow:ihigh], y[ilow:ihigh], y_err[ilow:ihigh])

	elif extrapolation_method == "bootstrap":
		# Assumes that y_raw is the bootstrapped samples.
		y0, y0_error, tau_int0 = _extract_bootstrap_fit(x0, _f, x[ilow:ihigh],
			y[ilow:ihigh], y_err[ilow:ihigh], y_raw[ilow:ihigh], 
			tau_int[ilow:ihigh], tau_int_err[ilow:ihigh], F=raw_func,
			FDer=raw_func_der, plot_samples=False)

	elif extrapolation_method == "nearest":
		x0 = x[fit_index]
		y0 = y[fit_index]
		y0_error = y_err[fit_index]

		if isinstance(y_raw, types.NoneType):
			y0_raw = y_raw[fit_index]

		if isinstance(tau_int, types.NoneType):
			tau_int0 = tau_int[fit_index]

	elif extrapolation_method == "interpolate":
		y_spline = ErrorPropagationSpline(x[ilow:ihigh], y[ilow:ihigh],
			y_err[ilow:ihigh], k=interpolation_rank)
		y0, y0_error = y_spline(fit_target)
		y0 = y0[0]
		y0_error = y0_error[0]

	if plot_fit:
		title_string = "Fit: %s" % extrapolation_method
		__plot_fit_target(x[ilow:ihigh], y[ilow:ihigh], y_err[ilow:ihigh],
			x0, y0, y0_error, title_string)

	if verbose:
		msg = "Method:       %s" % extrapolation_method
		msg += "\nx0:       %16.10f" % x0
		msg += "\ny0:       %16.10f" % y0
		msg += "\ny0_error: %16.10f" % y0_error
		if not isinstance(y0_raw, types.NoneType):
			msg += "\ny0_raw:   %16.10f" % y0_raw
		if not isinstance(tau_int0, types.NoneType):
			msg += "\ntau_int0:  %16.10f" % tau_int0
		if not isinstance(chi_squared, types.NoneType):
			msg += "\nchi^2     %16.10f" % chi_squared
		print msg

	return x0, y0, y0_error, y0_raw, tau_int0

def __plot_fit_target(x, y, yerr, x0, y0, y0err, title_string=""):
	"""
	Internal method for plotting the fit window.

	Args:
		x, y, yerr: numpy 1D float arrays, containing unfitted data.
		x0, y0, y0err: float values, containing fit parameters.
		title_string: optional, str. Title name, default is "".
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.axvline(x0, linestyle="dashed", color="tab:grey")
	ax.errorbar(x, y, yerr=yerr, color="tab:orange", 
		label="Original data points", capsize=5, fmt="_", ls=":",
		ecolor="tab:orange")
	ax.errorbar([x0, x0], [y0, y0], yerr=[y0err, y0err], fmt="o",
		capsize=10, color="tab:blue", ecolor="tab:blue")
	ax.legend(loc="best", prop={"size":8})
	ax.set_title(title_string)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	plt.show()
	plt.close(fig)

def _extract_plateau_fit(x0, f, x, y, y_err, y_raw, tau_int, tau_int_err):
	"""
	Extract y0 with y0err at a given x0 by using a line fit with the 
	covariance matrix from the raw y data.

	Args:
		x0: float, value at which to extract a y0 and y0err.
		f: function to fit against.
		x: numpy float array, x-axis to fit against.
		y: numpy float array, y values.
		y_err: numpy float array, y standard deviation values.
		y_raw: numpy float array, y raw values. E.g. bootstrap, jackknifed or 
			analyzed values.
		tau_int: numpy float array, autocorrelation times.
		tau_int_err: numpy float array, autocorrelation time errors.

	Returns:
		y0, y0_error, tau_int0, chi_squared
	"""

	# Uses bootstrap, jackknifed or analyzed values directly.
	cov_raw = np.cov(y_raw)

	assert not isinstance(y_raw, types.NoneType), \
		"missing y_raw values."
	assert not isinstance(tau_int, types.NoneType), \
		"missing tau_int values."
	assert not isinstance(tau_int_err, types.NoneType), \
		"missing tau_int_err values."

	# Get eigenvalues for covariance matrix
	eig = np.linalg.eigvals(cov_raw)

	counter = 1
	while np.min(eig) <= 1e-15:
		# Gets the magnitude of the smallest eigenvalue
		magnitude = np.floor(np.log10(np.absolute(np.min(eig))))
		# Increments magnitude til we have positive definite cov-matrix
		eps = 10**(magnitude + counter)
		eps_matrix = np.zeros(cov_raw.shape)

		# Adds a small diagonal epsilon to make it positive definite
		np.fill_diagonal(eps_matrix, eps)
		cov_raw += eps_matrix

		eig = np.linalg.eigvals(cov_raw)

		# In order no to get stuck at a particular place
		counter += 1
		if counter >= 10:
			raise ValueError("Exceeding maximum iteration 10.")

	# Line fit from the mean values and the raw values covariance matrix
	pol_raw, polcov_raw = sciopt.curve_fit(f, x, y, sigma=cov_raw,
		absolute_sigma=False, p0=[0.18, 0.0], maxfev=1200)
	pol_raw_err = np.sqrt(np.diag(polcov_raw))

	# Extract fit target values
	lfit_raw = LineFit(x, y, y_err)
	lfit_raw.set_fit_parameters(pol_raw[1], pol_raw_err[1], pol_raw[0],
		pol_raw_err[0], weighted=True)
	y0, y0_error, _, chi_squared = lfit_raw.fit_weighted(x0)

	# Errors should be equal in positive and negative directions.
	if np.abs(y0 - y0_error[0]) != np.abs(y0 - y0_error[1]):
		print "Warning: uneven errors:\nlower: %.10f\nupper: %.10f" % (
			y0_error[0], y0_error[1])

	y0 = y0[0] #  y0 and y0_error both comes in form of arrays
	y0_error = ((y0_error[1] - y0_error[0])/2)
	y0_error = y0_error[0]

	# Perform line fit for tau int as well to error correct
	tau_int0 = __get_tau_int(x0, x, tau_int, tau_int_err)

	# Corrects error with the tau int
	y0_error *= np.sqrt(2*tau_int0)

	return y0, y0_error, tau_int0, chi_squared

def _extract_plateau_mean_fit(x0, f, x, y, y_err):
	"""
	Extract y0 with y0err at a given x0 by using a line fitw ith y_err as 
	weights.

	Args:
		x0: float, value at which to extract a y0 and y0err.
		f: function to fit against.
		x: numpy float array, x-axis to fit against.
		y: numpy float array, y values.
		y_err: numpy float array, y standard deviation values.

	Returns:
		y0, y0_error, chi_squared
	"""

	# Line fit from the mean values and their standard deviations
	pol_line, polcov_line = sciopt.curve_fit(f, x, y, sigma=y_err,
	 	absolute_sigma=False)
	pol_line_err = np.sqrt(np.diag(polcov_line))

	# Extract fit target values
	lfit_default = LineFit(x, y, y_err)
	lfit_default.set_fit_parameters(pol_line[1], pol_line_err[1], 
		pol_line[0], pol_line_err[0], weighted=True)
	y0, y0_error, chi_squared = lfit_default(x0, weighted=True)
	y0_error = ((y0_error[1] - y0_error[0])/2)
	return y0[0], y0_error[0], chi_squared

def _extract_bootstrap_fit(x0, f, x, y, y_err, y_raw, tau_int, tau_int_err,
	plot_samples=False, F=lambda y: y, FDer=lambda y, yerr: yerr):
	"""
	Extract y0 with y0err at a given x0 by using line fitting the y_raw data.
	Error will be corrected by line fitting tau int and getting the exact
	value at x0.

	Args:
		x0: float, value at which to extract a y0 and y0err.
		f: function to fit against.
		x: numpy float array, x-axis to fit against.
		y: numpy float array, y values.
		y_err: numpy float array, y standard deviation values.
		y_raw: numpy float array, y raw values. E.g. bootstrap, jackknifed or 
			analyzed values.
		tau_int: numpy float array, autocorrelation times.
		tau_int_err: numpy float array, autocorrelation time errors.
		plot_bs: bool, optional. Will plot the bootstrapped line fits and show.
		F: function, optional, will modify the bootstrap data after 
			samples has been taken by this function.
		FDer: function, optional, will propagate the error of the bootstrapped 
			line fit. Should take y and yerr. Calculated by regular error 
			propagation.

	Returns:
		y0, y0_error, tau_int0, chi_squared
	"""

	y0_sample = np.zeros(y_raw.shape[-1])
	y0_sample_err = np.zeros(y_raw.shape[-1])

	if plot_samples:
		fig_samples = plt.figure()
		ax_samples = fig_samples.add_subplot(111)

		# Empty arrays for storing plot means and errors		
		plot_ymean = np.zeros(y_raw.shape)
		plot_yerr = np.zeros((y_raw.shape[0], y_raw.shape[1], 2))

	# Gets the bootstrapped line fits
	# for i, y_sample in enumerate(tqdm(y_raw.T, desc="Sample line fitting")):
	for i, y_sample in enumerate(y_raw.T):
		p, pcov = sciopt.curve_fit(f, x, y_sample, p0=[0.01, 0.18])
		pfit_err = np.sqrt(np.diag(pcov))

		# Fits sample
		fit_sample = LineFit(x, y_sample)
		fit_sample.set_fit_parameters(p[1], pfit_err[1], p[0], pfit_err[0])
		y0_sample[i], _tmp_err = fit_sample(x0)
		y0_sample_err[i] = (_tmp_err[1] - _tmp_err[0])/2

		if plot_samples:
			plot_ymean[:,i], _p_err = fit_sample(x)
			plot_yerr[:,i] = np.asarray(_p_err).T

			ax_samples.fill_between(x, plot_yerr[:,i,0], plot_yerr[:,i,1], 
				alpha=0.01, color="tab:red")
			ax_samples.plot(x, plot_ymean, label="Scipy curve fit",
				color="tab:red", alpha=0.1)

	# Gets the tau int
	tau_int0 = __get_tau_int(x0, x, tau_int, tau_int_err)

	y0_mean = F(np.mean(y0_sample, axis=0))
	y0_std = FDer(np.mean(y0_sample, axis=0), 
		np.std(y0_sample, axis=0) * np.sqrt(2*tau_int0))

	# y0_mean = np.mean(y0_sample, axis=0)
	# y0_std = np.std(y0_sample, axis=0)*np.sqrt(8*tau_int0)

	if plot_samples:
		sample_mean = F(np.mean(plot_ymean, axis=1))
		sample_std = FDer(np.mean(plot_ymean, axis=1), 
			np.std(plot_ymean, axis=1) * np.sqrt(2*tau_int0))

		# sample_mean = np.mean(plot_ymean, axis=1)
		# sample_std = np.std(plot_ymean, axis=1)*np.sqrt(2*tau_int0)

		# Sets up sample std edges
		ax_samples.plot(x, sample_mean - sample_std, x, 
			sample_mean + sample_std, color="tab:blue",	alpha=0.6)

		# Plots sample mean
		ax_samples.plot(x, sample_mean, label="Averaged samples fit", 
			color="tab:blue")

		# Plots original data with error bars
		ax_samples.errorbar(x, y, yerr=y_err, marker=".", 
			linestyle="none", color="tab:orange", label="Original")

		print "Samples Chi^2: ", LineFit.chi_squared(y, y_err, sample_mean)

		plt.show()
		plt.close(fig_samples)

	# print y[np.argmin(np.abs(x-x0))], y_err[np.argmin(np.abs(x-x0))], "%.10f" % np.abs(y[np.argmin(np.abs(x-x0))] - y0_mean)
	# print y0_mean, y0_std, tau_int0

	return y0_mean, y0_std, tau_int0

def __get_tau_int(x0, x, tau_int, tau_int_err):
	"""Smal internal function for getting tau int at x0."""
	tauFit = LineFit(x, tau_int, y_err=tau_int_err)
	tau_int0, _, _, _ = tauFit.fit_weighted(x0)
	return tau_int0[0]



def _fit_var_printer(var_name, var, var_error, w=16):
	return "{0:<s} = {1:<.{w}f} +/- {2:<.{w}f}".format(var_name, var, 
		var_error, w=w)

def _test_simple_line_fit():
	"""
	Function for testing the case where one compares the curve_fit module in
	scipy.optimize with what I have created.

	SciPy and LineFit should fit exact down to around 8th digit.
	"""
	import random
	import scipy.optimize as sciopt

	# Generates signal with noise
	a = 0.65
	b = 1.1
	N = 5
	x = np.linspace(0, 5, N)
	signal_spread = 0.5
	signal = a*x + b + np.random.uniform(-signal_spread, signal_spread, N)
	signal_err = np.random.uniform(0.1, 0.3, N)

	# Fits without any weights first
	pol1, polcov1 = sciopt.curve_fit(lambda x, a, b : x*a + b, x, signal)

	# Numpy polyfit
	polyfit1, polyfitcov1 = np.polyfit(x, signal, 1, cov=True)
	polyfit_err = np.sqrt(np.diag(polyfitcov1))

	fit = LineFit(x, signal, signal_err)
	x_hat = np.linspace(0, 5, 100)

	# Unweigthed fit
	y_hat, y_hat_err, f_params, chi_unweigthed = fit.fit(x_hat)
	b0, b0_err, b1, b1_err = f_params

	# Fit target
	fit_target = 2.5
	x_fit, x_fit_err = fit.inverse_fit(fit_target)

	print "UNWEIGTHED LINE FIT"
	print "Numpy polyfit:"
	print _fit_var_printer("a", polyfit1[0], polyfit_err[0])
	print _fit_var_printer("b", polyfit1[1], polyfit_err[1])

	print "SciPy curve_fit:"
	print _fit_var_printer("a", pol1[0], polcov1[0,0])
	print _fit_var_printer("b", pol1[1], polcov1[1,1])

	print "LineFit:"
	print _fit_var_printer("a", b1, b1_err)
	print _fit_var_printer("b", b0, b0_err)
	print "Goodness of fit: %f" % chi_unweigthed
	# print "b = {0:<.10f} +/- {1:<.10f}".format(b0, self.b0_err)

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(211)
	ax1.axhline(fit_target, linestyle="dashed", color="tab:grey")
	ax1.plot(x_hat, y_hat, label="Unweigthed fit", color="tab:blue")
	ax1.fill_between(x_hat, y_hat_err[0], y_hat_err[1], alpha=0.5, 
		color="tab:blue")
	ax1.errorbar(x, signal, yerr=signal_err, marker="o", label="Signal", 
		linestyle="none", color="tab:orange")
	ax1.set_ylim(0.5, 5)
	ax1.axvline(x_fit, color="tab:orange")
	ax1.fill_betweenx(np.linspace(0,6,100), x_fit_err[0], x_fit_err[1], 
		label=r"$x_0\pm\sigma_{x_0}$", alpha=0.5, color="tab:orange")
	ax1.legend(loc="best", prop={"size":8})
	ax1.set_title("Fit test - unweigthed")

	# Weighted curve_fit
	print "WEIGTHED LINE FIT"

	# Numpy polyfit
	polyfit1, polyfitcov1 = np.polyfit(x, signal, 1, cov=True, w=1/signal_err)
	polyfit_err = np.sqrt(np.diag(polyfitcov1))
	print "Numpy polyfit:"
	print _fit_var_printer("a", polyfit1[0], polyfit_err[0])
	print _fit_var_printer("b", polyfit1[1], polyfit_err[1])

	# SciPy curve fit
	polw, polcovw = sciopt.curve_fit(lambda x, a, b : x*a + b, x, signal, 
		sigma=signal_err)
	print "SciPy curve_fit:"
	print _fit_var_printer("a", polw[0], polcovw[0,0])
	print _fit_var_printer("b", polw[1], polcovw[1,1])

	# Weighted LineFit
	yw_hat, yw_hat_err, f_params_weigthed, chi_weigthed = fit.fit_weighted(x_hat)
	b0, b0_err, b1, b1_err = f_params_weigthed
	xw_fit, xw_fit_error = fit.inverse_fit(fit_target, weigthed=True)

	print "LineFit:"
	print _fit_var_printer("a", b1, b1_err)
	print _fit_var_printer("b", b0, b0_err)
	print "Goodness of fit: %f" % chi_weigthed

	ax2 = fig1.add_subplot(212)
	ax2.axhline(fit_target, linestyle="dashed", color="tab:grey")
	ax2.errorbar(x, signal, yerr=signal_err, marker="o", label="Signal", 
		linestyle="none", color="tab:orange")
	ax2.plot(x_hat, yw_hat, label="Weighted fit", color="tab:blue")
	ax2.fill_between(x_hat, yw_hat_err[0], yw_hat_err[1], alpha=0.5, 
		color="tab:blue")
	ax2.set_ylim(0.5, 5)
	ax2.axvline(xw_fit, color="tab:orange")
	ax2.fill_betweenx(np.linspace(0,6,100), xw_fit_error[0], xw_fit_error[1], 
		label=r"$x_{0,w}\pm\sigma_{x_0,w}$", alpha=0.5, color="tab:orange")
	ax2.legend(loc="best", prop={"size":8})
	plt.show()

	# fit.plot(True)

def _test_inverse_line_fit():
	"""
	Function for testing the inverse line fit.
	"""
	import scipy.optimize as sciopt

	np.random.seed(1)

	# Generates signal with noise
	x_start = 0
	x_end = 5
	a = 0.65
	b = 1.1
	N = 10 # Number of data points
	M = 40 # Number of observations at each data point
	x = np.linspace(x_start, x_end, N)
	x_matrix = np.ones((M, N)) * x
	signal_spread = 0.5
	signal = np.cos(np.random.uniform(-signal_spread, signal_spread, (M, N)))
	signal = np.random.uniform(-signal_spread, signal_spread, (M, N))
	signal += a*x_matrix + b
	signal_err = np.std(signal, axis=0)
	signal_mean = np.mean(signal, axis=0)

	# My line fit
	x_hat = np.linspace(x_start, x_end, 10)
	X_values = np.linspace(x_start, x_end, 1000)

	# def covariance(M, bias=False):
	# 	"""
	# 	Method for determining coveriance.

	# 	Equivalent to:
	# 		np.cov(M.T, bias=True)
	# 	"""
	# 	_bias_correction = 0
	# 	if bias: _bias_correction = 1
	# 	NObs, NVar = M.shape
	# 	sigma_matrix = np.zeros((N, N))
	# 	M_mean = np.mean(M, axis=0)
	# 	for i in range(NVar):
	# 		v1 = M.T[i] - M_mean[i]
	# 		for j in range(NVar):
	# 			v2 = M.T[j] - M_mean[j]				
	# 			sigma_matrix[i,j] = np.sum(v1*v2)/(NObs - _bias_correction)
	# 	return sigma_matrix
	# if not (np.abs(covariance(signal) - np.cov(signal.T, bias=True)) < 1e-16).all(): \
	# 	print "covariances not equivalent"

	def _f(_x, a, b):
		return _x*a + b

	###############################
	######## Bootstrap fit ########
	###############################
	N_bs = 50
	random_fit_indexes = np.random.randint(0, M, size=(N_bs, M, N))

	print "random_fit_indexes.shape: ", random_fit_indexes.shape
	print "signal.shape: ", signal.shape
	bs_signals = np.zeros((N_bs, N))
	for i_bs in xrange(N_bs):
		for j in xrange(N):
			bs_signals[i_bs,j] = signal[random_fit_indexes[i_bs,:,j], j].mean()

	bs_vals = []
	bs_vals_err = []

	fig_bs = plt.figure()
	ax_bs = fig_bs.add_subplot(111)
	for bs_signal in bs_signals:
		p, pcov = sciopt.curve_fit(_f, x, bs_signal)#, sigma=np.cov(bs_signals.T))
		_fit_err = np.sqrt(np.diag(pcov))
		_lfit = LineFit(x, bs_signal)
		_lfit.set_fit_parameters(p[1], _fit_err[1], p[0], _fit_err[0], 
			weighted=False)
		_y_hat, _y_hat_err = _lfit(x_hat, weighted=False)

		bs_vals.append(_y_hat)
		bs_vals_err.append(_y_hat_err)

		ax_bs.fill_between(x_hat, _y_hat_err[0], _y_hat_err[1], alpha=0.01, 
			color="tab:red")
		ax_bs.plot(x_hat, _y_hat, label="Numpy curve fit", color="tab:red", 
			alpha=0.1)

	bs_mean = np.asarray(bs_vals).mean(axis=0)
	bs_std = np.asarray(bs_vals_err).mean(axis=0)

	lf = LineFit(x, signal_mean, signal_err)
	print "Bootstrap Chi^2: ", lf.chi_squared(signal_mean, signal_err, 
		bs_mean)

	# Sets up bs std edges
	ax_bs.plot(x_hat, bs_std[0], x_hat, bs_std[1], color="tab:blue", alpha=0.6)

	# Plots bs mean
	ax_bs.plot(x_hat, bs_mean, label="Averaged bootstrap fit", color="tab:blue")

	# Plots original data with error bars
	ax_bs.errorbar(x, signal_mean, yerr=signal_err, marker=".", linestyle="none", 
		color="tab:orange")
	plt.show()
	plt.close(fig_bs)

	exit(1)


	# import symfit as sf
	# _a, _b = sf.parameters('a,b')
	# _x, _y = sf.variables('x,y')
	# res = sf.Fit({_y: _a * _x + _b}, x=x, y=signal_mean, sigma_y=signal_err)
	# print res

	## Unweigthed fit
	# print help(sciopt.curve_fit)

	# pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=np.cov(signal.T))
	# print scipy.optimize.__all__

	# pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=signal_err, 
	# 	absolute_sigma=True)

	from autocorrelation import Autocorrelation

	# print help(Autocorrelation)

	_ac_array = np.zeros((N,M/2))
	_tau_ints = np.zeros(N)

	print signal.shape, M

	for i in xrange(N):
		ac = Autocorrelation(signal[:,i])
		_ac_array[i] = ac.R
		_tau_ints[i] = ac.integrated_autocorrelation_time()


	def autocov(x, y): 
		n = len(x)
		# variance = x.var()
		x = x-x.mean()
		y = y-y.mean()
		G = np.correlate(x, y, mode="full")[-n:]
		G /= np.arange(n, 0, -1)
		return G

	# autocov_mat = np.zeros((N,N))
	# for i in xrange(N):
	# 	for j in xrange(N):
	# 		autocov_mat[i,j] = autocov(signal[:,i], signal[:,j])
	# print autocov
	# print G_matrix.shape

	cov_mat = np.cov(signal.T)
	print cov_mat.shape

	for i in xrange(N):
		cov_mat[i,i] *= 2*_tau_ints[i]

	lfit = LineFit(x, signal_mean, y_err=signal_err)

	
	# pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=signal_err, 
	# 	absolute_sigma=True)

	signal_err_corrected = np.sqrt(np.diag(cov_mat))
	print "signal_err_corrected: ", signal_err_corrected[:5]

	get_err = lambda _err: (_err[1] - _err[0])/2

	pol1, polcov1 = np.polyfit(x, signal_mean, 1, w=1/np.diag(cov_mat), cov=True)
	lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0], 
		np.sqrt(polcov1[0,0]), weighted=True)
	y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
	print "With polyfit:            ", np.sqrt(np.diag(polcov1)), 
	get_err(y_hat_err)[:5]
	
	pol1, polcov1 = scipy.optimize.curve_fit(_f, x, signal_mean, 
		sigma=np.sqrt(np.diag(cov_mat)))
	lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0],
		np.sqrt(polcov1[0,0]), weighted=True)
	y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
	print "With naive autocorr:     ", np.sqrt(np.diag(polcov1)), 
	get_err(y_hat_err)[:5]
	
	pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=np.cov(signal.T))
	lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0],
		np.sqrt(polcov1[0,0]), weighted=True)
	y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
	print "With cov(signal.T):      ", np.sqrt(np.diag(polcov1)), 
	get_err(y_hat_err)[:5]
	
	pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=signal_err)
	lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0], 
		np.sqrt(polcov1[0,0]), weighted=True)
	y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
	print "With only signal errors: ", np.sqrt(np.diag(polcov1)), 
	get_err(y_hat_err)[:5]

	#INTERPOLATION
	
	# print help(interpolate.InterpolatedUnivariateSpline)
	s = spline(x, signal_mean, w=1/signal_err, k=1)
	# print s(X_values)

	spl = ErrorPropagationSpline(x, signal_mean, signal_err, k=1)
	spline_mean, spline_err = spl(X_values)
	print spline_mean[:5]
	print spline_err[:5]



	fit_par = pol1
	fit_err = np.sqrt(np.diag(polcov1))

	# print np.cov(signal_mean.T)
	# lfit.set_fit_parameters(fit_par[1], fit_err[1], fit_par[0], fit_err[0],
	# 	weighted=True)
	lfit.fit_weighted(x_arr=x_hat)
	y_hat, y_hat_err, chi_squared = lfit(x_hat, weighted=True)
	print "Regular weighted fit Chi^2: ", chi_squared

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)

	# REGULAR LINE FIT
	# ax1.plot(x_hat, y_hat, label="Numpy curve fit", color="tab:blue")
	# ax1.fill_between(x_hat, y_hat_err[0], y_hat_err[1], alpha=0.5,
	# 	color="tab:blue")

	# SPLINE FIT
	ax1.plot(X_values, spline_mean, label="Numpy curve fit", color="tab:blue")
	ax1.fill_between(X_values, spline_mean-spline_err, spline_mean+spline_err,
		alpha=0.5, color="tab:blue")
	ax1.errorbar(x, signal_mean, yerr=signal_err, marker=".",
		label=r"Signal $\chi=%.2f$" % chi_squared, linestyle="none", 
		color="tab:orange")
	ax1.set_ylim(0.5, 5)
	# ax1.axvline(x_fit, color="tab:orange")
	# ax1.fill_betweenx(np.linspace(0,6,100), x_fit_err[0], x_fit_err[1], 
	# 	label=r"$x_0\pm\sigma_{x_0}$", alpha=0.5, color="tab:orange")
	ax1.legend(loc="best", prop={"size":8})
	ax1.set_title((r"Fit test: $a=%.2f\pm%.4f, b=%.2f\pm%.4f$" %
		(fit_par[0], fit_err[0], fit_par[1], fit_err[1])))

	plt.show()
	# fit = LineFit(x, signal, signal_err)
	# x_hat = np.linspace(0, 5, 100)

def main():
	# _test_simple_line_fit()
	_test_inverse_line_fit()

if __name__ == '__main__':
	main()