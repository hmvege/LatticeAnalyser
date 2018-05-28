#!/usr/bin/env python2

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from linefit_tools import __plot_fit_target, _extract_bootstrap_fit, \
	_extract_plateau_fit, _extract_plateau_mean_fit, _extract_inverse
import scipy.optimize as sciopt
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import types
from tqdm import tqdm

__all__ = ["LineFit", "extract_fit_target"]

from matplotlib import rc
rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})

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
			_extract_inverse(y0, x, self._yw_hat(x), self._yw_hat_err(x))
			# # Finds the target value
			# x0_index = np.argmin(np.abs(y0 - self._yw_hat(x)))
			# x0 = x[x0_index]

			# # Finds the error bands
			# x_err_neg, x_err_pos = self._yw_hat_err(x)
			# x0_err_index_neg = np.argmin(np.abs(y0 - x_err_pos))
			# x0_err_index_pos = np.argmin(np.abs(y0 - x_err_neg))
			# x0_err = [x[x0_err_index_neg], x[x0_err_index_pos]]

		else:
			_extract_inverse(y0, x, self._y_hat(x), self._y_hat_err(x))
			# # Finds the target value
			# min_index = np.argmin(np.abs(y0 - self._y_hat(x)))
			# x0 = x[min_index]

			# # Finds the error bands
			# x_err_neg, x_err_pos = self._y_hat_err(x)
			# x0_err_index_neg = np.argmin(np.abs(y0 - x_err_pos))
			# x0_err_index_pos = np.argmin(np.abs(y0 - x_err_neg))
			# x0_err = [x[x0_err_index_neg], x[x0_err_index_pos]]

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
	raw_func_der=lambda y, yerr: yerr, inverse_fit=False, verbose=False, 
	**kwargs):
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
		plot_fit: bool, optional. Will plot and show the extrapolation window.
			Default is false.
		raw_func: function, optional, will modify the bootstrap data after 
			samples has been taken by this function.
		raw_func_err: function, optional, will propagate the error of the 
			bootstrapped line fitted data, raw_func_err(y, yerr). Calculated
			by regular error propagation.
		inverse_fit: bool, optional. If True, will return the x0 value with 
			x0_errors instead of y0 and its errors. Default is False.

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

def extract_inverse_fit_target(fit_target, x, y, y_err, y_raw=None, tau_int=None,
	tau_int_err=None, extrapolation_method="bootstrap", plateau_size=20, 
	interpolation_rank=3, plot_fit=False, raw_func=lambda y: y, 
	raw_func_der=lambda y, yerr: yerr, verbose=False, 
	**kwargs):
	"""
	Function for extracting a value at a specific point at the y-axis.

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
		plot_fit: bool, optional. Will plot and show the extrapolation window.
			Default is false.
		raw_func: function, optional, will modify the bootstrap data after 
			samples has been taken by this function.
		raw_func_err: function, optional, will propagate the error of the 
			bootstrapped line fitted data, raw_func_err(y, yerr). Calculated
			by regular error propagation.

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
	# x0 = fit_target
	y0_raw = None
	tau_int0 = None
	chi_squared = None

	extrap_method_list = ["plateau", "plateau_mean", "nearest", "interpolate",
		"bootstrap"]

	extrap_method_err = ("%s not an available extrapolation type: %s" % (
			(extrapolation_method, ", ".join(extrap_method_list))))
	assert extrapolation_method in extrap_method_list, extrap_method_err

	fit_index = np.argmin(np.abs(y - fit_target))
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


if __name__ == '__main__':
	exit()