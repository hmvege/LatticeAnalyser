#!/usr/bin/env python2

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import types

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
		self.chi = self.chi_squared(self.y, self.y_err, self._y_hat(self.x))

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
		_pt2 = scipy.stats.t.isf(0.32, self.n - 2) * np.sqrt(self.s_xyw_err) 
		_pt2 *= np.sqrt(1.0 / np.sum(self.w) + (x - self.xw_mean)**2 / self.xwi_xmean_sum)

		# print [_pt1 - _pt2, _pt1 + _pt2]

		return [_pt1 - _pt2, _pt1 + _pt2]

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

	def chi_squared(self, y, y_err, y_fit, n=None):
		"""Goodness test of fit."""
		if isinstance(n, types.NoneType):
			n = self.n
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
	# polyfit1, polyfitcov1 = np.polyfit(x, signal, 1, cov=True)
	# polyfit_err = np.sqrt(np.diag(polyfitcov1))
	polyfit1, polyfitcov1 = np.polyfit(x, signal, 1, cov=True)
	polyfit_err = np.sqrt(np.diag(polyfitcov1))

	fit = LineFit(x, signal, signal_err)
	x_hat = np.linspace(0, 5, 100)

	# fit.set_fit_parameters(polyfit1[0], polyfit_err[0], polyfit1[1], polyfit_err[1], weighted=False)

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
	signal_spread = 2
	signal = np.cos(np.random.uniform(-signal_spread, signal_spread, (M, N)))
	signal += a*x_matrix + b
	signal_err = np.std(signal, axis=0)
	signal_mean = np.mean(signal, axis=0)

	# My line fit
	x_hat = np.linspace(x_start, x_end, 10)
	X_values = np.linspace(x_start, x_end, 50)

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
	# if not (np.abs(covariance(signal) - np.cov(signal.T, bias=True)) < 1e-16).all(): print "covariances not equivalent"

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
		_lfit.set_fit_parameters(p[1], _fit_err[0], p[0], _fit_err[0], 
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
		bs_mean, n=N)

	# Sets up bs std edges
	ax_bs.plot(x_hat, bs_std[0], x_hat, bs_std[1], color="tab:blue", alpha=0.6)

	# Plots bs mean
	ax_bs.plot(x_hat, bs_mean, label="Averaged bootstrap fit", color="tab:blue")

	# Plots original data with error bars
	ax_bs.errorbar(x, signal_mean, yerr=signal_err, marker=".", linestyle="none", 
		color="tab:orange")

	plt.close(fig_bs)

	# exit(1)


	# import symfit as sf
	# _a, _b = sf.parameters('a,b')
	# _x, _y = sf.variables('x,y')
	# res = sf.Fit({_y: _a * _x + _b}, x=x, y=signal_mean, sigma_y=signal_err)
	# print res

	## Unweigthed fit
	# print help(sciopt.curve_fit)

	# pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=np.cov(signal.T))
	# print scipy.optimize.__all__

	# pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=signal_err, absolute_sigma=True)

	from autocorrelation import Autocorrelation

	# print help(Autocorrelation)

	_ac_array = np.zeros((N,20))
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

	
	# exit(1)
	# pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=signal_err, absolute_sigma=True)

	signal_err_corrected = np.sqrt(np.diag(cov_mat))
	print "signal_err_corrected: ", signal_err_corrected[:5]

	get_err = lambda _err: (_err[1] - _err[0])/2

	pol1, polcov1 = np.polyfit(x, signal_mean, 1, w=1/np.diag(cov_mat), cov=True)
	lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0], np.sqrt(polcov1[0,0]), weighted=True)
	y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
	print "With polyfit:            ", np.sqrt(np.diag(polcov1)), get_err(y_hat_err)[:5]
	
	pol1, polcov1 = scipy.optimize.curve_fit(_f, x, signal_mean, sigma=np.sqrt(np.diag(cov_mat)))
	lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0], np.sqrt(polcov1[0,0]), weighted=True)
	y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
	print "With naive autocorr:     ", np.sqrt(np.diag(polcov1)), get_err(y_hat_err)[:5]
	
	pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=np.cov(signal.T))
	lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0], np.sqrt(polcov1[0,0]), weighted=True)
	y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
	print "With cov(signal.T):      ", np.sqrt(np.diag(polcov1)), get_err(y_hat_err)[:5]
	
	pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=signal_err)
	lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0], np.sqrt(polcov1[0,0]), weighted=True)
	y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
	print "With only signal errors: ", np.sqrt(np.diag(polcov1)), get_err(y_hat_err)[:5]

	#INTERPOLATION
	from scipy.interpolate import InterpolatedUnivariateSpline as spline
	# print help(interpolate.InterpolatedUnivariateSpline)
	s = spline(x, signal_mean, w=1/signal_err, k=1)
	# print s(X_values)

	class ErrorPropagationSpline(object):
		"""
		Does a spline fit, but returns both the spline value and associated uncertainty.

		https://gist.github.com/kgullikson88/147f6beb6256307d1360
		"""
		def __init__(self, x, y, yerr, N=1000, *args, **kwargs):
			"""
			See docstring for InterpolatedUnivariateSpline
			"""
			yy = np.vstack([y + np.random.normal(loc=0, scale=yerr) for i in range(N)]).T
			print yy
			self._splines = [spline(x, yy[:, i], *args, **kwargs) for i in range(N)]

		def __call__(self, x, *args, **kwargs):
			"""
			Get the spline value and uncertainty at point(s) x. args and kwargs are passed to spline.__call__
			:param x:
			:return: a tuple with the mean value at x and the standard deviation
			"""
			x = np.atleast_1d(x) # Converts to at least one dimension
			s = np.vstack([curve(x, *args, **kwargs) for curve in self._splines])
			return (np.mean(s, axis=0), np.std(s, axis=0))

	spl = ErrorPropagationSpline(x,signal_mean,signal_err)
	spline_mean, spline_err = spl(X_values)
	print spline_mean[:5]
	print spline_err[:5]



	fit_par = pol1
	fit_err = np.sqrt(np.diag(polcov1))

	# print np.cov(signal_mean.T)
	# lfit.set_fit_parameters(fit_par[1], fit_err[1], fit_par[0], fit_err[0], weighted=True)
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
	ax1.fill_between(X_values, spline_mean-spline_err, spline_mean+spline_err, alpha=0.5,
		color="tab:blue")
	ax1.errorbar(x, signal_mean, yerr=signal_err, marker=".",
		label=r"Signal $\chi=%.2f$" % chi_squared, linestyle="none", 
		color="tab:orange")
	ax1.set_ylim(0.5, 5)
	# ax1.axvline(x_fit, color="tab:orange")
	# ax1.fill_betweenx(np.linspace(0,6,100), x_fit_err[0], x_fit_err[1], label=r"$x_0\pm\sigma_{x_0}$", alpha=0.5, color="tab:orange")
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