#!/usr/bin/env python2

from scipy.interpolate import InterpolatedUnivariateSpline as spline
import linefit as lfit
import scipy.optimize as sciopt
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
import types
import copy
from decimal import Decimal
# from tqdm import tqdm


def _extract_inverse(fit_target, X, Y, Y_err):
    """
    Simple method for finding the x axis value to a target value.

    Returns: 
        min(fit_target - Y)
        error(min(fit_target - Y))
    """

    # Finds the target value
    min_index = np.argmin(np.abs(fit_target - Y))
    x0 = X[min_index]

    X, Y, Y_err = map(np.asarray, (X, Y, Y_err))

    if len(Y_err.shape) != 2:
        Y_err = np.asarray([Y-Y_err, Y+Y_err])

    # Finds the error bands
    y_err_neg, y_err_pos = Y_err

    x0_err_index_neg = np.argmin(np.abs(fit_target - y_err_pos))
    x0_err_index_pos = np.argmin(np.abs(fit_target - y_err_neg))

    # If the indices are equal, i.e. the error is too small and we have the
    # same x_err for both positive and negative, we find the next error by
    # looking at the one closest to x_err.
    if x0_err_index_pos == x0_err_index_neg:
        if (np.abs(y_err_neg[x0_err_index_neg-1] - fit_target) >
                np.abs(y_err_pos[x0_err_index_pos+1] - fit_target)):
            x0_err_index_pos += 1
        else:
            x0_err_index_neg -= 1

    x0_err = [X[x0_err_index_neg], X[x0_err_index_pos]]

    return x0, x0_err


def _get_covariance_matrix_from_raw(y_raw, iscov=False):
    """
    Returns a covariance matrix that is guaranteed to not be singular.
    """
    y_raw.astype(Decimal)

    if not iscov:
        cov_raw = np.cov(y_raw)
    else:
        cov_raw = y_raw

    # min_eig = np.min(np.real(np.linalg.eigvals(cov_raw)))
    # print min_eig
    # min_eig = np.min(np.linalg.eigh(cov_raw)[0])
    # if min_eig <= 0:
    #     cov_raw -= 1*min_eig * np.eye(*cov_raw.shape)

    min_eig = np.min(np.linalg.eigh(cov_raw)[0])
    counter = 0

    # print "BEFORE: ",min_eig, "%10.16f" % cov_raw[10,10], min_eig * np.eye(*cov_raw.shape)[0,0]
    # print

    while min_eig < 0 and counter < 10000:
        cov_raw -= 0.5*min_eig * np.eye(*cov_raw.shape)
        min_eig = np.min(np.linalg.eigh(cov_raw)[0])
        counter += 1

        # print counter, min_eig, "%10.16f" % cov_raw[10,10], 0.01*min_eig * np.eye(*cov_raw.shape)[0,0]

    # min_eig = np.min(np.real(np.linalg.eigvals(cov_raw)))
    # print min_eig

    return cov_raw


def __plot_fit_target(x, y, yerr, x0, y0, y0err, title_string="",
                      inverse_fit=False):
    """
    Internal method for plotting the fit window.

    Args:
        x, y, yerr: numpy 1D float arrays, containing unfitted data.
        x0, y0, y0err: float values, containing fit parameters.
        title_string: optional, str. Title name, default is "".
        inverse_fit, optional, bool. Plots errors as inverse errorbar.
            Default is false.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=yerr, color="tab:orange",
                label="Original data points", capsize=5, fmt="_", ls=":",
                ecolor="tab:orange")
    if inverse_fit:
        fit_lab = "$y_0 = %.2f$" % x0
        lab = r"$x_0 = %.4f \pm %g$" % (y0, y0err)
        ax.axhline(x0, linestyle="dashed", color="tab:grey", label=fit_lab)
        ax.errorbar([y0, y0], [x0, x0], xerr=[y0err, y0err], fmt="o",
                    capsize=10, color="tab:blue", ecolor="tab:blue", label=lab)
    else:
        fit_lab = "$x_0 = %.2f$" % x0
        lab = r"$y_0 = %.4f \pm %g$" % (y0, y0err)
        ax.axvline(x0, linestyle="dashed", color="tab:grey", label=fit_lab)
        ax.errorbar([x0, x0], [y0, y0], yerr=[y0err, y0err], fmt="o",
                    capsize=10, color="tab:blue", ecolor="tab:blue", label=lab)
    ax.legend(loc="best", prop={"size": 10})
    ax.set_title(r"%s" % title_string)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    plt.show()
    plt.close(fig)


def _extract_plateau_fit(fit_target, f, x, y, y_err, y_raw, tau_int=None,
                         tau_int_err=None, inverse_fit=False):
    """
    Extract y0 with y0err at a given x0 by using a line fit with the 
    covariance matrix from the raw y data.

    Args:
        fit_target: float, value at which to extract a y0 and y0err.
        f: function to fit against.
        x: numpy float array, x-axis to fit against.
        y: numpy float array, y values.
        y_err: numpy float array, y standard deviation values.
        y_raw: numpy float array, y raw values. E.g. bootstrap, jackknifed or 
            analyzed values.
        tau_int: numpy float array, optional, autocorrelation times. Default 
            is None.
        tau_int_err: numpy float array, optional, autocorrelation time errors.
            Default is None.
        inverse_fit: bool, optional. If True, will perform an inverse fit. 
            y0, y0err -> x0, x0err. Default is False.

    Returns:
        y0, y0_error, tau_int0, chi_squared
    """

    assert not isinstance(y_raw, types.NoneType), \
        "missing y_raw values."
    assert not isinstance(tau_int, types.NoneType), \
        "missing tau_int values."
    assert not isinstance(tau_int_err, types.NoneType), \
        "missing tau_int_err values."

    cov_raw = _get_covariance_matrix_from_raw(y_raw)

    # Line fit from the mean values and the raw values covariance matrix
    pol_raw, polcov_raw = sciopt.curve_fit(f, x, y, sigma=cov_raw,
                                           p0=[0.18, 0.0], maxfev=1200)
    pol_raw_err = np.sqrt(np.diag(polcov_raw))

    # Extract fit target values
    lfit_raw = lfit.LineFit(x, y, y_err)
    lfit_raw.set_fit_parameters(pol_raw[1], pol_raw_err[1], pol_raw[0],
                                pol_raw_err[0], weighted=True)
    y0, y0_error, chi_squared = lfit_raw(fit_target, weighted=True)

    if inverse_fit:
        y0, y0_error = lfit_raw.inverse_fit(fit_target, weighted=True)
    else:
        y0 = y0[0]  # y0 and y0_error both comes in form of arrays

        def f_err(_x, a_err, b_err): return np.sqrt(
            (_x*a_err)**2 + (b_err)**2 + 2*a_err*_x*b_err)
        err = f_err(fit_target, pol_raw_err[0], pol_raw_err[1])

        # If I want to include covariances in old fit model, assuming that
        # y0_error is only a combination of a and b errors.
        y0_error = (y0_error[1][0] - y0_error[0][0])/2
        y0_error = np.sqrt(
            y0_error**2 + 2*fit_target*pol_raw_err[0]*pol_raw_err[1])

        # print y0_error, err, y0_error - err

        y0_error = f_err(fit_target, pol_raw_err[0], pol_raw_err[1])
        # y0_error = err

        # print "Plateau fit: %20.16f +/- %-20.16f, chi^2 %g" % (
        #     y0, y0_error, chi_squared)
        # exit(1)

    # Gets the tau int using a line fit, given it is provide.
    if not isinstance(tau_int, types.NoneType) and \
            not isinstance(tau_int_err, types.NoneType):
        if inverse_fit:
            tau_int0 = __get_tau_int(y0, x, tau_int, tau_int_err)
        else:
            tau_int0 = __get_tau_int(fit_target, x, tau_int, tau_int_err)
    else:
        tau_int0 = 0.5

    # Corrects error with the tau int
    y0_error *= np.sqrt(2*tau_int0)

    return y0, y0_error, tau_int0, chi_squared


def _extract_plateau_mean_fit(fit_target, f, x, y, y_err, inverse_fit=False):
    """
    Extract y0 with y0err at a given fit_target by using a line fitw ith y_err 
    as weights. If inverse, will return x0 with x0err at a given y0.

    Args:
        fit_target: float, value at which to extract a y0 and y0err.
        f: function to fit against.
        x: numpy float array, x-axis to fit against.
        y: numpy float array, y values.
        y_err: numpy float array, y standard deviation values.
        inverse_fit: bool, optional. If True, will perform an inverse fit. 
            Default is False.

    Returns:
        y0, y0_error, chi_squared
    """

    # Line fit from the mean values and their standard deviations
    pol_line, polcov_line = sciopt.curve_fit(f, x, y, sigma=y_err,
                                             absolute_sigma=False)
    pol_line_err = np.sqrt(np.diag(polcov_line))

    # Extract fit target values
    lfit_default = lfit.LineFit(x, y, y_err)
    lfit_default.set_fit_parameters(pol_line[1], pol_line_err[1],
                                    pol_line[0], pol_line_err[0], weighted=True)

    if inverse_fit:
        y0, y0_error = lfit_default.inverse_fit(fit_target, weighted=True)
        _, _, chi_squared = lfit_default(fit_target, weighted=True)
    else:
        y0, y0_error, chi_squared = lfit_default(fit_target, weighted=True)
        y0_error = ((y0_error[1] - y0_error[0])/2)

        # f_err = lambda _x, a_err, b_err: np.sqrt((_x*a_err)**2 + (b_err)**2 + 2*a_err*_x*b_err)
        # y0_error = f_err(fit_target, pol_line_err[0], pol_line_err[1])

        # print "OLD METHOD:               %20.16f +/- %-18.16f, chi^2 %g" % (y0, y0_error, chi_squared)

        # _a, _a_err = lfit_default.b1w, lfit_default.b1w_err
        # _b, _b_err = lfit_default.b0w, lfit_default.b0w_err
        # y_func = lambda _x: _a*_x + _b
        # y_error_func = lambda _x: np.sqrt(
        #     (_a_err*_x)**2 + (_b_err)**2 + 2*_a_err*_x*_b_err)
        # chi_squared = lfit_default.chi_squared(y_func(x), y_error_func(x), x)
        # y0 = y_func(fit_target)
        # y0_error = y_error_func(fit_target)

        # print _a, _a_err, _b, _b_err
        # print "WITH COVARIANCE TERM", y0, y0_error

    if isinstance(y0, (tuple, list, np.ndarray)):
        y0 = y0[0]
    if isinstance(y0_error, (tuple, list, np.ndarray)):
        y0_error = y0_error[0]

    return y0, y0_error, chi_squared


def _extract_bootstrap_fit(fit_target, f, x, y, y_err, y_raw, tau_int=None,
                           tau_int_err=None, plot_samples=False, 
                           F=lambda _y: _y,  FDer=lambda _y, _yerr: _yerr, 
                           inverse_fit=False, full=False):
    """
    Extract y0 with y0err at a given x0 by using line fitting the y_raw data.
    Error will be corrected by line fitting tau int and getting the exact
    value at x0.

    Args:
        fit_target: float, value at which to extract a y0 and y0err.
        f: function to fit against.
        x: numpy float array, x-axis to fit against.
        y: numpy float array, y values.
        y_err: numpy float array, y standard deviation values.
        y_raw: numpy float array, y raw values. E.g. bootstrap, jackknifed or 
            analyzed values.
        tau_int: optional, numpy float array, autocorrelation times.
        tau_int_err: optional, numpy float array, autocorrelation time errors.
        plot_bs: bool, optional. Will plot the bootstrapped line fits and show.
        F: function, optional, will modify the bootstrap data after 
            samples has been taken by this function.
        FDer: function, optional, will propagate the error of the bootstrapped 
            line fit. Should take y and yerr. Calculated by regular error 
            propagation.
        inverse_fit: bool, optional. If True, will perform an inverse fit, with
            the fit target as a y axis value. Default is False.
        full: bool, optional. If True, returns the entire line, and not just 
            y0 with its errors.

    Returns:
        y0, y0_error, tau_int0, chi_squared
    """

    N, M = y_raw.shape  # Points, samples
    y0_sample = np.zeros(M)
    y0_sample_err = np.zeros(M)

    if plot_samples:
        N_plot_values = 1000
        x_plot = np.linspace(x[0], x[-1], N_plot_values)
        fig_samples = plt.figure()
        ax_samples = fig_samples.add_subplot(111)

        # Empty arrays for storing plot means and errors
        y_plot_points = np.zeros((N, M))
        plot_ymean = np.zeros((N_plot_values, M))
        plot_yerr = np.zeros((N_plot_values, M, 2))

    # Gets the bootstrapped line fits
    # for i, y_sample in enumerate(tqdm(y_raw.T, desc="Sample line fitting")):
    for i, y_sample in enumerate(y_raw.T):
        p, pcov = sciopt.curve_fit(f, x, y_sample, p0=[0.01, 0.18])
        pfit_err = np.sqrt(np.diag(pcov))

        # Fits sample
        fit_sample = lfit.LineFit(x, y_sample)
        fit_sample.set_fit_parameters(p[1], pfit_err[1], p[0], pfit_err[0])

        if inverse_fit:
            # Performs an inverse fit where y0 is the target value, and not x0.
            # _x0_err is not needed, is error cannot be inflated by the
            # autocorrelation.
            try:
                y0_sample[i], y0_sample_err[i] = \
                    fit_sample.inverse_fit(fit_target)
            except IndexError as err:
                print "Points included: %s" % y0_raw.shape
                fit_sample.plot()
                exit(err)
        else:
            y0_sample[i], _tmp_err, _ = fit_sample(fit_target)
            y0_sample_err[i] = (_tmp_err[1] - _tmp_err[0])/2

        if plot_samples:
            # Retrieves at exactly sample values x
            y_plot_points[:, i], _, _ = fit_sample(x)

            # Retrieves a smooth set of y values.
            plot_ymean[:, i], _p_err, _ = fit_sample(x_plot)
            plot_yerr[:, i] = np.asarray(_p_err).T

            ax_samples.fill_between(x_plot, plot_yerr[:, i, 0], plot_yerr[:, i, 1],
                                    alpha=0.01, color="tab:red")
            ax_samples.plot(
                x_plot, plot_ymean[:, i], color="tab:red", alpha=0.1)

    y0_mean = F(np.mean(y0_sample, axis=0))

    # Gets the tau int. Asserted that is is provided.
    if not isinstance(tau_int, types.NoneType) and \
            not isinstance(tau_int_err, types.NoneType) and \
            not inverse_fit:
        tau_int0 = __get_tau_int(fit_target, x, tau_int, tau_int_err)
        # tau_int0 = __get_tau_int(y0_mean, x, tau_int, tau_int_err)
    else:
        tau_int0 = 0.5

    # # Using error propegation on the std of the fit and correcting by tau int
    # y0_std = FDer(np.mean(y0_sample, axis=0),
    #   np.std(y0_sample, axis=0) * np.sqrt(2*tau_int0))
    # print y0_std

    # # Using error propegation on the fit errors and correcting by tau int
    # y0_std = FDer(np.mean(y0_sample, axis=0),
    #   np.mean(y0_sample_err) * np.sqrt(2*tau_int0))
    # print y0_std

    # y0_std = np.mean(FDer(y0_sample, y0_sample_err * np.sqrt(2*tau_int0)))
    # print y0_std

    # y0_std = np.mean(FDer(y0_sample, y0_sample_err))*np.sqrt(2*tau_int0)
    # print y0_std

    # Using the mean of the line fit errors and correcting it by tau int
    # y0_std = np.mean(F(y0_sample_err))*np.sqrt(2*tau_int0)
    # print y0_std

    # Using standard deviation of the line fits and correcting it by tau int
    y0_std = np.std(F(y0_sample))*np.sqrt(2*tau_int0)
    # print y0_std

    if plot_samples:
        sample_mean = F(np.mean(plot_ymean, axis=1))
        sample_std = FDer(np.mean(plot_ymean, axis=1),
                          np.std(plot_ymean, axis=1) * np.sqrt(2*tau_int0))

        y_sample_points = F(np.mean(y_plot_points, axis=1))

        bs_chi2 = lfit.LineFit.chi_squared(y, y_err, y_sample_points)

        # sample_mean = np.mean(plot_ymean, axis=1)
        # sample_std = np.std(plot_ymean, axis=1)*np.sqrt(2*tau_int0)

        # Sets up sample std edges
        ax_samples.plot(x_plot, sample_mean - sample_std, x_plot,
                        sample_mean + sample_std, color="tab:blue", alpha=0.6)

        # Plots sample mean
        ax_samples.plot(x_plot, sample_mean, label="Averaged samples fit",
                        color="tab:blue")

        # Plots original data with error bars
        ax_samples.errorbar(x, y, yerr=y_err, marker=".",
                            linestyle="none", color="tab:orange", label="Original")

        ax_samples.set_title(r"$\chi^2: %g$" % bs_chi2)

        plt.show()
        plt.close(fig_samples)

    if full:
        return y0_mean, y0_std, tau_int0, bs_chi2, x_plot, sample_mean, sample_std
    else:
        return y0_mean, y0_std, tau_int0


def __get_tau_int(x0, x, tau_int, tau_int_err):
    """Smal internal function for getting tau int at x0."""
    tauFit = lfit.LineFit(x, tau_int, y_err=tau_int_err)
    tau_int0, _, _, _ = tauFit.fit_weighted(x0)
    return tau_int0[0]


def _test_simple_line_fit():
    """
    Function for testing the case where one compares the curve_fit module in
    scipy.optimize with what I have created.

    SciPy and lfit.LineFit should fit exact down to around 8th digit.
    """
    import random
    import scipy.optimize as sciopt
    np.random.seed(1234)

    def _fit_var_printer(var_name, var, var_error, w=16):
        return "{0:<s} = {1:<.{w}f} +/- {2:<.{w}f}".format(var_name, var,
                                                           var_error, w=w)

    # Generates signal with noise
    a = 0.65
    b = 1.1
    N = 8
    M = 10
    x_start = 0
    x_end = 5
    x = np.linspace(x_start, x_end, N)
    x_matrix = np.ones((M, N)) * x
    signal_spread = 0.7
    signal = a*x + b + np.random.uniform(-signal_spread, signal_spread, (M, N))
    signal_err = np.std(signal, axis=0)
    signal_mean = np.mean(signal, axis=0)

    fit_target = 2.5

    # # Generates signal with noise
    # a = 0.65
    # b = 1.1
    # N = 5
    # x = np.linspace(0, 5, N)
    # signal_spread = 0.5
    # signal_mean = a*x + b + np.random.uniform(-signal_spread, signal_spread, N)
    # signal_err = np.random.uniform(0.1, 0.3, N)

    # Fits without any weights first
    pol1, polcov1 = sciopt.curve_fit(lambda x, a, b: x*a + b, x, signal_mean)
    scipy_lfit = lfit.LineFit(x, signal_mean, signal_err)
    scipy_lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1, 1]), pol1[0],
                                  np.sqrt(polcov1[0, 0]), weighted=False)
    y_scipy = scipy_lfit(fit_target, weighted=False)

    # Numpy polyfit
    polyfit1, polyfitcov1 = np.polyfit(x, signal_mean, 1, cov=True)
    polyfit_err = np.sqrt(np.diag(polyfitcov1))

    fit = lfit.LineFit(x, signal_mean, signal_err)
    x_hat = np.linspace(0, 5, 100)

    # Unweighted fit
    y_hat, y_hat_err, f_params, chi_unweighted = fit.fit(x_hat)
    b0, b0_err, b1, b1_err = f_params

    # Fit target
    x_fit, x_fit_err = fit.inverse_fit(fit_target)

    print "UNWEIGTHED LINE FIT"
    print "Numpy polyfit:"
    print _fit_var_printer("a", polyfit1[0], polyfit_err[0])
    print _fit_var_printer("b", polyfit1[1], polyfit_err[1])

    print "SciPy curve_fit:"
    print _fit_var_printer("a", pol1[0], polcov1[0, 0])
    print _fit_var_printer("b", pol1[1], polcov1[1, 1])
    print "Extraction point x0 value: ", y_scipy[0][0], ((y_scipy[1][-1] - y_scipy[1][0])/2.0)[0]

    print "lfit.LineFit:"
    print _fit_var_printer("a", b1, b1_err)
    print _fit_var_printer("b", b0, b0_err)
    print "Goodness of fit: %f" % chi_unweighted
    # print "b = {0:<.10f} +/- {1:<.10f}".format(b0, self.b0_err)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax1.axhline(fit_target, linestyle="dashed", color="tab:grey")
    ax1.plot(x_hat, y_hat, label="Unweighted fit", color="tab:blue")
    ax1.fill_between(x_hat, y_hat_err[0], y_hat_err[1], alpha=0.5,
                     color="tab:blue")
    ax1.errorbar(x, signal_mean, yerr=signal_err, marker="o", label="Signal",
                 linestyle="none", color="tab:orange")
    ax1.set_ylim(0, 5)
    ax1.axvline(x_fit, color="tab:orange")
    ax1.fill_betweenx(np.linspace(0, 6, 100), x_fit - x_fit_err, x_fit + x_fit_err,
                      label=r"$x_0\pm\sigma_{x_0}$", alpha=0.5, color="tab:orange")
    ax1.legend(loc="best", prop={"size": 8})
    ax1.set_title("Fit test")

    # Weighted curve_fit
    print "WEIGTHED LINE FIT"

    # Numpy polyfit
    polyfit1, polyfitcov1 = np.polyfit(x, signal_mean, 1, cov=True,
                                       w=1/signal_err)
    polyfit_err = np.sqrt(np.diag(polyfitcov1))
    print "Numpy polyfit:"
    print _fit_var_printer("a", polyfit1[0], polyfit_err[0])
    print _fit_var_printer("b", polyfit1[1], polyfit_err[1])

    # SciPy curve fit
    polw, polcovw = sciopt.curve_fit(lambda x, a, b: x*a + b, x, signal_mean,
                                     sigma=signal_err)
    scipy_lfit_w = lfit.LineFit(x, signal_mean, y_err=signal_err)
    scipy_lfit_w.set_fit_parameters(polw[1], np.sqrt(polcovw[1, 1]), polw[0],
                                    np.sqrt(polcovw[0, 0]), weighted=True)
    y_scipy_w = scipy_lfit_w(fit_target, weighted=True)
    print "SciPy curve_fit:"
    print _fit_var_printer("a", polw[0], polcovw[0, 0])
    print _fit_var_printer("b", polw[1], polcovw[1, 1])
    print "Extraction point x0 value: ", y_scipy_w[0][0], \
        ((y_scipy_w[1][-1] - y_scipy_w[1][0])/2.0)[0]

    signal_cov = _get_covariance_matrix_from_raw(signal.T)

    polwc, polcovwc = sciopt.curve_fit(lambda _x, _a, _b: _x*_a + _b, x,
                                       signal_mean, sigma=signal_cov)
    scipy_lfit_wc = lfit.LineFit(x, signal_mean, y_err=signal_err)
    scipy_lfit_wc.set_fit_parameters(polwc[1], np.sqrt(polcovwc[1, 1]),
                                     polwc[0], np.sqrt(polcovwc[0, 0]), weighted=True)

    _a, _a_err = scipy_lfit_wc.b1w, scipy_lfit_wc.b1w_err
    _b, _b_err = scipy_lfit_wc.b0w, scipy_lfit_wc.b0w_err
    # y = a*x + b
    # dy = da*x + db
    # dy**2 = np.sqrt((da*x)**2 + db**2 + 2*da*x*db)
    _y = _a*fit_target + _b
    _y_err = np.sqrt(
        (_a_err*fit_target)**2 + (_b_err)**2 + 2*_a_err*fit_target*_b_err)
    y_scipy_wc = [_y, [_y-_y_err, _y+_y_err]]

    y_scipy_wc = scipy_lfit_wc(fit_target, weighted=True)

    print "SciPy curve_fit with covariance:"
    print _fit_var_printer("a", polwc[0], polcovwc[0, 0])
    print _fit_var_printer("b", polwc[1], polcovwc[1, 1])
    print "Extraction point x0 value: ", y_scipy_wc[0], \
        ((y_scipy_wc[1][-1] - y_scipy_wc[1][0])/2.0)

    # Weighted lfit.LineFit
    yw_hat, yw_hat_err, f_params_weighted, chi_weighted = fit.fit_weighted(
        x_hat)
    b0, b0_err, b1, b1_err = f_params_weighted
    xw_fit, xw_fit_error = fit.inverse_fit(fit_target, weighted=True)
    print "lfit.LineFit:"
    print _fit_var_printer("a", b1, b1_err)
    print _fit_var_printer("b", b0, b0_err)
    print "Goodness of fit: %f" % chi_weighted

    ax2 = fig1.add_subplot(212)
    ax2.axhline(fit_target, linestyle="dashed", color="tab:grey")
    ax2.errorbar(x, signal_mean, yerr=signal_err, marker="o", label="Signal",
                 linestyle="none", color="tab:orange")
    ax2.plot(x_hat, yw_hat, label="Weighted fit", color="tab:blue")
    ax2.fill_between(x_hat, yw_hat_err[0], yw_hat_err[1], alpha=0.5,
                     color="tab:blue")
    ax2.set_ylim(0, 5)
    ax2.axvline(xw_fit, color="tab:orange")
    ax2.fill_betweenx(np.linspace(0, 6, 100), xw_fit - xw_fit_error,
                      xw_fit + xw_fit_error, label=r"$x_{0,w}\pm\sigma_{x_0,w}$",
                      alpha=0.5, color="tab:orange")
    ax2.legend(loc="best", prop={"size": 8})
    fig1.savefig("tests/line_fit_example.pdf", dpi=400)
    plt.show()
    plt.close(fig1)


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
    N = 10  # Number of data points
    M = 40  # Number of observations at each data point
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

    def _f(_x, a, b):
        return _x*a + b

    from autocorrelation import Autocorrelation

    # print help(Autocorrelation)

    _ac_array = np.zeros((N, M/2))
    _tau_ints = np.zeros(N)

    print signal.shape, M

    for i in xrange(N):
        ac = Autocorrelation(signal[:, i])
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

    cov_mat = np.cov(signal.T)
    print cov_mat.shape

    for i in xrange(N):
        cov_mat[i, i] *= 2*_tau_ints[i]

    l = lfit.LineFit(x, signal_mean, y_err=signal_err)

    signal_err_corrected = np.sqrt(np.diag(cov_mat))
    print "signal_err_corrected: ", signal_err_corrected[:5]

    def get_err(_err): return (_err[1] - _err[0])/2

    pol1, polcov1 = np.polyfit(
        x, signal_mean, 1, w=1/np.diag(cov_mat), cov=True)
    l.set_fit_parameters(pol1[1], np.sqrt(polcov1[1, 1]), pol1[0],
                         np.sqrt(polcov1[0, 0]), weighted=True)
    y_hat, y_hat_err, chi_squared = l(X_values, weighted=True)
    print "With polyfit:            ", np.sqrt(np.diag(polcov1)), \
        get_err(y_hat_err)[:5]

    pol1, polcov1 = scipy.optimize.curve_fit(_f, x, signal_mean,
                                             sigma=np.sqrt(np.diag(cov_mat)))
    l.set_fit_parameters(pol1[1], np.sqrt(polcov1[1, 1]), pol1[0],
                         np.sqrt(polcov1[0, 0]), weighted=True)
    y_hat, y_hat_err, chi_squared = l(X_values, weighted=True)
    print "With naive autocorr:     ", np.sqrt(np.diag(polcov1)), \
        get_err(y_hat_err)[:5]

    pol1, polcov1 = sciopt.curve_fit(
        _f, x, signal_mean, sigma=np.cov(signal.T))
    l.set_fit_parameters(pol1[1], np.sqrt(polcov1[1, 1]), pol1[0],
                         np.sqrt(polcov1[0, 0]), weighted=True)
    y_hat, y_hat_err, chi_squared = l(X_values, weighted=True)
    print "With cov(signal.T):      ", np.sqrt(np.diag(polcov1)), \
        get_err(y_hat_err)[:5]

    pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=signal_err)
    l.set_fit_parameters(pol1[1], np.sqrt(polcov1[1, 1]), pol1[0],
                         np.sqrt(polcov1[0, 0]), weighted=True)
    y_hat, y_hat_err, chi_squared = l(X_values, weighted=True)
    print "With only signal errors: ", np.sqrt(np.diag(polcov1)), \
        get_err(y_hat_err)[:5]


def _test_bootstrap(x, y, x0, _f=lambda _x, a, b: _x*a + b, N_bs=50):
    """Bootstrap example"""
    x_hat = x[0, :]
    y_mean, y_err = y.mean(axis=0), y.std(axis=0)

    # M, N = y.shape
    # random_fit_indexes = np.random.randint(0, M, size=(N_bs, M, N))
    # y_bs = np.zeros((N_bs, N))
    # for i_bs in xrange(N_bs):
    #     for j in xrange(N):
    #         y_bs[i_bs,j] = y[random_fit_indexes[i_bs,:,j], j].mean()

    tau_int, tau_int_err = np.ones(len(x_hat))*0.5, np.zeros(len(x_hat))+0.001

    # Gets values to plot
    # bs_mean, bs_err = np.zeros(len(x_hat)), np.zeros(len(x_hat))
    # for i, _x0 in enumerate(x_hat):
    #     res = _extract_bootstrap_fit(_x0, _f, x_hat, y_mean, y_err, y_bs.T, tau_int,
    #         tau_int_err)
    #     bs_mean[i] = res[0]
    #     bs_err[i] = res[1]

    y0, y0err, _, bs_chi2, bs_x, bs_mean, bs_err = \
        _extract_bootstrap_fit(x0, _f, x_hat, y_mean, y_err, y.T, tau_int,
                               tau_int_err, plot_samples=True, full=True)

    # bs_mean, bs_err = np.zeros(len(x_hat)), np.zeros(len(x_hat))
    # for i, _x0 in enumerate(x_hat):
    #     res = _extract_bootstrap_fit(_x0, _f, x_hat, y_mean, y_err, y.T, tau_int,
    #         tau_int_err, plot_samples=True)
    #     bs_mean[i] = res[0]
    #     bs_err[i] = res[1]

    # skaff en kontinuerlig linje her!

    # return_value[i] = _extract_bootstrap_fit(_x0, _f, x_hat, y_mean, y_err, y.T, tau_int,
    #     tau_int_err)

    fig_bs = plt.figure()
    ax_bs = fig_bs.add_subplot(111)

    # Sets up bs std edges
    ax_bs.fill_between(bs_x, bs_mean-bs_err, bs_mean+bs_err, alpha=0.5,
                       color="tab:blue")

    # Plots bs mean
    ax_bs.plot(bs_x, bs_mean, label="Bootstrap fit", color="tab:blue")

    # Plots original data with error bars
    ax_bs.errorbar(x_hat, y_mean, yerr=y_err, marker=".", linestyle="none",
                   color="tab:orange", label=r"Signal")
    ax_bs.set_title(r"$\chi^2=%g$" % bs_chi2)
    ax_bs.axvline(x0, color="tab:grey", linestyle=":",
                  label=r"$x_0$ fit target")
    ax_bs.legend(loc="best")
    ax_bs.set_ylim(0, 5)
    fig_name = "tests/bootstrap_inverse_fit.pdf"
    # print "%s saved." % fig_name
    fig_bs.savefig(fig_name, dpi=400)
    # plt.show()
    plt.close(fig_bs)

    return y0, y0err


def _test_plateau(x, y, x0, _f=lambda _x, a, b: _x*a + b):
    """
    Performs a line fit through points x points to get an exact value for x0.
    """
    x_hat = x[0, :]
    y_mean, y_err = y.mean(axis=0), y.std(axis=0)

    tau_int, tau_int_err = np.ones(len(x_hat))*0.5, np.zeros(len(x_hat))+0.001

    _extract_plateau_fit

    # Gets values to plot
    x0_plot = np.linspace(x_hat[0], x_hat[-1], 1000)
    y0_mean, y0_err = np.zeros(1000), np.zeros(1000)
    for i, _x0 in enumerate(x0_plot):
        res = _extract_plateau_fit(_x0, _f, x_hat, y_mean, y_err, y.T, tau_int,
                                   tau_int_err)
        y0_mean[i] = res[0]
        y0_err[i] = res[1]

    # Gets values to find chi^2
    y0_mean_chi2 = np.zeros(len(x_hat))
    for i, _x0 in enumerate(x_hat):
        res = _extract_plateau_fit(_x0, _f, x_hat, y_mean, y_err, y.T, tau_int,
                                   tau_int_err)
        y0_mean_chi2[i] = res[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x0_plot, y0_mean, label="Line fit", color="tab:blue")
    ax.fill_between(x0_plot, y0_mean-y0_err, y0_mean+y0_err, alpha=0.5,
                    color="tab:blue")
    ax.errorbar(x_hat, y_mean, yerr=y_err, marker=".", linestyle="none",
                color="tab:orange", label=r"Signal")
    ax.set_ylim(0, 5)
    ax.set_title(r"$\chi^2=%g$" %
                 lfit.LineFit.chi_squared(y_mean, y_err, y0_mean_chi2))
    ax.axvline(x0, color="tab:grey", linestyle=":", label=r"$x_0$ fit target")
    ax.legend(loc="best", prop={"size": 8})
    fig.savefig("tests/plateau_inverse_fit.pdf", dpi=400)
    # plt.show()
    plt.close(fig)

    return _extract_plateau_fit(x0, _f, x_hat, y_mean, y_err, y.T, tau_int,
                                tau_int_err)[:2]


def _test_plateau_mean(x, y, x0, _f=lambda _x, a, b: _x*a + b):
    """
    Performs a line fit through points x points to get an exact value for x0.
    """
    x_hat = x[0, :]
    y_mean, y_err = y.mean(axis=0), y.std(axis=0)

    # Line fit from the mean values and their standard deviations
    pol_line, polcov_line = sciopt.curve_fit(_f, x_hat, y_mean, sigma=y_err,
                                             absolute_sigma=False)
    pol_line_err = np.sqrt(np.diag(polcov_line))
    x_fit_arr = np.linspace(x_hat[0], x_hat[-1], 1000)
    # Extract fit target values
    lfit_default = lfit.LineFit(x_hat, y_mean, y_err)
    lfit_default.set_fit_parameters(pol_line[1], pol_line_err[1],
                                    pol_line[0], pol_line_err[0], weighted=True)
    y0, y0_error, chi_squared = lfit_default(x0, weighted=True)
    y0_error = ((y0_error[1] - y0_error[0])/2)

    y_fit_mean, y_fit_err, _ = lfit_default(x_fit_arr, weighted=True)
    y_fit_err = ((y_fit_err[1] - y_fit_err[0])/2.0)

    y_hat, _, _ = lfit_default(x_hat, weighted=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_fit_arr, y_fit_mean, label="Line fit", color="tab:blue")
    ax.fill_between(x_fit_arr, y_fit_mean-y_fit_err,
                    y_fit_mean+y_fit_err, alpha=0.5, color="tab:blue")
    ax.errorbar(x_hat, y_mean, yerr=y_err, marker=".", linestyle="none",
                color="tab:orange", label=r"Signal")
    ax.set_ylim(0, 5)
    ax.set_title(r"$\chi^2=%g$" %
                 lfit.LineFit.chi_squared(y_mean, y_err, y_hat))
    ax.axvline(x0, color="tab:grey", linestyle=":", label=r"$x_0$ fit target")
    ax.legend(loc="best", prop={"size": 8})
    fig.savefig("tests/plateau_mean_inverse_fit.pdf", dpi=400)
    # plt.show()
    plt.close(fig)

    return _extract_plateau_mean_fit(x0, _f, x_hat, y_mean, y_err)[:2]


def _test_interpolation(x, y, x0, _f=lambda _x, a, b: _x*a + b, k_spline=1):
    """Spline fit."""
    x_hat = x[0, :]
    x_fit_arr = np.linspace(x_hat[0], x_hat[-1], 1000)
    signal_mean, signal_err = y.mean(axis=0), y.std(axis=0)
    s = spline(x_hat, signal_mean, w=1/signal_err, k=1)

    spl = lfit.ErrorPropagationSpline(
        x_hat, signal_mean, signal_err, k=k_spline)
    spline_mean, spline_err = spl(x_fit_arr)
    spl_fit, spl_err = spl(x_hat)

    fig_spl = plt.figure()
    ax_spl = fig_spl.add_subplot(111)
    ax_spl.plot(x_fit_arr, spline_mean, label="Spline interpolation fit",
                color="tab:blue")
    ax_spl.fill_between(x_fit_arr, spline_mean-spline_err,
                        spline_mean+spline_err, alpha=0.5, color="tab:blue")
    ax_spl.errorbar(x_hat, signal_mean, yerr=signal_err, marker=".",
                    linestyle="none", color="tab:orange", label=r"Signal")
    ax_spl.set_ylim(0, 5)
    ax_spl.set_title(r"$\chi^2=%g$" %
                     lfit.LineFit.chi_squared(signal_mean, signal_err, spl_fit))
    ax_spl.axvline(x0, color="tab:grey", linestyle=":",
                   label=r"$x_0$ fit target")
    ax_spl.legend(loc="best", prop={"size": 8})
    fig_spl.savefig("tests/spline_inverse_fit.pdf", dpi=400)
    # plt.show()
    plt.close(fig_spl)

    x0_mean, x0_std = spl(x0)
    return x0_mean[0], x0_std[0]


def _test_nearest(x, y, x0, _f=lambda _x, a, b: _x*a + b):
    x_hat = x[0, :]
    fit_index = np.argmin(np.abs(x_hat - x0))
    y_mean, y_err = y.mean(axis=0), y.std(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x_hat, y_mean, yerr=y_err, marker=".",
                linestyle="none", color="tab:orange", label=r"Signal")
    ax.set_ylim(0, 5)
    ax.axvline(x0, color="tab:grey", linestyle=":",
               label=r"$x_0$ fit target")

    # Plots hline to illustrate error bars
    ax.axhline(y_mean[fit_index], color="tab:blue",
               label=r"$y(x_0)$ value estimation")
    err_upper = (y_mean[fit_index]+y_err[fit_index])*np.ones(len(x_hat))
    err_lower = (y_mean[fit_index]-y_err[fit_index])*np.ones(len(x_hat))
    ax.fill_between(x_hat, err_lower, err_upper, alpha=0.5, color="tab:blue")
    ax.set_title("Nearest value")
    ax.legend(loc="best")
    fig.savefig("tests/nearest_inverse_fit.pdf", dpi=400)
    # plt.show()
    plt.close(fig)

    return y_mean[fit_index], y_err[fit_index]


def _test_fit_methods():
    """Tests different methods of extracting an exact fit value at x0."""

    # Generates signal for a line with cosine gaussian noise
    x_start = 0
    x_end = 5
    a = 0.65
    b = 1.1
    N = 5  # Number of data points
    M = 20  # Number of observations at each data point
    x = np.linspace(x_start, x_end, N)
    x_matrix = np.ones((M, N)) * x
    signal_spread = 1.7
    np.random.seed(1)
    signal = np.random.uniform(-signal_spread, signal_spread, (M, N))
    # signal = np.cos(np.random.uniform(-signal_spread, signal_spread, (M, N)))
    signal += a*x_matrix + b

    x0 = x_end / 2

    def tab_print(name, values):
        y0, y0err = values
        print "%s & $%.2f$ & $%.2f$ \\\\" % (name, y0, y0err)

    tab_print("Nearest", _test_nearest(x_matrix, signal, x0))
    tab_print("Interpolation", _test_interpolation(x_matrix, signal, x0,
                                                   k_spline=3))
    tab_print("Bootstrap", _test_bootstrap(x_matrix, signal, x0))
    tab_print("Plateau mean", _test_plateau_mean(x_matrix, signal, x0))
    tab_print("Plateau", _test_plateau(x_matrix, signal, x0))

    # print "Bootstrap fit: ", _test_bootstrap(x_matrix, signal, x0)
    # print "Spline interpolation: ",   _test_interpolation(x_matrix, signal, x0,
    #   k_spline=3)
    # print "Plateau mean linefit: ", _test_plateau_mean(x_matrix, signal, x0)
    # print "Nearest: ", _test_nearest(x_matrix, signal, x0)
    # print "Plateau linefit: ", _test_plateau(x_matrix, signal, x0)


def main():
    _test_fit_methods()
    # _test_simple_line_fit()
    # _test_inverse_line_fit()


if __name__ == '__main__':
    main()
