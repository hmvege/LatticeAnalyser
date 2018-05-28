#!/usr/bin/env python2

from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.optimize as sciopt
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import types

def _extract_inverse(fit_target X, Y, Y_err):
    """
    Simple method for finding the x axis value to a target value.

    Returns: 
        min(fit_target - X)
        error(min(fit_target - X))
    """
    # Finds the target value
    min_index = np.argmin(np.abs(fit_target - Y))
    x0 = X[min_index]

    # Finds the error bands
    x_err_neg, x_err_pos = Y_err
    x0_err_index_neg = np.argmin(np.abs(fit_target - x_err_pos))
    x0_err_index_pos = np.argmin(np.abs(fit_target - x_err_neg))
    x0_err = [X[x0_err_index_neg], X[x0_err_index_pos]]
    return x0, x0_err

def _get_covariance_matrix_from_raw(y_raw):
    """
    Returns a covariance matrix that is guaranteed to not be singular.
    """

    # Uses bootstrap, jackknifed or analyzed values directly.
    cov_raw = np.cov(y_raw)

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

    return cow_raw

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

def _extract_plateau_fit(fit_target, f, x, y, y_err, y_raw, tau_int, 
    tau_int_err, inverse_fit=False):
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
        tau_int: numpy float array, autocorrelation times.
        tau_int_err: numpy float array, autocorrelation time errors.
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

    cow_raw = _get_covariance_matrix_from_raw(y_raw)

    # Line fit from the mean values and the raw values covariance matrix
    pol_raw, polcov_raw = sciopt.curve_fit(f, x, y, sigma=cov_raw,
        absolute_sigma=False, p0=[0.18, 0.0], maxfev=1200)
    pol_raw_err = np.sqrt(np.diag(polcov_raw))

    # Extract fit target values
    lfit_raw = LineFit(x, y, y_err)
    lfit_raw.set_fit_parameters(pol_raw[1], pol_raw_err[1], pol_raw[0],
        pol_raw_err[0], weighted=True)
    y0, y0_error, _, chi_squared = lfit_raw.fit_weighted(fit_target)

    # Errors should be equal in positive and negative directions.
    if np.abs(np.abs(y0 - y0_error[0]) - np.abs(y0 - y0_error[1])) > 1e-15:
        print "Warning: uneven errors:\nlower: %.10f\nupper: %.10f" % (
            y0_error[0], y0_error[1])

    y0 = y0[0] #  y0 and y0_error both comes in form of arrays
    y0_error = ((y0_error[1] - y0_error[0])/2)
    y0_error = y0_error[0]

    # Perform line fit for tau int as well to error correct
    tau_int0 = __get_tau_int(fit_target, x, tau_int, tau_int_err)

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
    lfit_default = LineFit(x, y, y_err)
    lfit_default.set_fit_parameters(pol_line[1], pol_line_err[1], 
        pol_line[0], pol_line_err[0], weighted=True)

    if inverse_fit:
        y0, y0_error = lfit_default.inverse_fit(fit_target)
        _, _, chi_squared = lfit_default(fit_target, weighted=True)
    else:
        y0, y0_error, chi_squared = lfit_default(fit_target, weighted=True)

    y0_error = ((y0_error[1] - y0_error[0])/2)

    return y0[0], y0_error[0], chi_squared

def _extract_bootstrap_fit(fit_target, f, x, y, y_err, y_raw, tau_int, 
    tau_int_err, plot_samples=False, F=lambda _y: _y, FDer=lambda _y, 
    _yerr: _yerr):
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
        y0_sample[i], _tmp_err = fit_sample(fit_target)
        y0_sample_err[i] = (_tmp_err[1] - _tmp_err[0])/2

        if plot_samples:
            plot_ymean[:,i], _p_err = fit_sample(x)
            plot_yerr[:,i] = np.asarray(_p_err).T

            ax_samples.fill_between(x, plot_yerr[:,i,0], plot_yerr[:,i,1], 
                alpha=0.01, color="tab:red")
            ax_samples.plot(x, plot_ymean, label="Scipy curve fit",
                color="tab:red", alpha=0.1)

    # Gets the tau int
    tau_int0 = __get_tau_int(fit_target, x, tau_int, tau_int_err)

    y0_mean = F(np.mean(y0_sample, axis=0))

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

        # sample_mean = np.mean(plot_ymean, axis=1)
        # sample_std = np.std(plot_ymean, axis=1)*np.sqrt(2*tau_int0)

        # Sets up sample std edges
        ax_samples.plot(x, sample_mean - sample_std, x, 
            sample_mean + sample_std, color="tab:blue", alpha=0.6)

        # Plots sample mean
        ax_samples.plot(x, sample_mean, label="Averaged samples fit", 
            color="tab:blue")

        # Plots original data with error bars
        ax_samples.errorbar(x, y, yerr=y_err, marker=".", 
            linestyle="none", color="tab:orange", label="Original")

        ax_samples.title(r"$\chi^2: %g$" % 
            LineFit.chi_squared(y, y_err, sample_mean))

        plt.show()
        plt.close(fig_samples)

    return y0_mean, y0_std, tau_int0


def _extract_inverse_bootstrap_fit(y0, f, x, y, y_err, y_raw, tau_int, 
    tau_int_err, plot_samples=False, F=lambda _y: _y, FDer=lambda _y, 
    _yerr: _yerr):
    """
    Extract x0 with x0err at a given y0 by using line fitting the y_raw data.
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
        tau_int: numpy float array, autocorrelation times.
        tau_int_err: numpy float array, autocorrelation time errors.
        plot_bs: bool, optional. Will plot the bootstrapped line fits and show.
        F: function, optional, will modify the bootstrap data after 
            samples has been taken by this function.
        FDer: function, optional, will propagate the error of the bootstrapped 
            line fit. Should take y and yerr. Calculated by regular error 
            propagation.

    Returns:
        x0, x0_error, chi_squared
    """

    x0_sample = np.zeros(y_raw.shape[-1])
    x0_sample_err = np.zeros(y_raw.shape[-1])

    # Values for performing inverse fit with autocorrelation
    n = 100000
    x = np.linspace(np.min(x), np.max(x), n)

    # Gets the bootstrapped line fits
    # for i, y_sample in enumerate(tqdm(y_raw.T, desc="Sample line fitting")):
    for i, y_sample in enumerate(y_raw.T):
        p, pcov = sciopt.curve_fit(f, x, y_sample, p0=[0.01, 0.18])
        pfit_err = np.sqrt(np.diag(pcov))

        # Fits sample
        fit_sample = LineFit(x, y_sample)
        fit_sample.set_fit_parameters(p[1], pfit_err[1], p[0], pfit_err[0])

        # _x0_err is not needed, is error cannot be inflated by the autocorrelation.
        x0, _tmp_err = fit_sample.inverse_fit(fit_target)
        x0_sample[i] = x0
        x0_sample_err[i] = (_tmp_err[1] - _tmp_err[0])/2

    # # Gets the tau int
    # tau_int0 = __get_tau_int(fit_target, x, tau_int, tau_int_err)

    x0_mean = F(np.mean(x0_sample, axis=0))

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
    x0_std = np.std(F(y0_sample))*np.sqrt(2*tau_int0)
    # print y0_std

    print "bootstrap inverse:", x0_mean, x0_std

    return x0_mean, x0_std#, tau_int0


def __get_tau_int(x0, x, tau_int, tau_int_err):
    """Smal internal function for getting tau int at x0."""
    tauFit = LineFit(x, tau_int, y_err=tau_int_err)
    tau_int0, _, _, _ = tauFit.fit_weighted(x0)
    return tau_int0[0]


def _test_simple_line_fit():
    """
    Function for testing the case where one compares the curve_fit module in
    scipy.optimize with what I have created.

    SciPy and LineFit should fit exact down to around 8th digit.
    """
    import random
    import scipy.optimize as sciopt

    def _fit_var_printer(var_name, var, var_error, w=16):
        return "{0:<s} = {1:<.{w}f} +/- {2:<.{w}f}".format(var_name, var, 
            var_error, w=w)

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
    fig1.savefig("tests/line_fit_example.png", dpi=400)
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

    def _f(_x, a, b):
        return _x*a + b

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

    cov_mat = np.cov(signal.T)
    print cov_mat.shape

    for i in xrange(N):
        cov_mat[i,i] *= 2*_tau_ints[i]

    lfit = LineFit(x, signal_mean, y_err=signal_err)

    signal_err_corrected = np.sqrt(np.diag(cov_mat))
    print "signal_err_corrected: ", signal_err_corrected[:5]

    get_err = lambda _err: (_err[1] - _err[0])/2

    pol1, polcov1 = np.polyfit(x, signal_mean, 1, w=1/np.diag(cov_mat), cov=True)
    lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0], 
        np.sqrt(polcov1[0,0]), weighted=True)
    y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
    print "With polyfit:            ", np.sqrt(np.diag(polcov1)), \
        get_err(y_hat_err)[:5]
    
    pol1, polcov1 = scipy.optimize.curve_fit(_f, x, signal_mean, 
        sigma=np.sqrt(np.diag(cov_mat)))
    lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0],
        np.sqrt(polcov1[0,0]), weighted=True)
    y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
    print "With naive autocorr:     ", np.sqrt(np.diag(polcov1)), \
        get_err(y_hat_err)[:5]

    pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=np.cov(signal.T))
    lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0],
        np.sqrt(polcov1[0,0]), weighted=True)
    y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
    print "With cov(signal.T):      ", np.sqrt(np.diag(polcov1)), \
        get_err(y_hat_err)[:5]
    
    pol1, polcov1 = sciopt.curve_fit(_f, x, signal_mean, sigma=signal_err)
    lfit.set_fit_parameters(pol1[1], np.sqrt(polcov1[1,1]), pol1[0], 
        np.sqrt(polcov1[0,0]), weighted=True)
    y_hat, y_hat_err, chi_squared = lfit(X_values, weighted=True)
    print "With only signal errors: ", np.sqrt(np.diag(polcov1)), \
        get_err(y_hat_err)[:5]

def _test_bootstrap(x, y, x0, _f=lambda _x, a, b: _x*a + b, N_bs=50):
    """Bootstrap example"""
    x_hat = x[0,:]
    y_mean, y_err = y.mean(axis=0), y.std(axis=0)

    M, N = y.shape
    random_fit_indexes = np.random.randint(0, M, size=(N_bs, M, N))
    y_bs = np.zeros((N_bs, N))
    for i_bs in xrange(N_bs):
        for j in xrange(N):
            y_bs[i_bs,j] = y[random_fit_indexes[i_bs,:,j], j].mean()

    tau_int, tau_int_err = np.ones(len(x_hat))*0.5, np.zeros(len(x_hat))+0.001

    # Gets values to plot
    bs_mean, bs_err = np.zeros(len(x_hat)), np.zeros(len(x_hat))
    for i, _x0 in enumerate(x_hat):
        res = _extract_bootstrap_fit(_x0, _f, x_hat, y_mean, y_err, y_bs.T, tau_int,
            tau_int_err)
        bs_mean[i] = res[0]
        bs_err[i] = res[1]

    fig_bs = plt.figure()
    ax_bs = fig_bs.add_subplot(111)

    # Sets up bs std edges
    ax_bs.fill_between(x_hat, bs_mean-bs_err, bs_mean+bs_err, alpha=0.5,
        color="tab:blue")

    # Plots bs mean
    ax_bs.plot(x_hat, bs_mean, label="Bootstrap fit", color="tab:blue")

    # Plots original data with error bars
    ax_bs.errorbar(x_hat, y_mean, yerr=y_err, marker=".", linestyle="none", 
        color="tab:orange", label=r"Signal")
    ax_bs.set_title(r"$\chi^2=%g$" % 
        LineFit.chi_squared(y_mean, y_err, bs_mean))
    ax_bs.axvline(x0, color="tab:grey", linestyle=":",
        label=r"$x_0$ fit target")
    ax_bs.legend(loc="best")
    fig_name = "tests/bootstrap_inverse_fit.png"
    # print "%s saved." % fig_name
    fig_bs.savefig(fig_name, dpi=400)
    # plt.show()
    plt.close(fig_bs)

    return _extract_bootstrap_fit(x0, _f, x_hat, y_mean, y_err, y_bs.T, tau_int,
            tau_int_err)[:2]

def _test_plateau(x, y, x0, _f=lambda _x, a, b: _x*a + b):
    """
    Performs a line fit through points x points to get an exact value for x0.
    """
    x_hat = x[0,:]
    y_mean, y_err = y.mean(axis=0), y.std(axis=0)

    tau_int, tau_int_err = np.ones(len(x_hat))*0.5, np.zeros(len(x_hat))+0.001

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
    ax.set_ylim(0.5, 5)
    ax.set_title(r"$\chi^2=%g$" % 
        LineFit.chi_squared(y_mean, y_err, y0_mean_chi2))
    ax.axvline(x0, color="tab:grey", linestyle=":", label=r"$x_0$ fit target")
    ax.legend(loc="best", prop={"size": 8})
    fig.savefig("tests/plateau_inverse_fit.png", dpi=400)
    # plt.show()
    plt.close(fig)  

    return _extract_plateau_fit(x0, _f, x_hat, y_mean, y_err, y.T, tau_int,
        tau_int_err)[:2]

def _test_plateau_mean(x, y, x0, _f=lambda _x, a, b: _x*a + b):
    """
    Performs a line fit through points x points to get an exact value for x0.
    """
    x_hat = x[0,:]
    y_mean, y_err = y.mean(axis=0), y.std(axis=0)

    # Line fit from the mean values and their standard deviations
    pol_line, polcov_line = sciopt.curve_fit(_f, x_hat, y_mean, sigma=y_err,
        absolute_sigma=False)
    pol_line_err = np.sqrt(np.diag(polcov_line))
    x_fit_arr = np.linspace(x_hat[0], x_hat[-1], 1000)
    # Extract fit target values
    lfit_default = LineFit(x_hat, y_mean, y_err)
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
    ax.set_ylim(0.5, 5)
    ax.set_title(r"$\chi^2=%g$" % LineFit.chi_squared(y_mean, y_err, y_hat))
    ax.axvline(x0, color="tab:grey", linestyle=":", label=r"$x_0$ fit target")
    ax.legend(loc="best", prop={"size": 8})
    fig.savefig("tests/plateau_mean_inverse_fit.png", dpi=400)
    # plt.show()
    plt.close(fig)

    return _extract_plateau_mean_fit(x0, _f, x_hat, y_mean, y_err)[:2]

def _test_interpolation(x, y, x0, _f=lambda _x, a, b: _x*a + b, k_spline=1):
    """Spline fit."""
    x_hat = x[0,:]
    x_fit_arr = np.linspace(x_hat[0], x_hat[-1], 1000)
    signal_mean, signal_err = y.mean(axis=0), y.std(axis=0)
    s = spline(x_hat, signal_mean, w=1/signal_err, k=1)

    spl = ErrorPropagationSpline(x_hat, signal_mean, signal_err, k=k_spline)
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
    ax_spl.set_ylim(0.5, 5)
    ax_spl.set_title(r"$\chi^2=%g$" % 
        LineFit.chi_squared(signal_mean, signal_err, spl_fit))
    ax_spl.axvline(x0, color="tab:grey", linestyle=":",
        label=r"$x_0$ fit target")
    ax_spl.legend(loc="best", prop={"size": 8})
    fig_spl.savefig("tests/spline_inverse_fit.png", dpi=400)
    # plt.show()
    plt.close(fig_spl)

    x0_mean, x0_std = spl(x0)
    return x0_mean[0], x0_std[0]

def _test_nearest(x, y, x0, _f=lambda _x, a, b: _x*a + b):
    x_hat = x[0,:]
    fit_index = np.argmin(np.abs(x_hat - x0))
    y_mean, y_err = y.mean(axis=0), y.std(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x_hat, y_mean, yerr=y_err, marker=".", 
        linestyle="none", color="tab:orange", label=r"Signal")
    ax.set_ylim(0.5, 5)
    ax.axvline(x0, color="tab:grey", linestyle=":",
        label=r"$x_0$ fit target")

    # Plots hline to illustrate error bars
    ax.axhline(y_mean[fit_index], color="tab:blue", label=r"$y(x_0)$ value estimation")
    err_upper = (y_mean[fit_index]+y_err[fit_index])*np.ones(len(x_hat))
    err_lower = (y_mean[fit_index]-y_err[fit_index])*np.ones(len(x_hat))
    ax.fill_between(x_hat, err_lower, err_upper, alpha=0.5, color="tab:blue")
    ax.set_title("Nearest value")
    ax.legend(loc="best")
    fig.savefig("tests/nearest_inverse_fit.png", dpi=400)
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
    N = 5 # Number of data points
    M = 20 # Number of observations at each data point
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
    from linefit import ErrorPropagationSpline, LineFit
    global ErrorPropagationSpline
    global LineFit

    _test_fit_methods()
    # _test_simple_line_fit()
    # _test_inverse_line_fit()

if __name__ == '__main__':
    main()