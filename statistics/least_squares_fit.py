#!/usr/bin/env python2

import scipy.io as sio
import numpy as np
import scipy.optimize as sciopt
import scipy.stats as scista
import types
import linefit as lfit
import linefit_tools as lftools
import matplotlib.pyplot as plt
import copy as cp

# import statsmodels
from statsmodels.stats.correlation_tools import cov_nearest

def nearPSD(A,epsilon=0):
   n = A.shape[0]
   eigval, eigvec = np.linalg.eig(A)
   val = np.matrix(np.maximum(eigval,epsilon))
   vec = np.matrix(eigvec)
   T = 1/(np.multiply(vec,vec) * val.T)
   T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
   B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
   out = B*B.T
   return(out)

def near_psd(x, epsilon=0):
    '''
    Calculates the nearest postive semi-definite matrix for a correlation/covariance matrix

    Parameters
    ----------
    x : array_like
      Covariance/correlation matrix
    epsilon : float
      Eigenvalue limit (usually set to zero to ensure positive definiteness)

    Returns
    -------
    near_cov : array_like
      closest positive definite covariance/correlation matrix

    Notes
    -----
    Document source
    http://www.quarchome.org/correlationmatrix.pdf

    '''

    if min(np.linalg.eigvals(x)) > epsilon:
        return x

    # Removing scaling factor of covariance matrix
    n = x.shape[0]
    var_list = np.array([np.sqrt(x[i,i]) for i in xrange(n)])
    y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])

    # getting the nearest correlation matrix
    eigval, eigvec = np.linalg.eig(y)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    near_corr = B*B.T    

    # returning the scaling factors
    near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])
    return near_cov


################################################################################################################
################################################ DATA-TESTS ####################################################
################################################################################################################

# data = sio.loadmat("../../../MATLAB-TEST/line_fit_data4.mat")
# data = sio.loadmat("../../../MATLAB-TEST/line_fit_data2.mat")

x = np.load("../../x.npy")
y = np.load("../../y.npy")
y_err = np.load("../../y_err.npy")
y_raw = np.load("../../y_raw.npy")
tau_int = np.load("../../tau_int.npy")

# x = data["x"][0]
# y = data["y"][0]
# y_err = data["y_err"][0]
# y_raw = data["y_raw"]
# tau_int = data["tau_int"][0]
fit_target = 0.6


y_raw_1 = y_raw

print "DATA SHAPES: ", x.shape, y.shape, y_err.shape, y_raw.shape, tau_int.shape
# print data["covmat"].shape

N, M = y_raw.shape
print "Data shape:", N, M
print "R =", (np.mean(x*y) - np.mean(x)*np.mean(y)) / (np.std(x)*np.std(y))

V = np.zeros((N, N))

# Hardcoded covariance differs from the numpy implementation
for i in xrange(N):
    xi_mean = y_raw[i,:].mean()
    for j in xrange(N):
        xj_mean = y_raw[j,:].mean()
        V[i,j] = np.mean(y_raw[i,:]*y_raw[j,:]) - xj_mean*xi_mean


f = lambda _x, a, b: _x*a + b
f_err = lambda _x, a_err, b_err: np.sqrt((_x*a_err)**2 + (b_err)**2 + 2*a_err*_x*b_err)

slope, intercept, r_value, p_value, std_err = scista.linregress(x, y)
# print slope, intercept, r_value, p_value, std_err
yy = f(x, slope, intercept)

pol_raw0, polcov_raw0 = sciopt.curve_fit(f, x, y, sigma=y_err, p0=[intercept, slope], maxfev=1200) 
pol_raw_err0 = np.sqrt(np.diag(polcov_raw0))

# cov1 = lftools._get_covariance_matrix_from_raw(y_raw)
# cov1 = cov_nearest(np.cov(y_raw), method="nearest", threshold=9e-16, n_fact=500)
# cov1 = near_psd(np.cov(y_raw))
# cov1 = nearPSD(np.cov(y_raw))
cov1 = cov_nearest(np.cov(y_raw), method="nearest")
# print np.linalg.eigvals(cov1).min()
pol_raw1, polcov_raw1 = sciopt.curve_fit(f, x, y, sigma=cov1, p0=[intercept, slope], maxfev=2000, ftol=1e-16, epsfcn=1e-10, xtol=1e-10)
pol_raw_err1 = np.sqrt(np.diag(polcov_raw1))

# cov2 = lftools._get_covariance_matrix_from_raw(V, iscov=True)
cov2 = cov_nearest(V, method="nearest", threshold=9e-16, n_fact=500)
pol_raw2, polcov_raw2 = sciopt.curve_fit(f, x, y, sigma=cov2, p0=[intercept, slope], maxfev=1200) 
pol_raw_err2 = np.sqrt(np.diag(polcov_raw2))

print slope, pol_raw0[0]
print slope, pol_raw1[0]
print slope, pol_raw2[0]

# Extract fit target values
lfit_raw = lfit.LineFit(x, y, y_err)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.errorbar(x, y, yerr=y_err, label="Original", color="tab:blue")

lfit_raw.set_fit_parameters(pol_raw0[1], pol_raw_err0[1], pol_raw0[0],
    pol_raw_err0[0], weighted=True)
# y1, _, _ = lfit_raw(x, weighted=True)
y0 = f(x, pol_raw0[0], pol_raw0[1])
y0_err = f_err(x, pol_raw_err0[0], pol_raw_err0[1])
ax1.plot(x, y0, label=r"Method 0: $w=y_{err}$", color="tab:brown")
ax1.fill_between(x, y0-y0_err, y0+y0_err, alpha=0.5, color="tab:brown")

lfit_raw.set_fit_parameters(pol_raw1[1], pol_raw_err1[1], pol_raw1[0],
    pol_raw_err1[0], weighted=True)
# y1, _, _ = lfit_raw(x, weighted=True)
y1 = f(x, pol_raw1[0], pol_raw1[1])
y1_err = f_err(x, pol_raw_err1[0], pol_raw_err1[1])
ax1.plot(x, y1, label=r"Method 1: $cov(y_{raw})$", color="tab:green")
ax1.fill_between(x, y1-y1_err, y1+y1_err, alpha=0.5, color="tab:green")

lfit_raw.set_fit_parameters(pol_raw2[1], pol_raw_err2[1], pol_raw2[0],
    pol_raw_err2[0], weighted=True)
# y2, _, _ = lfit_raw(x, weighted=True)
y2 = f(x, pol_raw2[0], pol_raw2[1])
y2_err = f_err(x, pol_raw_err2[0], pol_raw_err2[1])
ax1.plot(x, y2, label=r"Method 2: $V$", color="tab:red")
ax1.fill_between(x, y2-y2_err, y2+y2_err, alpha=0.5, color="tab:red")


ax1.plot(x, yy, label=r"Linregress", color="tab:cyan")

plt.legend()

print "y_err:  ", pol_raw0, pol_raw_err0
print "y_err:  ", f(fit_target, pol_raw0[0], pol_raw0[1]), f_err(fit_target, pol_raw_err0[0], pol_raw_err0[1])

print "cov(y): ", pol_raw1, pol_raw_err1
print "cov(y): ", f(fit_target, pol_raw1[0], pol_raw1[1]), f_err(fit_target, pol_raw_err1[0], pol_raw_err1[1])

print "V:      ", pol_raw2, pol_raw_err2
print "V:      ", f(fit_target, pol_raw2[0], pol_raw2[1]), f_err(fit_target, pol_raw_err2[0], pol_raw_err2[1])

lftools.__plot_fit_target(x, y, y_err, fit_target, f(fit_target, pol_raw1[0], pol_raw1[1]), f_err(fit_target, pol_raw_err1[0], pol_raw_err1[1]))

# print y[0]
# print y1[0]
# print y2[0]

print "FROM MAIN PROG:"
print "[0.00809165 0.18245169] [0.00032281 0.00072717]"
print "params:  [0.008091649328135688, 0.18245168951426122] [0.0003228066628403942, 0.0007271702456212646]"
print "Plateau fit:   0.1873066791111426 +/- 0.0009208542433255  , chi^2 0.00280592"

