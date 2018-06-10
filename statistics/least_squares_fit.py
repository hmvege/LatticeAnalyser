import scipy.io as sio
import numpy as np
import scipy.optimize as sciopt
import scipy.linalg as scilin
import types

def _get_covariance_matrix_from_raw(y_raw, autocorr=None, iscov=False):
    """
    Returns a covariance matrix that is guaranteed to not be singular.
    """
    if isinstance(autocorr, types.NoneType):
        autocorr = np.eye(len(y_raw))
    else:
        autocorr = np.eye(len(y_raw))*autocorr

    # Uses bootstrap, jackknifed or analyzed values directly.
    if not iscov:
        cov_raw = np.cov(y_raw)
    else:
        cov_raw = y_raw

    for i in xrange(len(cov_raw)):
        cov_raw[i,i] *= autocorr[i,i]

    # Get eigenvalues for covariance matrix
    eig = np.linalg.eigvals(cov_raw)

    counter = 1

    while np.min(eig) <= 1e-15:
        # Gets the magnitude of the smallest eigenvalue
        magnitude = np.floor(np.log10(np.absolute(np.min(eig))))
        # Increments magnitude til we have positive definite cov-matrix
        eps = 10**(magnitude + counter)
        eps_matrix = np.eye(cov_raw.shape[0])*eps

        # Adds a small diagonal epsilon to make it positive definite
        cov_raw += eps_matrix

        eig = np.linalg.eigvals(cov_raw)

        # In order no to get stuck at a particular place
        counter += 1
        if counter >= 10:
            raise ValueError("Exceeding maximum iteration 10.")

    return cov_raw

################################################################################################################
################################################ DATA-TESTS ####################################################
################################################################################################################

data = sio.loadmat("../../../MATLAB-TEST/line_fit_data.mat")

x = data["x"][0]
y = data["y"][0]
y_err = data["y_err"][0]
y_raw = data["y_raw"]
tau_int = data["tau_int"][0]

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

res = sciopt.curve_fit(f, x, y, sigma=y_err, p0=[0.18, 0.0], maxfev=1200) 
print "y_err: ", res[0], np.sqrt(np.diag(res[1]))

cov1 = _get_covariance_matrix_from_raw(y_raw)
pol_raw1, polcov_raw1 = sciopt.curve_fit(f, x, y, sigma=cov1, p0=[0.18, 0.0], maxfev=1200) 
pol_raw_err1 = np.sqrt(np.diag(polcov_raw1))
print "cov(y):", pol_raw1, pol_raw_err1

cov2 = _get_covariance_matrix_from_raw(V, iscov=True)
pol_raw2, polcov_raw2 = sciopt.curve_fit(f, x, y, sigma=cov2, p0=[0.18, 0.0], maxfev=1200) 
pol_raw_err2 = np.sqrt(np.diag(polcov_raw2))
print "V:     ", pol_raw2, pol_raw_err2

import matplotlib.pyplot as plt
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.errorbar(x, y, yerr=y_err, label="Original", color="tab:blue")

y1 = f(x, pol_raw1[0], pol_raw1[1])
y1_err = f_err(x, pol_raw_err1[0], pol_raw_err1[1])
ax1.plot(x, y1, label=r"Method 1: $cov(y_{raw})$", color="tab:green")
ax1.fill_between(x, y1-y1_err, y1+y1_err, alpha=0.5, color="tab:green")

y2 = f(x, pol_raw2[0], pol_raw2[1])
y2_err = f_err(x, pol_raw_err2[0], pol_raw_err2[1])
ax1.plot(x, y2, label=r"Method 2: $V$", color="tab:red")
ax1.fill_between(x, y2-y2_err, y2+y2_err, alpha=0.5, color="tab:red")
plt.legend()


print f(0.6, pol_raw1[0], pol_raw1[1]), f_err(0.6, pol_raw_err1[0], pol_raw_err1[1])
print f(0.6, pol_raw2[0], pol_raw2[1]), f_err(0.6, pol_raw_err2[0], pol_raw_err2[1])

print y[0]
print y1[0]
print y2[0]


# plt.show()

# # Extract fit target values
# lfit_raw = lfit.LineFit(x, y, y_err)
# lfit_raw.set_fit_parameters(pol_raw[1], pol_raw_err[1], pol_raw[0],
#     pol_raw_err[0], weighted=True)
# y0, y0_error, _, chi_squared = lfit_raw.fit_weighted(fit_target)

# print "params: ", lfit_raw.b0w, lfit_raw.b0w_err, lfit_raw.b1w, lfit_raw.b1w_err


