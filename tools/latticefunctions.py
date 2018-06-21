import numpy as np

def get_lattice_spacing(beta):
	"""
	Function for getting the lattice spacing. From paper by Guagnelli et. al., 
	Precision computation of a low-energy reference scale in quenched lattice
	LQCD, 1998.

	Args:
		beta: beta value

	Returns:
		a: lattice spacing in fermi
	"""

	r0 = 0.5
	beta_low = 5.7
	beta_high = 6.57

	if beta < beta_low or beta > beta_high:
		raise Warning("Beta value of %f is outside of defined area [%f, %f]."
			% (beta, beta_low, beta_high))

	def _get_a(b):
		"""Gets the beta value without any error."""
		bval = (b - 6.0)
		_a = np.exp(-1.6805 - 1.7139*bval + 0.8155*bval**2 - 0.6667*bval**3)*r0
		return _a

	a = _get_a(beta)
	a_err_slope = ((0.6 - 0.3)/100.0)/(beta_high - beta_low) # err% / beta
	a_err_const = 0.3/100 - a_err_slope*beta_low
	a_err_percent = lambda _b: a_err_slope*_b + a_err_const
	a_err = a*a_err_percent(beta)

	return a, a*a_err_percent(beta) # fermi

def witten_veneziano(chi, chi_error):
	"""
	Witten-Veneziano formula, https://arxiv.org/pdf/hep-th/0407052.pdf.

	Args:
		chi: topsus in MeV, float
		chi_error: topsus error in MeV, float

	Returns:
		N_f: number of flavors
		N_f_error: error in number of flavors
	"""

	chi **=4
	chi_error **= 4

	# Pi decay constant, pi^0
	# https://en.wikipedia.org/wiki/Pion_decay_constant
	F_pi = 0.130 / np.sqrt(2) # [GeV]
	F_pi_error = 0.005 / np.sqrt(2) # [GeV]

	# Eta prime mass
	eta_prime = 0.95778 # [GeV]
	eta_prime_error = 0.06*10**(-3) # [GeV]

	# Number of flavors
	N_f = F_pi**2 * eta_prime**2 / (2 * chi)
	t1 = F_pi_error * F_pi * (eta_prime**2 / chi)
	t2 = eta_prime_error * eta_prime * (F_pi**2 / chi)
	t3 = - chi_error * F_pi**2 * eta_prime**2 / (2 * chi**2)
	# N_f_error = np.sqrt(t1**2 + t2**2 + t3**2 + 2*t1*t2 + 2*t1*t3 + 2*t2*t3)
	N_f_error = np.sqrt(t1**2 + t2**2 + t3**2)

	return N_f, N_f_error

def main():
	exit("Exit: latticefunctions.py not run as a standalone module.")

if __name__ == '__main__':
	main()