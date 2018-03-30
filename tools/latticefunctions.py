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
	if beta < 5.7: raise ValueError("Beta should be larger than 5.7!")
	r = 0.5
	bval = (beta - 6)
	a = np.exp(-1.6805 - 1.7139*bval + 0.8155*bval**2 - 0.6667*bval**3)*0.5
	return a # fermi

def main():
	exit("Exit: latticefunctions.py not run as a standalone module.")

if __name__ == '__main__':
	main()