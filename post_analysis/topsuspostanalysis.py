from core.topsuscore import TopsusCore
from tools.latticefunctions import get_lattice_spacing
from statistics.linefit import LineFit
import matplotlib.pyplot as plt
import numpy as np
import os

class TopsusPostAnalysis(TopsusCore):
	observable_name = "Topological Susceptibility"
	observable_name_compact = "topsus"

	# Regular plot variables
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi_t^{1/4}[GeV]$"
	formula = r"$\chi_t^{1/4}=\frac{\hbar c}{aV^{1/4}}\langle Q^2 \rangle^{1/4}$"

	# Continuum plot variables
	y_label_continuum = r"$\chi^{1/4}[GeV]$"
	# x_label_continuum = r"$a/{{r_0}^2}$"
	x_label_continuum = r"$a^2/t_0$"

def main():
	exit("Exit: TopsusPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()