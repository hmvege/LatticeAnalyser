from pre_analysis.core.flowanalyser import FlowAnalyser

class TopcAnalyser(FlowAnalyser):
	"""Topological charge analysis class."""
	observable_name = "Topological charge"
	observable_name_compact = "topc"
	x_label = r"$\sqrt{8t_{flow}}[fm]$" # Implied multiplication by a
	# y_label = r"$Q = - \sum_x \frac{1}{64 \cdot 32\pi^2}\epsilon_{\mu\nu\rho\sigma}Tr\{G^{clov}_{\mu\nu}G^{clov}_{\rho\sigma}\}$"
	y_label = r"$\langle Q \rangle$" # Dimensionsless, a's from discretization cancel out the 1/a^4 in the integration

# def main():
# 	exit("Module TopcAnalyser not intended for standalone usage.")

# if __name__ == '__main__':
# 	main()