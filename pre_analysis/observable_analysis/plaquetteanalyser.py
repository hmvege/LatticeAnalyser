from pre_analysis.core.flowanalyser import FlowAnalyser

class PlaquetteAnalyser(FlowAnalyser):
	"""Plaquette analysis class."""
	observable_name = "Plaquette"
	observable_name_compact = "plaq"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$P_{\mu\nu}$"


def main():
	exit("Module PlaquetteAnalyser not intended for standalone usage.")

if __name__ == '__main__':
	main()