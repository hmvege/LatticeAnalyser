from pre_analysis.core.flowanalyser import FlowAnalyser

class PlaquetteAnalyser(FlowAnalyser):
	"""Plaquette analysis class."""
	observable_name = "Plaquette"
	observable_name_compact = "plaq"
	x_label = r"$\sqrt{8t_{f}}$ [fm]"
	y_label = r"$\langle P \rangle$"


def main():
	exit("Module PlaquetteAnalyser not intended for standalone usage.")

if __name__ == '__main__':
	main()