from post_analysis.core.multiplotcore import MultiPlotCore

class TopcteIntervalPostAnalysis(MultiPlotCore):
	"""Post-analysis of the topological charge in euclidean time intervals."""
	observable_name = "Topological Charge in Euclidean Time intervals"
	observable_name_compact = "topcte"
	x_label = r"$\sqrt{8t_{f}}$ [fm]"
	y_label = r"$\langle Q \rangle$"
	sub_obs = True
	subfolder_type = "teInt"

def main():
	exit(("Exit: TopcteIntervalPostAnalysis not intended to be a standalone "
		"module."))

if __name__ == '__main__':
	main()