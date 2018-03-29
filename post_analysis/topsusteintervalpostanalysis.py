from core.multiplotcore import MultiPlotCore

class TopsusteIntervalPostAnalysis(MultiPlotCore):
	"""Post-analysis of the topsus in euclidean time intervals."""
	observable_name = "Topological Susceptibility in MC Time"
	observable_name_compact = "topsuste"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi^{1/4} [GeV]$"
	sub_obs = True

def main():
	exit("Exit: TopsusteIntervalPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()