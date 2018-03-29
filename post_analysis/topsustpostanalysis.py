from core.multiplotcore import MultiPlotCore

class TopsustPostAnalysis(MultiPlotCore):
	"""Post-analysis of the topsus with with one Q at fixed euclidean time."""
	observable_name = "Topological Susceptibility in Euclidean Time"
	observable_name_compact = "topsust"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi(\langle Q_t Q_{t_{euclidean}} \rangle)^{1/4} [GeV]$"
	sub_obs = True

def main():
	exit("Exit: TopsustPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()