from core.multiplotcore import MultiPlotCore

class TopctPostAnalysis(MultiPlotCore):
	"""Post-analysis of the topological charge at fixed euclidean time."""
	observable_name = "Topological Charge in Euclidean Time"
	observable_name_compact = "topct"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$Q_{t_{euclidean}}$"
	sub_obs = True

def main():
	exit("Exit: TopctPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()