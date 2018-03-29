from core.multiplotcore import MultiPlotCore

class TopsusQtQ0PostAnalysis(MultiPlotCore):
	"""Post-analysis of the topsus at a fixed flow time."""
	observable_name = r"$\chi(\langle Q_t Q_{t_0} \rangle)^{1/4}$"
	observable_name_compact = "topsusqtq0"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi(\langle Q_{t} Q_{t_0} \rangle)^{1/4} [GeV]$" # $\chi_t^{1/4}[GeV]$
	sub_obs = True

def main():
	exit("Exit: TopsusQtQ0PostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()