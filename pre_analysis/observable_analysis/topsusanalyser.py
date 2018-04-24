from pre_analysis.core.topsusanalysercore import TopsusAnalyserCore

class TopsusAnalyser(TopsusAnalyserCore):
	"""Topological susceptibility analysis class."""
	observable_name = "Topological Susceptibility"
	observable_name_compact = "topsus"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi_t^{1/4}[GeV]$"

	def __init__(self, *args, **kwargs):
		super(TopsusAnalyser, self).__init__(*args, **kwargs)
		self.y **= 2

def main():
	exit("Module TopSusAnalyser not intended for standalone usage.")

if __name__ == '__main__':
	main()