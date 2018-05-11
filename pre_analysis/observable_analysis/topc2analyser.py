from pre_analysis.core.flowanalyser import FlowAnalyser

class Topc2Analyser(FlowAnalyser):
	"""Class for topological charge with quartic topological charge."""
	observable_name = r"$\langle Q^2 \rangle$"
	observable_name_compact = "topc2"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\langle Q^2 \rangle$"

	def __init__(self, *args, **kwargs):
		super(Topc2Analyser, self).__init__(*args, **kwargs)
		self.y **= 2

def main():
	exit("Module Topc2Analyser not intended for standalone usage.")

if __name__ == '__main__':
	main()