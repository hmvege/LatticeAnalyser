from core.flowanalyser import FlowAnalyser

class Topc4Analyser(FlowAnalyser):
	"""Class for topological charge with quartic topological charge."""
	observable_name = r"$\langle Q^4 \rangle$"
	observable_name_compact = "topq4"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\langle Q^4 \rangle$"

	def __init__(self, *args, **kwargs):
		super(Topc4Analyser, self).__init__(*args, **kwargs)
		self.y **= 4

def main():
	exit("Module Topc4Analyser not intended for standalone usage.")

if __name__ == '__main__':
	main()