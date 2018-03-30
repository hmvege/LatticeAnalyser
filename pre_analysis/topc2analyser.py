from core.flowanalyser import FlowAnalyser

class Topc2Analyser(FlowAnalyser):
	"""Class for topological charge with quartic topological charge."""
	observable_name = r"$Q^2$"
	observable_name_compact = "topq2"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	# y_label = r"$\chi(\langle Q^4 \rangle)^{1/8} = \frac{\hbar}{aV^{1/4}} \langle Q^4 \rangle^{1/8} [GeV]$" # 1/8 correct?
	y_label = r"$Q^2$" # 1/8 correct?

	def __init__(self, *args, **kwargs):
		super(Topc2Analyser, self).__init__(*args, **kwargs)
		self.y **= 2

def main():
	exit("Module Topc2Analyser not intended for standalone usage.")

if __name__ == '__main__':
	main()