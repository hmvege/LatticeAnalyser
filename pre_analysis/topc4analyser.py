from core.topsusanalysercore import TopsusAnalyserCore

class Topc4Analyser(TopsusAnalyserCore):
	"""Class for topological charge with quartic topological charge."""
	observable_name = r"$Q^4$"
	observable_name_compact = "topq4"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	# y_label = r"$\chi(\langle Q^4 \rangle)^{1/8} = \frac{\hbar}{aV^{1/4}} \langle Q^4 \rangle^{1/8} [GeV]$" # 1/8 correct?
	y_label = r"$Q^4$" # 1/8 correct?

	def __init__(self, *args, **kwargs):
		super(Topc4Analyser, self).__init__(*args, **kwargs)
		self.y **= 4

def main():
	exit("Module Topc4Analyser not intended for standalone usage.")

if __name__ == '__main__':
	main()