from pre_analysis.core.topsusanalysercore import TopsusAnalyserCore

class Topsus4Analyser(TopsusAnalyserCore):
	"""Class for topological susceptibility with quartic topological charge."""
	observable_name = r"$\chi(\langle Q^4 \rangle)^{1/8}$"
	observable_name_compact = "topsus4"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	# y_label = r"$\chi(\langle Q^4 \rangle)^{1/8} = \frac{\hbar}{aV^{1/4}} \langle Q^4 \rangle^{1/8} [GeV]$" # 1/8 correct?
	y_label = r"$\chi(\langle Q^4 \rangle)^{1/8} [GeV]$" # 1/8 correct?

	def __init__(self, *args, **kwargs):
		super(Topsus4Analyser, self).__init__(*args, **kwargs)
		self.y **= 4

	def chi(self, Q4):
		"""Topological susceptibility function."""
		return self.const * Q4**(0.125)

	def chi_std(self, Q4, Q4_std):
		"""Topological susceptibility with error propagation."""
		return 0.125*self.const * Q4_std / Q4**(0.875)

def main():
	exit("Module Topsus4Analyser not intended for standalone usage.")

if __name__ == '__main__':
	main()
	