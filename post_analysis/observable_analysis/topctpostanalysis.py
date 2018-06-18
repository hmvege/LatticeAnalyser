from post_analysis.core.multiplotcore import MultiPlotCore

class TopctPostAnalysis(MultiPlotCore):
	"""Post-analysis of the topological charge at fixed euclidean time."""
	observable_name = "Topological Charge at a fixed Euclidean Time"
	observable_name_compact = "topct"
	x_label = r"$\sqrt{8t_f}[fm]$"
	y_label = r"$\langle Q_{t_e} \rangle$"
	sub_obs = True
	subfolder_type = "teucl"

	def _convert_label(self, label):
		"""Short method for formatting time in labels."""
		try:
			return r"$t_e=%d$" % int(label)
		except ValueError:
			return r"$%s$" % label

def main():
	exit("Exit: TopctPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()