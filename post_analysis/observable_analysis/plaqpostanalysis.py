from post_analysis.core.postcore import PostCore

class PlaqPostAnalysis(PostCore):
	"""Post-analysis of the topological charge."""
	observable_name = "Plaquette"
	observable_name_compact = "plaq"
	y_label = r"$\langle P \rangle$"
	x_label = r"$\sqrt{8t_f}$ [fm]"
	formula = r"$P = \frac{1}{16V} \sum_{x,\mu,\nu} \tr\mathcal{Re} P_{\mu\nu}$"


def main():
	exit("Exit: PlaqPostAnalysis not intended to be a standalone module.")


if __name__ == '__main__':
	main()