from post_analysis.core.postcore import PostCore

class TopcPostAnalysis(PostCore):
	"""Post-analysis of the topological charge."""
	observable_name = "Topological Charge"
	observable_name_compact = "topc"
	x_label = r"$\sqrt{8t_f}$[fm]"
	y_label = r"$\langle Q \rangle$"
	formula = (r"$Q = - \sum_x \frac{1}{64 \cdot 32\pi^2}"
		r"\epsilon_{\mu\nu\rho\sigma}"
		r"Tr\{G^{clov}_{\mu\nu}G^{clov}_{\rho\sigma}\}$")

def main():
	exit("Exit: TopcPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()