from core.postcore import PostCore

class Topc4PostAnalysis(PostCore):
	"""Post-analysis of the topsus with Q^4."""
	observable_name = r"$Q^4$"
	observable_name_compact = "topq4"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\langle Q^4 \rangle$" # 1/8 correct?

def main():
	exit("Exit: Topc4PostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()