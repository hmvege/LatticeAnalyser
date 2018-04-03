from core.postcore import PostCore

class Topc2PostAnalysis(PostCore):
	"""Post-analysis of the topc with Q^2."""
	observable_name = r"$Q^2$"
	observable_name_compact = "topq2"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$Q^2$" # 1/8 correct?

def main():
	exit("Exit: Topc2PostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()