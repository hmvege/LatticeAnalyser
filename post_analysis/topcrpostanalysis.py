from core.postcore import PostCore

class TopcRPostAnalysis(object):
	"""Post-analysis of the topc ratio with Q^4/Q^2. Requires that Q4 and Q2 has been imported."""
	observable_name = r"$R=\frac{Q^4_C{Q^2}$"
	observable_name_compact = "topqr"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$R=\frac{Q^4_C{Q^2}$" # 1/8 correct?

def main():
	exit("Exit: TopcRPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()