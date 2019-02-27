# from pre_analysis.core.mcintervalanalysercore import MCIntervalAnalyser
from topcanalyser import TopcAnalyser

class TopcMCIntervalAnalyser(TopcAnalyser):
	"""
	Analysis where one can split the topological charge in Monte Carlo time to
	obtain an estimate of the topological charge.
	"""
	observable_name = "Topological Charge in MC Time"
	observable_name_compact = "topcMC"
	x_label = r"$\sqrt{8t_{f}}$ [fm]"
	y_label = r"$\langle Q \rangle$"

	# def __init__(self, *args, **kwargs):
	# 	super(TopcMCIntervalAnalyser, self).__init__(*args, **kwargs)
	# 	self.NT = self.y_original.shape[-1]
	# 	self.observable_output_folder_path_old = self.observable_output_folder_path

	# def set_mc_interval(self, *args):
	# 	"""Runs first the inherited time setter function, then its own."""
	# 	super(TopcMCIntervalAnalyser, self).set_mc_interval(*args)
	# 	self.observable_name = r"Q in MC interval $[%d,%d)$" % self.mc_interval


def main():
	exit("Module TopcMCIntervalAnalyser not intended for standalone usage.")

if __name__ == '__main__':
	main()