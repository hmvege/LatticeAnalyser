from pre_analysis.core.flowanalyser import FlowAnalyser

class EnergyAnalyser(FlowAnalyser):
	"""Energy/action density analysis class."""
	observable_name = "Energy"
	observable_name_compact = "energy"
	# Dimensionsless, Implied multiplication by a^2
	x_label = r"$t/r_0^2$"
	# Energy is dimension 4, while t^2 is dimension inverse 4, or 
	# length/time which is inverse energy, see Peskin and Schroeder
	y_label = r"$t^2\langle E \rangle$" 

	def __init__(self, *args, **kwargs):
		super(EnergyAnalyser, self).__init__(*args, **kwargs)
		self.y *= -1.0

		self.x_vals = self.x / self.r0**2 * self.a**2

		self.plot_hline_at = 0.3

	def correction_function(self, y):
		return y * self.x * self.x # factor 0.5 left out

	def plot_original(self):
		super(EnergyAnalyser, self).plot_boot(x=self.x_vals, 
			correction_function=self.correction_function, _plot_bs=False)

	def plot_boot(self):
		super(EnergyAnalyser, self).plot_boot(x=self.x_vals, 
			correction_function=self.correction_function)

	def plot_jackknife(self):
		super(EnergyAnalyser, self).plot_jackknife(x=self.x_vals, 
			correction_function=self.correction_function)

	def plot_histogram(self, *args, **kwargs):
		kwargs["x_label"] = r"$\langle E \rangle$[GeV]"
		super(EnergyAnalyser, self).plot_histogram(*args, **kwargs)

def main():
	exit("Module EnergyAnalyser not intended for standalone usage.")

if __name__ == '__main__':
	main()