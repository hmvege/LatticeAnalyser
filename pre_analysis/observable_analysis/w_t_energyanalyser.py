from pre_analysis.core.flowanalyser import FlowAnalyser

def numerical_derivative(y, eps, axis=0):
	"""
	Numerical symmetric derivative.

	Args:
		y: numpy array to be differentiated of length N.
		eps: float, numerical derivative precision, h

	Returns:
		a
	"""
	dy = np.zeros(y.shape[axis] - 2)
	dy = np.roll(y, 1, axis=axis)[1:-1] - np.roll(y, -1, axis=axis)[1:-1]
	dy /= (2*eps)
	return dy


def W_function(t, dE, E):
	return t**2 * (2*E + t*dE)

class WtEnergyAnalyser(FlowAnalyser):
	"""W(t) analysis class."""
	observable_name = r"$W(t)$"
	observable_name_compact = "w_t_energy"
	# Dimensionsless, Implied multiplication by a^2
	x_label = r"$t/r_0^2$"
	# Energy is dimension 4, while t^2 is dimension inverse 4, or 
	# length/time which is inverse energy, see Peskin and Schroeder
	y_label = r"$W(t)$"

	def __init__(self, *args, **kwargs):
		super(WtEnergyAnalyser, self).__init__(*args, **kwargs)
		self.y *= -1.0

		self.x_vals = self.x / self.r0**2 * self.a**2
		self.dE = numerical_derivative(self.y, self.flow_epsilon, axis=1)
		self.y = self.y[1:-1]
		self.NFlows -= 2

		self.plot_hline_at = 0.3

	def update_func_der_params(self, i):
		pass

	def autocorrelation(self, store_raw_ac_error_correction=True, 
		method="wolff"):
		"""Ensuring that the correct autocorrelation method is used."""
		super(autocorrelation, self).__init__(
			store_raw_ac_error_correction=True, 
			method="wolff_full")

	def correction_function(self, y):
		return y * self.x * self.x # factor 0.5 left out

	def plot_original(self):
		super(WtEnergyAnalyser, self).plot_boot(x=self.x_vals, 
			correction_function=self.correction_function, _plot_bs=False)

	def plot_boot(self):
		super(WtEnergyAnalyser, self).plot_boot(x=self.x_vals, 
			correction_function=self.correction_function)

	def plot_jackknife(self):
		super(WtEnergyAnalyser, self).plot_jackknife(x=self.x_vals, 
			correction_function=self.correction_function)

	def plot_histogram(self, *args, **kwargs):
		kwargs["x_label"] = r"$\langle E \rangle$[GeV]"
		super(WtEnergyAnalyser, self).plot_histogram(*args, **kwargs)

def main():
	exit("Module WtWtEnergyAnalyser not intended for standalone usage.")

if __name__ == '__main__':
	main()