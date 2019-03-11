from pre_analysis.core.flowanalyser import FlowAnalyser
import numpy as np


def numerical_derivative(y, eps, axis=1):
	"""
	Numerical symmetric derivative.

	Args:
		y: numpy array to be differentiated of length N.
		eps: float, numerical derivative precision, h

	Returns:
		a
	"""
	dy = np.zeros(y.shape[axis] - 2)
	dy = np.roll(y, -1, axis=axis)[:,1:-1] - np.roll(y, 1, axis=axis)[:,1:-1]
	dy /= (2*eps)
	return dy


def W_function(vals, t=0.0):
	E, dE = vals
	return t**2 * (2*E + t*dE)


def W_funder_E(vals, t=0.0):
	E, dE = vals
	return 2*t**2


def W_funder_dE(vals, t=0.0):
	E, dE = vals
	return t**3


class WtEnergyAnalyser(FlowAnalyser):
	"""W(t) analysis class."""
	observable_name = r"$W(t)$"
	observable_name_compact = "w_t_energy"
	# Dimensionsless, Implied multiplication by a^2
	x_label = r"$t_f/r_0^2$"
	# Energy is dimension 4, while t^2 is dimension inverse 4, or 
	# length/time which is inverse energy, see Peskin and Schroeder
	y_label = r"$W(t)$"

	def __init__(self, *args, **kwargs):
		super(WtEnergyAnalyser, self).__init__(*args, **kwargs)
		self.y *= -1.0
		self.x = self.x[1:-1]
		self.x_vals = self.x * (self.a/self.r0)**2
		self.E = self.y[:,1:-1]
		self.dE = numerical_derivative(self.y, self.flow_epsilon, axis=1)
		self.y = W_function([self.E, self.dE], t=self.x)
		self.NFlows -= 2
		self.plot_hline_at = 0.3
		# Shape of ac data: [flowtime][replicums][observalbes][datapoints]
		self.ac_data = np.asarray([np.asarray([self.E, self.dE])])
		self.ac_data = np.rollaxis(self.ac_data, 3, 0)
		self.function_derivative = [W_funder_E, W_funder_dE]
		self._analysis_arrays_setup()
		self.function_derivative_parameters = [
			{"t": self.x[i]} for i in xrange(self.NFlows)]

		# TODO: Is the W pre-boot and post-boot equal if we bootstrap E first and then take W?

	def autocorrelation(self, store_raw_ac_error_correction=True, 
		method="wolff_full"):
		"""Ensuring that the correct autocorrelation method is used."""
		super(WtEnergyAnalyser, self).autocorrelation(
			store_raw_ac_error_correction=store_raw_ac_error_correction, 
			method="wolff_full")

	def plot_original(self):
		super(WtEnergyAnalyser, self).plot_bootstrap(x=self.x_vals, _plot_bs=False)

	def plot_bootstrap(self):
		super(WtEnergyAnalyser, self).plot_bootstrap(x=self.x_vals)

	def plot_jackknife(self):
		super(WtEnergyAnalyser, self).plot_jackknife(x=self.x_vals)

	# def plot_histogram(self, *args, **kwargs):
	# 	kwargs["x_label"] = r"$W(t)$[GeV]"
	# 	super(WtEnergyAnalyser, self).plot_histogram(*args, **kwargs)


def main():
	exit("Module WtWtEnergyAnalyser not intended for standalone usage.")


if __name__ == '__main__':
	main()