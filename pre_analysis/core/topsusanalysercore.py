from flowanalyser import FlowAnalyser
import statistics.parallel_tools as ptools
import numpy as np

class TopsusAnalyserCore(FlowAnalyser):
	"""Topological susceptibility analysis base class."""
	observable_name = "Topological Susceptibility"
	observable_name_compact = "topsus"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi_t^{1/4}[GeV]$"
	NTemporal = {6.0: 48, 6.1: 56, 6.2: 64, 6.45: 96}

	def __init__(self, *args, **kwargs):
		super(TopsusAnalyserCore, self).__init__(*args, **kwargs)
		self.__set_size()

	def chi(self, Q_squared):
		"""Topological susceptibility function."""
		return self.const*Q_squared**(0.25)

	def chi_std(self, Q_squared, Q_squared_std):
		"""Topological susceptibility with error propagation."""
		return 0.25*self.const*Q_squared_std / Q_squared**(0.75)

	def _check_skip_wolff_condition(self):
		"""Checks if we have a negative mean of Q^2."""
		if np.all(np.mean(self.y, axis=0) >= 0):
			return True
		else:
			msg = "\nWARNING: cannot perform propagating autocorrelation, "
			msg += "performing non-propagating autocorrelation. "
			if self.verbose:
				msg += "Since we have negative values in the mean of Q^2 and "
				msg += "would thus be taking the negative cube root in the Wolff "
				msg += "method.\n"
			print msg
			return False


	def __set_size(self):
		"""Function that sets the lattice size deepending on the beta value."""
		# Sets up constants used in the chi function for topsus
		self.function_derivative = ptools._chi_derivative
		self.V = self.lattice_sizes[self.beta]
		self.const = self.hbarc/self.a/self.V**(1./4)
		self.function_derivative_parameters = {"const": self.const}

	def jackknife(self, F=None, F_error=None, store_raw_jk_values=True):
		"""Overriding the jackknife class by adding the chi-function"""
		super(TopsusAnalyserCore, self).jackknife(F=self.chi,
			F_error=self.chi_std, store_raw_jk_values=store_raw_jk_values)

	def boot(self, N_bs, F=None, F_error=None, store_raw_bs_values=True, 
		index_lists=None):
		"""Overriding the bootstrap class by adding the chi-function"""
		super(TopsusAnalyserCore, self).boot(N_bs, F=self.chi,
			F_error=self.chi_std, store_raw_bs_values=store_raw_bs_values,
			index_lists=index_lists)

# def main():
# 	exit("Module TopSusAnalyser not intended for standalone usage.")

# if __name__ == '__main__':
# 	main()