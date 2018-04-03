from flowanalyser import FlowAnalyser
import statistics.parallel_tools as ptools

class TopsusAnalyserCore(FlowAnalyser):
	"""Topological susceptibility analysis base class."""
	observable_name = "Topological Susceptibility"
	observable_name_compact = "topsus"
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi_t^{1/4}[GeV]$"

	def __init__(self, *args, **kwargs):
		super(TopsusAnalyserCore, self).__init__(*args, **kwargs)
		self.__set_size()

	def chi(self, Q_squared):
		"""Topological susceptibility function."""
		return self.const*Q_squared**(0.25)

	def chi_std(self, Q_squared, Q_squared_std):
		"""Topological susceptibility with error propagation."""
		return 0.25*self.const*Q_squared_std / Q_squared**(0.75)

	def __set_size(self):
		"""Function that sets the lattice size deepending on the beta value."""
		# Retrieves lattice spacing
		self.function_derivative = ptools._chi_derivative

		# Sets up constants used in the chi function for topological susceptibility
		self.V = self.lattice_sizes[self.beta]
		self.const = self.hbarc/self.a/self.V**(1./4)
		self.function_derivative_parameters = {"const": self.const}

	def jackknife(self, F=None, F_error=None, store_raw_jk_values=True):
		"""Overriding the jackknife class by adding the chi-function"""
		super(TopsusAnalyserCore, self).jackknife(F=self.chi,
			F_error=self.chi_std, store_raw_jk_values=store_raw_jk_values)

	def boot(self, N_bs, F=None, F_error=None, store_raw_bs_values=True):
		"""Overriding the bootstrap class by adding the chi-function"""
		super(TopsusAnalyserCore, self).boot(N_bs, F=self.chi,
			F_error=self.chi_std, store_raw_bs_values=store_raw_bs_values)

# def main():
# 	exit("Module TopSusAnalyser not intended for standalone usage.")

# if __name__ == '__main__':
# 	main()