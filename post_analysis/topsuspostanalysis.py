from core.postcore import PostCore
import matplotlib.pyplot as plt
import numpy as np
import os

class TopsusPostAnalysis(PostCore):
	observable_name = "Topological Susceptibility"
	observable_name_compact = "topsus"

	# Regular plot variables
	x_label = r"$\sqrt{8t_{flow}}[fm]$"
	y_label = r"$\chi_t^{1/4}[GeV]$"
	formula = r"$\chi_t^{1/4}=\frac{\hbar c}{aV^{1/4}}\langle Q^2 \rangle^{1/4}$"

	# Continuum plot variables
	y_label_continuum = r"$\chi^{1/4}[GeV]$"
	# x_label_continuum = r"$a/{{r_0}^2}$"
	x_label_continuum = r"$a[fm]$"

	def _initiate_plot_values(self, data):
		"""
		Function that sorts data into a format specific for the plotting method.
		"""		
		for beta in sorted(data.keys()):
			if beta == 6.45: self.flow_time *= 2
			values = {}
			values["a"] = getLatticeSpacing(beta)
			values["x"] = values["a"]*np.sqrt(8*self.flow_time)
			values["y"] = data[beta]["y"]
			values["bs"] = self.bs_raw[beta][self.observable_name_compact]
			values["y_err"] = data[beta]["y_error"]
			values["label"] = r"%s $\beta=%2.2f$" % (self.size_labels[beta], beta)
			values["color"] = self.colors[beta]
			self.plot_values[beta] = values

	def plot_continuum(self, fit_target):
		# Gets the beta values

		a, obs, obs_err = [], [], []
		for beta in sorted(self.plot_values):
			x = self.plot_values[beta]["x"]
			y = self.plot_values[beta]["y"]
			y_err = self.plot_values[beta]["y_err"]

			fit_index = np.argmin(np.abs(x - fit_target))

			a.append(self.plot_values[beta]["a"])
			obs.append(y[fit_index])
			obs_err.append(y_err[fit_index])

		# Initiates empty arrays for the continuum limit
		a = np.asarray(a)[::-1]
		obs = np.asarray(obs)[::-1]
		obs_err = np.asarray(obs_err)[::-1]

		# Continuum limit arrays
		N_cont = 1000
		a_cont = np.linspace(-0.01, a[-1]*1.1, N_cont)

		# Fits to continuum and retrieves values to be plotted
		continuum_fit = LineFit(a, obs, obs_err)

		y_cont, y_cont_err, fit_params, chi_squared = continuum_fit.fit_weighted(a_cont)

		# continuum_fit.plot(True)
		
		# Gets the continium value and its error
		cont_index = np.argmin(np.abs(a_cont))
		a0 = [a_cont[cont_index], a_cont[cont_index]]
		y0 = [y_cont[cont_index], y_cont[cont_index]]

		if y_cont_err[1][cont_index] < y_cont_err[0][cont_index]:
			y0_err_lower = y0[0] - y_cont_err[1][cont_index]
			y0_err_upper = y_cont_err[0][cont_index] - y0[0]
		else:
			y0_err_lower = y0[0] - y_cont_err[0][cont_index]
			y0_err_upper = y_cont_err[1][cont_index] - y0[0]

		y0_err = [[y0_err_lower, 0], [y0_err_upper, 0]]

		# Creates figure and plot window
		fig = plt.figure()
		ax = fig.add_subplot(111)

		# Plots linefit with errorband
		ax.plot(a_cont, y_cont, color="tab:blue", alpha=0.5)
		ax.fill_between(a_cont, y_cont_err[0], y_cont_err[1],
			alpha=0.5, edgecolor='', facecolor="tab:blue")

		# Plot lattice points
		ax.errorbar(a, obs, yerr=obs_err, fmt="o",
			color="tab:orange", ecolor="tab:orange")

		# plots continuum limit
		ax.errorbar(a0, y0, yerr=y0_err, fmt="o", capsize=None, # 5 is a good value for cap size
			capthick=1, color="tab:red", ecolor="tab:red",
			label=r"$\chi^{1/4}=%.3f\pm%.3f$" % (y0[0], (y0_err_lower + y0_err_upper)/2.0))

		ax.set_ylabel(self.y_label_continuum)
		ax.set_xlabel(self.x_label_continuum)
		ax.set_title(r"$\sqrt{8t_{flow,0}} = %.2f[fm], \chi^2 = %.2g$" % (fit_target, chi_squared))
		ax.set_xlim(-0.01, a[-1]*1.1)
		ax.legend()
		ax.grid(True)

		# print "Target: %.16f +/- %.16f" % (c[0], y0_err[0])

		# Saves figure
		fname = os.path.join(self.output_folder_path, 
			"post_analysis_%s_continuum%s_%s.png" % (self.observable_name_compact, 
				str(fit_target).replace(".",""), self.analysis_data_type))
		fig.savefig(fname, dpi=self.dpi)

		print "Continuum plot of %s created in %s" % (self.observable_name.lower(), fname)
		# plt.show()
		plt.close(fig)

def main():
	exit("Exit: TopsusPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
	main()