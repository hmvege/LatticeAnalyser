from post_analysis.core.postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from tools.table_printer import TablePrinter
import tools.sciprint as sciprint
from statistics.linefit import LineFit, extract_fit_target
import matplotlib.pyplot as plt
import numpy as np
import os
import copy

class WtPostAnalysis(PostCore):
    """Post analysis of the energy, <E>."""
    observable_name = "Wt"
    observable_name_compact = "w_t_energy"

    # Regular plot variables
    y_label = r"$W(t)$"
    x_label = r"$t_f/r_0^2$"
    formula = r"$W(t) = t\frac{\partial}{\partial t_f}\left(t_f^2\langle E \rangle\right)$"

    # Continuum plot variables
    x_label_continuum = r"$(a/r_0)^2$"
    y_label_continuum = r"$w_0$"

    @staticmethod
    def derivative(y, eps):
        """
        First order derivative of O(n^2).

        Args:
            y, array, to derivate.
            eps, integration step length.
        Returns:
            y derivative.
        """
        y_der = np.zeros(len(y) - 2)
        for i in xrange(1, len(y) - 1):
            y_der[i-1] = (y[i+1] - y[i-1]) / (2*eps)
        return y_der

    def calculateW(self, x, y, y_err, y_raw, feps, t_unscaled):
        """
        Calculates the W(t) used in the scale setting definition given in
        http://xxx.lanl.gov/pdf/1203.4469v2
        """
        t = np.zeros(len(x) - 2)
        W = np.zeros(len(x) - 2)
        W_err = np.zeros(len(x) - 2)
        W_raw = np.zeros((len(x) - 2, y_raw.shape[-1]))
        dE_raw = np.zeros((len(x) - 2, y_raw.shape[-1]))
        for i in xrange(1, len(y) - 1):
            # t[i-1] = x[i]
            t[i-1] = t_unscaled[i]

        dE = self.derivative(y, feps)
        for iBoot in xrange(W_raw.shape[-1]):
            dE_raw[:,iBoot] = self.derivative(y_raw[:,iBoot], feps)

        # Calculates W(t) = t d/dt { t^2 E(t) }
        for i in xrange(1, len(y) - 1):
            W[i-1] = 2*t[i-1]**2*y[i] + t[i-1]**3*dE[i-1]
            W_raw[i-1] = 2*t[i-1]**2*y_raw[i] + t[i-1]**3*dE_raw[i-1]

        # Uncertainty propagation in derivative: 
        # https://physics.stackexchange.com/questions/200029/how-does-uncertainty-error-propagate-with-differentiation
        for i in xrange(1, len(y) - 1):
            # W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2)# + (t[i-1]**3*np.sqrt(2)*y_err[i]/feps)**2)
            # print np.std(W_raw[i-1])
            # W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2 + (t[i-1]**3*np.std(W_raw[i-1]))**2)
            W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2)

        # Sets up the x-axis value for t_f*a^2
        ta2 = np.zeros(len(x) - 2)
        for i in xrange(1, len(y) - 1):
            ta2[i-1] = t_unscaled[i] 

        # # plt.plot(x_der, y_der)
        # # plt.errorbar(t, W, yerr=W_err)
        # plt.plot(x[1:-1], W, color="tab:red", alpha=0.5)
        # plt.fill_between(x[1:-1], W-W_err, W+W_err, alpha=0.5, edgecolor='',
        #     facecolor="tab:red")
        # plt.grid(True)
        # # plt.hlines(0.3,t[0], t[-1], linestyle=":", alpha=0.75, color="gray")
        # plt.hlines(0.3,x[1], x[-2], linestyle=":", alpha=0.75, color="gray")
        # plt.xlabel(r"$t/{r_0^2}$")
        # plt.ylabel(r"$W(t)$")
        # plt.show()

        return ta2, W, W_err, W_raw

    def _initiate_plot_values(self, data, data_raw):
        # Sorts data into a format specific for the plotting method
        for beta in sorted(data.keys()):
            values = {}
            values["beta"] = beta
            values["a"], values["a_err"] = get_lattice_spacing(beta)
            values["xraw"] = data[beta]["x"] # t_f/a^2
            values["t"] = values["xraw"]*values["a"]**2
            values["x"] = values["t"]/self.r0**2
            values["y"] = data[beta]["y"]
            values["y_err"] = data[beta]["y_error"]
            values["flow_epsilon"] = self.flow_epsilon[beta]
            # values["tder"], values["W"], values["W_err"], values["W_raw"] = \
            #     self.calculateW(values["x"], data[beta]["y"], 
            #         data[beta]["y_error"], 
            #         data_raw[beta][self.observable_name_compact], 
            #         values["flow_epsilon"], data[beta]["x"])
            # exit("Exits in plot values initiations.")

            if self.with_autocorr:
                values["tau_int"] = data[beta]["ac"]["tau_int"]
                values["tau_int_err"] = data[beta]["ac"]["tau_int_err"]
            else:
                values["tau_int"] = None
                values["tau_int_err"] = None

            values[self.analysis_data_type] = \
                (data_raw[beta][self.observable_name_compact].T \
                    *(data[beta]["x"]**2)).T

            values["label"] = (r"%s $\beta=%2.2f$" %
                (self.size_labels[beta], beta))

            self.plot_values[beta] = values

    def get_w0_scale(self, extrapolation_method="plateau_mean", W0=0.3, 
        **kwargs):
        """
        Method for retrieving the w0 reference scale setting, based on paper:
        http://xxx.lanl.gov/pdf/1203.4469v2
        """
        if self.verbose:
            print "Scale w0 extraction method:      " + extrapolation_method
            print "Scale w0 extraction data:        " + self.analysis_data_type

        # Retrieves t0 values from data
        a_values = []
        a_values_err = []
        w0_values = []
        w0err_values = []

        for beta, bval in sorted(self.plot_values.items(), key=lambda i: i[0]):
            # print bval["xraw"]
            # exit("WT")
            y0, w0, w0_err, _, _ = extract_fit_target(W0, bval["xraw"], 
                bval["y"], y_err=bval["y_err"], 
                y_raw=bval[self.analysis_data_type], 
                tau_int=bval["tau_int"][1:-1], 
                tau_int_err=bval["tau_int_err"][1:-1],
                extrapolation_method=extrapolation_method, plateau_size=10,
                inverse_fit=True, **kwargs)

            # TODO: fix a lattice spacing error here @ w_t
            a_values.append(bval["a"]**2)
            a_values_err.append(2*bval["a"]*bval["a_err"])
            w0_values.append(np.sqrt(w0)*bval["a"])
            w0err_values.append(0.5*w0_err/np.sqrt(w0))

        a_values = np.asarray(a_values[::-1])
        a_values_err = np.asarray(a_values_err[::-1])
        w0_values = np.asarray(w0_values[::-1])
        w0err_values = np.asarray(w0err_values[::-1])

        # # Functions for t0 and propagating uncertainty
        # t0_func = lambda _t0: np.sqrt(8*_t0)/self.r0
        # t0err_func = lambda _t0, _t0_err: _t0_err*np.sqrt(8/_t0)/(2.0*self.r0)


        # # Sets up t0 and t0_error values to plot
        # y = t0_func(t0_values)
        # yerr = t0err_func(t0_values, t0err_values)

        # Extrapolates t0 to continuum
        N_cont = 1000
        a_squared_cont = np.linspace(-0.00025, a_values[-1]*1.1, N_cont)

        # Fits to continuum and retrieves values to be plotted
        continuum_fit = LineFit(a_values, w0_values, y_err=w0err_values)
        y_cont, y_cont_err, fit_params, chi_squared = \
            continuum_fit.fit_weighted(a_squared_cont)

        res = continuum_fit(0, weighted=True)
        self.w0_cont = res[0][0]
        self.w0_cont_error = (res[1][-1][0] - res[1][0][0])/2
        # self.t0_cont = self.sqrt_8t0_cont**2/8
        # self.t0_cont_error = self.sqrt_8t0_cont_error*np.sqrt(self.t0_cont/2.0)

        # Creates figure and plot window
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plots linefit with errorband
        ax.plot(a_squared_cont, y_cont, color="tab:red", alpha=0.5,
            label=r"$\chi=%.2f$" % chi_squared)
        ax.fill_between(a_squared_cont, y_cont_err[0], 
            y_cont_err[1], alpha=0.5, edgecolor='',
            facecolor="tab:red")
        ax.axvline(0, linestyle="dashed", color="tab:red")
        ax.errorbar(a_values, w0_values, xerr=a_values_err, yerr=w0err_values,
            fmt="o", capsize=5, capthick=1, color="#000000", ecolor="#000000")
        ax.set_ylabel(r"$w_0[\mathrm{fm}]$")
        ax.set_xlabel(r"$a^2[\mathrm{GeV}^{-2}]$")
        ax.set_xlim(a_squared_cont[0], a_squared_cont[-1])
        ax.legend()
        ax.grid(True)

        # Saves figure
        fname = os.path.join(self.output_folder_path, 
            "post_analysis_extrapmethod%s_w0reference_continuum_%s.png" % (
                extrapolation_method, self.analysis_data_type))
        fig.savefig(fname, dpi=self.dpi)
        if self.verbose:
            print "Figure saved in %s" % fname

        plt.close(fig)

        w0_values = w0_values[::-1]
        w0err_values = w0err_values[::-1]

        _tmp_beta_dict = {
            b: {
                "w0": w0_values[i],
                "w0err": w0err_values[i],
                "aL": self.plot_values[b]["a"]*self.lattice_sizes[b][0],
                "aLerr": (self.plot_values[b]["a_err"] \
                    * self.lattice_sizes[b][0]),
                "L": self.lattice_sizes[b][0],
                "a": self.plot_values[beta]["a"],
                "a_err": self.plot_values[b]["a_err"],
            }
            for i, b in enumerate(self.beta_values)
        }

        w0_dict = {"w0cont": self.w0_cont, "w0cont_err": self.w0_cont_error}
        w0_dict.update(_tmp_beta_dict)

        if self.verbose:
            print "w0 reference values table: "
            print "w0 = %.16f +/- %.16f" % (self.w0_cont, self.w0_cont_error)
            for b in self.beta_values:
                msg = "beta = %.2f || w0 = %10f +/- %-10f" % (b, 
                    w0_dict[b]["w0"], w0_dict[b]["w0err"])
                print msg

        if self.print_latex:
            # Header:
            # beta  w0  a^2  L/a  L  a

            header = [r"$\beta$", r"$w_0[\fm]$", 
                r"$a^2[\mathrm{GeV}^{-2}]$", r"$L/a$", r"$L[\fm]$", 
                r"$a[\fm]$"]

            bvals = self.beta_values
            tab = [
                [r"{0:.2f}".format(b) for b in bvals],
                [r"{0:s}".format(sciprint.sciprint(w0_dict[b]["w0"], 
                    w0_dict[b]["w0err"])) for b in bvals],
                [r"{0:s}".format(sciprint.sciprint(self.plot_values[b]["a"]**2,
                    self.plot_values[b]["a_err"]*2*self.plot_values[b]["a"])) 
                    for b in bvals],
                [r"{0:d}".format(self.lattice_sizes[b][0]) for b in bvals],
                [r"{0:s}".format(sciprint.sciprint(
                    self.lattice_sizes[b][0]*self.plot_values[b]["a"], 
                    self.lattice_sizes[b][0]*self.plot_values[b]["a_err"])) 
                    for b in bvals],
                [r"{0:s}".format(sciprint.sciprint(
                    self.plot_values[b]["a"],
                    self.plot_values[b]["a_err"])) for b in bvals],
            ]

            ptab = TablePrinter(header, tab)            
            ptab.print_table(latex=True, width=15)


    def plot(self, *args, **kwargs):
        """Plots the W(t)."""
        w_plot_values = copy.deepcopy(self.plot_values)
        # for beta in sorted(self.beta_values):
        #     w_plot_values[beta]["x"] = self.plot_values[beta]["tder"]
        #     w_plot_values[beta]["y"] = self.plot_values[beta]["W"]
        #     w_plot_values[beta]["y_err"] = self.plot_values[beta]["W_err"]

        # kwargs["observable_name_compact"] = "energyW"
        # kwargs["x_label"] = r"$t_f$"
        # kwargs["y_label"] = r"$W(t)$"
        kwargs["x_label"] = self.x_label
        kwargs["y_label"] = self.y_label
        kwargs["plot_hline_at"] = 0.3
        # kwargs["show_plot"] = True

        self._plot_core(w_plot_values, *args, **kwargs)

    def __str__(self):
        """Class string representation method."""
        msg = "\n" + self.section_seperator
        msg += "\nPost analaysis for:        " + self.observable_name_compact
        msg += "\n" + self.__doc__
        msg += "\nAnalysis-type:             " + self.analysis_data_type
        # msg += "\nE0 extraction method:      " + self.extrapolation_method
        msg += "\nIncluding autocorrelation: " + self.ac
        msg += "\nOutput folder:             " + self.output_folder_path
        msg += "\n" + self.section_seperator
        return msg

def main():
    exit("Exit: EnergyPostAnalysis not intended to be a standalone module.")

if __name__ == '__main__':
    main()