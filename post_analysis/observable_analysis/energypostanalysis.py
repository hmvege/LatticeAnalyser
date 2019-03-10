import numba as nb
import copy
import os
import numpy as np
from post_analysis.core.postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from tools.table_printer import TablePrinter
import tools.sciprint as sciprint
from statistics.linefit import LineFit, extract_fit_target
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt


class EnergyPostAnalysis(PostCore):
    """Post analysis of the energy, <E>."""
    observable_name = "Energy"
    observable_name_compact = "energy"

    # Regular plot variables
    y_label = r"$t_f^2\langle E\rangle$"
    x_label = r"$t_f/r_0^2$"
    formula = (r"$\langle E\rangle = -\frac{1}{64|\Lambda|}"
               r"F_{\mu\nu}^a{F^a}^{\mu\nu}$")

    # Continuum plot variables
    x_label_continuum = r"$(a/r_0)^2$"
    y_label_continuum = r"$\frac{\sqrt{8t_{f,0}}}{r_0}$"

    @staticmethod
    @nb.njit(cache=True)
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
        for i in range(1, len(y) - 1):
            y_der[i-1] = (y[i+1] - y[i-1]) / (2*eps)

        return y_der

    def calculateW(self, x, y, y_err, y_raw, feps, t_unscaled):
        """
        Calculates the W(t) used in the scale setting definition given in
        http://xxx.lanl.gov/pdf/1203.4469v2
        """
        t = t_unscaled[1:-1]  # t unscaled since t^2 <E> uses t unscaled.
        W = np.zeros(len(t))
        W_err = np.zeros(len(t))
        W_raw = np.zeros((len(t), y_raw.shape[-1]))
        dE_raw = np.zeros((len(t), y_raw.shape[-1]))

        dE = self.derivative(y, feps)
        for iBoot in xrange(W_raw.shape[-1]):
            dE_raw[:, iBoot] = self.derivative(y_raw[:, iBoot], feps)

        # dE_err = np.zeros(len(x) - 2)
        # for i in xrange(1, len(y) - 1):
        # 	dE_err[i-1] = (y_err[+1] - y_err[-1])/(2*feps)

        # Calculates W(t) = t d/dt { t^2 E(t) }
        for i in xrange(1, len(y) - 1):
            W[i-1] = 2*t[i-1]**2*y[i] + t[i-1]**3*dE[i-1]
            W_raw[i-1] = 2*t[i-1]**2*y_raw[i] + t[i-1]**3*dE_raw[i-1]

        # Uncertainty propagation in derivative:
        # https://physics.stackexchange.com/questions/200029/how-does-uncertainty-error-propagate-with-differentiation
        for i in xrange(1, len(y) - 1):
            # W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2)# + (t[i-1]**3*np.sqrt(2)*y_err[i]/feps)**2)
            # print np.std(W_raw[i-1])
            W_err[i-1] = np.std(W_raw[i-1])
            # W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2 + (t[i-1]**3*np.std(W_raw[i-1]))**2)
            # W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2)
            # W_err[i-1] = np.sqrt((2*t[i-1]**2*y_err[i])**2 + (y_err[i+1]/(2*feps))**2 + (y_err[i-1]/(2*feps))**2 )

        # # Sets up the x-axis value for t_f*a^2
        # ta2 = np.zeros(len(x) - 2)
        # for i in xrange(1, len(y) - 1):
        # 	ta2[i-1] = t_unscaled[i]

        # W = np.mean(W_raw, axis=1)
        # plt.plot(x[1:-1], W, color="tab:red", alpha=0.5)
        # plt.fill_between(x[1:-1], W-W_err, W+W_err, alpha=0.5, edgecolor='',
        # 	facecolor="tab:red")
        # plt.grid(True)
        # plt.hlines(0.3,x[1], x[-2], linestyle=":", alpha=0.75, color="gray")
        # plt.xlabel(r"$t_f/{r_0^2}$")
        # plt.ylabel(r"$W(t_f)$")
        # plt.show()
        # exit(1)

        return x, W, W_err, W_raw

    def _initiate_plot_values(self, data, data_raw):
        # Sorts data into a format specific for the plotting method
        for beta in sorted(data.keys()):
            values = {}
            values["beta"] = beta
            values["a"], values["a_err"] = get_lattice_spacing(beta)
            values["t"] = data[beta]["x"]*values["a"]**2
            values["sqrt8t"] = values["a"]*np.sqrt(8*data[beta]["x"])
            values["x"] = values["t"]/self.r0**2
            values["y"] = data[beta]["y"]*data[beta]["x"]**2
            values["y_err"] = data[beta]["y_error"]*data[beta]["x"]**2
            values["flow_epsilon"] = self.flow_epsilon[beta]
            values["y_uraw"] = \
                self.data_raw["unanalyzed"][beta][self.observable_name_compact]

            # Calculates the energy derivatve
            values["tder"], values["W"], values["W_err"], values["W_raw"] = \
                self.calculateW(values["t"], data[beta]["y"],
                                data[beta]["y_error"],
                                data_raw[beta][self.observable_name_compact],
                                values["flow_epsilon"], data[beta]["x"])

            if self.with_autocorr:
                values["tau_int"] = data[beta]["ac"]["tau_int"]
                values["tau_int_err"] = data[beta]["ac"]["tau_int_err"]
                values["tau_raw"] = self.ac_raw["ac_raw"][beta]
                values["tau_raw_err"] = self.ac_raw["ac_raw_error"][beta]
            else:
                values["tau_int"] = None
                values["tau_int_err"] = None
                values["tau_raw"] = None
                values["tau_raw_err"] = None

            # Calculates the t^2<E> for the raw values
            values[self.analysis_data_type] = \
                (data_raw[beta][self.observable_name_compact].T
                 * (data[beta]["x"]**2)).T

            values["label"] = (r"%s $\beta=%2.2f$" %
                               (self.size_labels[beta], beta))

            self.plot_values[beta] = values

    def _extract_flow_time_index(self, target_flow):
        """
        Returns index corresponding to given flow time

        Args:
            target_flow: float, some fraction between 0.0-0.6 usually
        """

        for beta_ in self.plot_values:
            assert target_flow < self.plot_values[beta_]["sqrt8t"][-1], (
                "Flow time exceeding bounds for %f which has max flow "
                "time value of %f" % (beta_, self.plot_values[beta_]["x"][-1]))

        # Selects and returns fit target index
        return [np.argmin(np.abs(self.plot_values[beta_]["sqrt8t"] - target_flow))
                for beta_ in sorted(self.plot_values)]


    def get_t0_scale(self, extrapolation_method="plateau_mean", E0=0.3,
                     **kwargs):
        """
        Method for retrieveing reference value t0 based on Luscher(2010),
        Properties and uses of the Wilson flow in lattice QCD.
        t^2<E_t>|_{t=t_0} = 0.3
        Will return t0 values and make a plot of the continuum value 
        extrapolation.

        Args:
                extrapolation_method: str, optional. Method of t0 extraction. 
                        Default is plateau_mean.
                E0: float, optional. Default is 0.3.

        Returns:
                t0: dictionary of t0 values for each of the betas, and a 
                        continuum value extrapolation.
        """
        if self.verbose:
            print "Scale t0 extraction method:      " + extrapolation_method
            print "Scale t0 extraction data:        " + self.analysis_data_type

        # Retrieves t0 values from data
        a_values = []
        a_values_err = []
        t0_values = []
        t0err_values = []

        for beta, bval in sorted(self.plot_values.items(), key=lambda i: i[0]):
            y0, t0, t0_err, _, _ = extract_fit_target(
                E0, bval["t"], bval["y"],
                y_err=bval["y_err"], y_raw=bval[self.analysis_data_type],
                tau_int=bval["tau_int"], tau_int_err=bval["tau_int_err"],
                extrapolation_method=extrapolation_method, plateau_size=10,
                inverse_fit=True, **kwargs)

            a_values.append(bval["a"]**2/t0)
            a_values_err.append(np.sqrt((2*bval["a_err"]*bval["a"]/t0)**2
                                        + (bval["a"]**2*t0_err/t0**2)**2))

            t0_values.append(t0)
            t0err_values.append(t0_err)

        a_values = np.asarray(a_values[::-1])
        a_values_err = np.asarray(a_values_err[::-1])
        t0_values = np.asarray(t0_values[::-1])
        t0err_values = np.asarray(t0err_values[::-1])

        # Functions for t0 and propagating uncertainty
        def t0_func(_t0): return np.sqrt(8*_t0)/self.r0

        def t0err_func(_t0, _t0_err): return _t0_err * \
            np.sqrt(8/_t0)/(2.0*self.r0)

        # Sets up t0 and t0_error values to plot
        y = t0_func(t0_values)
        yerr = t0err_func(t0_values, t0err_values)

        # Extrapolates t0 to continuum
        N_cont = 1000
        a_squared_cont = np.linspace(-0.025, a_values[-1]*1.1, N_cont)

        # Fits to continuum and retrieves values to be plotted
        continuum_fit = LineFit(a_values, y, y_err=yerr)
        y_cont, y_cont_err, fit_params, chi_squared = \
            continuum_fit.fit_weighted(a_squared_cont)

        res = continuum_fit(0, weighted=True)
        self.sqrt_8t0_cont = res[0][0]
        self.sqrt_8t0_cont_error = (res[1][-1][0] - res[1][0][0])/2
        self.t0_cont = self.sqrt_8t0_cont**2/8
        self.t0_cont_error = self.sqrt_8t0_cont_error*np.sqrt(self.t0_cont/2.0)

        # Creates continuum extrapolation figure
        fname = os.path.join(
            self.output_folder_path,
            "post_analysis_extrapmethod%s_t0reference_continuum_%s.pdf" % (
                extrapolation_method, self.analysis_data_type))
        self.plot_continuum_fit(a_squared_cont, y_cont, y_cont_err,
                                chi_squared, a_values, a_values_err,
                                y, yerr, 0, 0,
                                self.sqrt_8t0_cont, self.sqrt_8t0_cont_error,
                                r"\frac{\sqrt{8t_{0,\mathrm{cont}}}}{r_0}",
                                fname, r"$\frac{\sqrt{8t_0}}{r_0}$",
                                r"$a^2/t_0$")

        self.extrapolation_method = extrapolation_method

        # Reverses values for storage.
        a_values, a_values_err, t0_values, t0err_values = map(
            lambda k: np.flip(k, 0),
            (a_values, a_values_err, t0_values, t0err_values))

        _tmp_beta_dict = {
            b: {
                "t0": t0_values[i],
                "t0err": t0err_values[i],
                "t0a2": t0_values[i]/self.plot_values[b]["a"]**2,
                # Including error term in lattice spacing, a
                "t0a2err": np.sqrt((t0err_values[i] / \
                                    self.plot_values[b]["a"]**2)**2 \
                                   + (2*self.plot_values[b]["a_err"] * \
                                      t0_values[i] /\
                                      self.plot_values[b]["a"]**3)**2),
                "t0r02": t0_values[i]/self.r0**2,
                "t0r02err": t0err_values[i]/self.r0**2,
                "aL": self.plot_values[b]["a"]*self.lattice_sizes[b][0],
                "aLerr": (self.plot_values[b]["a_err"] \
                          * self.lattice_sizes[b][0]),
                "L": self.lattice_sizes[b][0],
                "a": self.plot_values[beta]["a"],
                "a_err": self.plot_values[b]["a_err"],
            }
            for i, b in enumerate(self.beta_values)
        }

        t0_dict = {"t0cont": self.t0_cont, "t0cont_err": self.t0_cont_error}
        t0_dict.update(_tmp_beta_dict)

        if self.verbose:
            print "t0 reference values table: "
            print "sqrt(8t0)/r0 = %.16f +/- %.16f" % (
                self.sqrt_8t0_cont, self.sqrt_8t0_cont_error)
            print "t0 = %.16f +/- %.16f" % (self.t0_cont,
                                            self.t0_cont_error)
            for b in self.beta_values:
                msg = "beta = %.2f || t0 = %10f +/- %-10f" % (
                    b, t0_dict[b]["t0"], t0_dict[b]["t0err"])
                msg += " || t0/a^2 = %10f +/- %-10f" % (
                    t0_dict[b]["t0a2"], t0_dict[b]["t0a2err"])
                msg += " || t0/r0^2 = %10f +/- %-10f" % (
                    t0_dict[b]["t0r02"], t0_dict[b]["t0r02err"])
                print msg

        if self.print_latex:
            # Header:
            # beta   t0a2   t0r02   L/a   L   a

            header = [r"$\beta$", r"$t_0/a^2$", r"$t_0/{r_0^2}$", r"$L/a$",
                      r"$L[\fm]$", r"$a[\fm]$"]

            bvals = self.beta_values
            tab = [
                [r"{0:.2f}".format(b) for b in bvals],
                [r"{0:s}".format(sciprint.sciprint(
                    t0_dict[b]["t0a2"],
                    t0_dict[b]["t0a2err"])) for b in bvals],
                [r"{0:s}".format(sciprint.sciprint(
                    t0_dict[b]["t0r02"],
                    t0_dict[b]["t0r02err"])) for b in bvals],
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

        return t0_dict

    def get_w0_scale(self, extrapolation_method="bootstrap", W0=0.3,
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
        w0a_values = []
        w0aerr_values = []

        for beta, bval in sorted(self.plot_values.items(), key=lambda i: i[0]):
            y0, t_w0, t_w0_err, _, _ = extract_fit_target(
                W0, bval["tder"], bval["W"],
                y_err=bval["W_err"], y_raw=bval["W_raw"],
                tau_int=bval["tau_int"][1:-1],
                tau_int_err=bval["tau_int_err"][1:-1],
                extrapolation_method=extrapolation_method, plateau_size=10,
                inverse_fit=True, **kwargs)

            # NOTE: w0 has units of [fm].
            # t_w0 = a^2 * t_f / r0^2
            # Returns t_w0 = (w0)^2 / r0^2.

            # Lattice spacing
            a_values.append(bval["a"]**2)
            a_values_err.append(2*bval["a_err"]*bval["a"])

            # Plain w_0 retrieval
            w0_values.append(np.sqrt(t_w0))
            w0err_values.append(0.5*t_w0_err/np.sqrt(t_w0))

            # w_0 / a
            w0a_values.append(np.sqrt(t_w0)/bval["a"])
            w0aerr_values.append(
                np.sqrt((t_w0_err/(2*np.sqrt(t_w0))/bval["a"])**2
                        + (np.sqrt(t_w0)*bval["a_err"]/bval["a"]**2)**2))

        # Reverse lists and converts them to arrays
        a_values = np.asarray(a_values[::-1])
        a_values_err = np.asarray(a_values_err[::-1])
        w0_values = np.asarray(w0_values[::-1])
        w0err_values = np.asarray(w0err_values[::-1])
        w0a_values, w0aerr_values = map(np.asarray,
                                        (w0a_values, w0aerr_values))

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
        self.sqrt8w0_cont = np.sqrt(8*self.w0_cont)
        self.sqrt8w0_cont_error = 4*self.w0_cont_error/np.sqrt(8*self.w0_cont)

        # Creates continuum extrapolation plot
        fname = os.path.join(
            self.output_folder_path,
            "post_analysis_extrapmethod%s_w0reference_continuum_%s.pdf" % (
                extrapolation_method, self.analysis_data_type))
        self.plot_continuum_fit(a_squared_cont, y_cont, y_cont_err,
                                chi_squared, a_values, a_values_err,
                                w0_values, w0err_values,
                                0, 0, self.w0_cont, self.w0_cont_error,
                                r"w_{0,\mathrm{cont}}", fname,
                                r"$w_0[\mathrm{fm}]$",
                                r"$a^2[\mathrm{GeV}^{-2}]$")

        # Reverses values for storage.
        a_values, a_values_err, w0_values, w0err_values, = map(
            lambda k: np.flip(k, 0),
            (a_values, a_values_err, w0_values, w0err_values))

        # Populates dictionary with w0 values for each beta value
        _tmp_beta_dict = {
            b: {
                "w0": w0_values[i],
                "w0err": w0err_values[i],
                "w0a": w0a_values[i],
                "w0aerr": w0aerr_values[i],
                "w0r0": w0_values[i]/self.r0,
                "w0r0err": w0err_values[i]/self.r0,
                "aL": self.plot_values[b]["a"]*self.lattice_sizes[b][0],
                "aLerr": (self.plot_values[b]["a_err"]
                          * self.lattice_sizes[b][0]),
                "L": self.lattice_sizes[b][0],
                "a": self.plot_values[beta]["a"],
                "a_err": self.plot_values[b]["a_err"],
            }
            for i, b in enumerate(sorted(self.beta_values))
        }

        # Populates dictionary with continuum w0 value
        w0_dict = {
            "w0cont": self.w0_cont,
            "w0cont_err": self.w0_cont_error,
        }
        w0_dict.update(_tmp_beta_dict)

        if self.verbose:
            print "w0 reference values table: "
            print "w0 = %.16f +/- %.16f" % (self.w0_cont, self.w0_cont_error)
            for b in sorted(self.beta_values):
                msg = "beta = %.2f" % b
                msg += " || w0 = %10f +/- %-10f" % (
                    w0_dict[b]["w0"], w0_dict[b]["w0err"])
                msg += " || w0/a = %10f +/- %-10f" % (
                    w0_dict[b]["w0a"], w0_dict[b]["w0aerr"])
                msg += " || w0/r0 = %10f +/- %-10f" % (
                    w0_dict[b]["w0r0"], w0_dict[b]["w0r0err"])
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
                [r"{0:s}".format(sciprint.sciprint(
                    w0_dict[b]["w0"], w0_dict[b]["w0err"])) for b in bvals],
                [r"{0:s}".format(sciprint.sciprint(
                    self.plot_values[b]["a"]**2,
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

        return w0_dict

    def plot_continuum_fit(self, a_squared_cont, y_cont, y_cont_err,
                           chi_squared, x_fit, x_fit_err, y_fit, y_fit_err,
                           a0_cont, a0err_cont, y0_cont, y0err_cont,
                           cont_label, figname, xlabel, ylabel):
        """
        Creates continuum extrapolation plot.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plots an ax-line at 0
        ax.axvline(0, linestyle="dashed",
                   color=self.cont_axvline_color, zorder=5, linewidth=1.0)

        # Plots the fit
        ax.plot(a_squared_cont, y_cont, color=self.fit_color, alpha=0.5,
                label=r"$\chi=%.2f$" % chi_squared, zorder=10)
        ax.fill_between(a_squared_cont, y_cont_err[0],
                        y_cont_err[1], alpha=0.5, edgecolor='',
                        facecolor=self.fit_fill_color, zorder=0)

        # Plots lattice points
        ax.errorbar(x_fit, y_fit, xerr=x_fit_err, yerr=y_fit_err, fmt="o",
                    capsize=5, capthick=1, color=self.lattice_points_color,
                    ecolor=self.lattice_points_color, zorder=15)

        # Plots the continuum limit errorbar
        ax.errorbar(a0_cont, y0_cont,
                    xerr=[[a0err_cont], [a0err_cont]],
                    yerr=[[y0err_cont], [y0err_cont]],
                    fmt="o", capsize=5, capthick=1, 
                    color=self.cont_error_color, elinewidth=2.0,
                    ecolor=self.cont_error_color, 
                    label=r"$%s=%.3f\pm%.3f$" % (
                        cont_label, y0_cont, y0err_cont), zorder=15)

        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        ax.set_xlim(a_squared_cont[0], a_squared_cont[-1])
        ax.legend()
        ax.grid(True)

        # Saves figure
        fig.savefig(figname, dpi=self.dpi)
        if self.verbose:
            print "Figure saved in %s" % figname

        plt.close(fig)

    def plot_w(self, *args, **kwargs):
        """Plots the W(t)."""
        w_plot_values = copy.deepcopy(self.plot_values)
        for beta in sorted(self.beta_values):
            w_plot_values[beta]["x"] = self.plot_values[beta]["x"][1:-1]
            w_plot_values[beta]["y"] = self.plot_values[beta]["W"]
            w_plot_values[beta]["y_err"] = self.plot_values[beta]["W_err"]

        kwargs["observable_name_compact"] = "energyW"
        kwargs["x_label"] = r"$t_f/r_0^2$"
        kwargs["y_label"] = r"$W(t)$"
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
