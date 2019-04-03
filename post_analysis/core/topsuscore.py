import types
import copy
import os
import numpy as np
from postcore import PostCore
from tools.latticefunctions import get_lattice_spacing
from tools.latticefunctions import witten_veneziano
from statistics.linefit import LineFit, extract_fit_target
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt


class TopsusCore(PostCore):
    """Core class for all topological susceptiblity calculations."""
    observable_name = "topsus_core"
    observable_name_compact = "topsus_core"
    obs_name_latex = "MISSING LATEX NAME FOR TOPSUS"

    # Regular plot variables
    x_label = r"$\sqrt{8t_f}$ [fm]"

    # Continuum plot variables
    y_label_continuum = r"$\chi_{t_f}^{1/4}$ [GeV]"
    # x_label_continuum = r"$a/{{r_0}^2}$"
    x_label_continuum = r"$a^2/t_0$"

    # For specialized observables
    extra_continuum_msg = ""

    # For topsus function
    hbarc = 0.19732697  # eV micro m

    chi_const = {}
    chi_const_err = {}
    chi = {}
    chi_der = {}

    # For description in printing the different parameters from fit
    descr = ""

    # Description for the extrapolation method in fit
    extrapolation_method = ""

    # String for intervals
    intervals_str = ""

    include_title = False

    def __init__(self, *args, **kwargs):
        super(TopsusCore, self).__init__(*args, **kwargs)
        self._initialize_topsus_func()

    def set_analysis_data_type(self, *args, **kwargs):
        self._initialize_topsus_func()
        super(TopsusCore, self).set_analysis_data_type(*args, **kwargs)

    def _initialize_topsus_func_const(self):
        """Sets the constant in the topsus function for found batches."""
        for bn in self.batch_names:
            a, a_err = get_lattice_spacing(self.beta_values[bn])
            V = float(self.lattice_sizes[bn][0]**3 * self.lattice_sizes[bn][1])
            self.chi_const[bn] = self.hbarc/a/V**0.25
            self.chi_const_err[bn] = self.hbarc*a_err/a**2/V**0.25

    def _initialize_topsus_func(self):
        """Sets the topsus function for all found batches."""

        self._initialize_topsus_func_const()

        for bn in self.batch_names:
            self.chi[bn] = Chi(self.chi_const[bn])
            self.chi_der[bn] = ChiDer(
                self.chi_const[bn], self.chi_const_err[bn])

        return

    def get_fit_targets(self, fit_target):
        """Sets up the fit targets."""

        if isinstance(fit_target, str):
            tmp_ref = (self.reference_values[self.extrapolation_method]
                       [self.analysis_data_type])

            # Either t0 or w0
            if fit_target == "t0":

                # Sets up sqrt(8*t0)
                fit_targets = [np.sqrt(8*tmp_ref[bn]["t0"])
                               for bn in self.batch_names]
                self.fit_target = r"\sqrt{8t_0}=[%s]" % (
                    str(", ".join(["{0:.4f}".format(_t)
                                   for _t in fit_targets])))

            elif fit_target == "w0":

                # Sets up sqrt(8*w0^2)
                fit_targets = [np.sqrt(8*tmp_ref[bn]["w0"]**2)
                               for bn in self.batch_names]
                self.fit_target = r"\sqrt{8w_0^2}=[%s]" % (
                    ", ".join(["{0:.4f}".format(_t) for _t in fit_targets]))

            elif fit_target == "t0cont":

                # Continuum fit of t0
                fit_targets = [np.sqrt(8*tmp_ref["t0cont"])
                               for bn in self.batch_names]
                self.fit_target = (
                    r"\sqrt{8 t_{0,\mathrm{cont}}}=%.4f"
                    % np.sqrt(8*tmp_ref["t0cont"]*self.r0**2))

            elif fit_target == "w0cont":

                # Continuum fit of w0
                fit_targets = [np.sqrt(8*tmp_ref["w0cont"]**2)
                               for bn in self.batch_names]
                self.fit_target = (r"\sqrt{8 w_{0,\mathrm{cont}}^2} = %.4f"
                                   % np.sqrt(8*(tmp_ref["w0cont"])**2))

            else:

                raise KeyError("Fit target key '{}' not "
                               "reecognized.".format(fit_target))

        elif isinstance(fit_target, float):
            fit_targets = [fit_target for bn in self.batch_names]
            self.fit_target = (r"\sqrt{8 t_{f}} = %.2f" %
                               fit_target)
        else:
            raise ValueError("Fit target not recognized: "
                             "{}".format(fit_target))

        return fit_targets

    def check_continuum_extrapolation(self):
        """
        Small method for checking if on can do a continuum extrapolation. If
        we have several beta values for a lattice spacing, this will not be 
        possible.
        """
        if len(list(set(self.beta_values.values()))) != len(self.batch_names):
            print("Multiple values for a beta value: {} --> Skipping"
                  " continuum extrapolation".format(self.beta_values.values()))
            return False
        else:
            return True

    def plot_continuum(self, fit_target, title_addendum="",
                       extrapolation_method="bootstrap",
                       plateau_fit_size=20, interpolation_rank=3,
                       plot_continuum_fit=False):
        """Method for plotting the continuum limit of topsus at a given
        fit_target.

        Args:
            fit_target: float or str. If float, will choose corresponding
                float time t_f value of where we extrapolate from. If string,
                one can choose either 't0', 't0cont', 'w0' and 'w0cont'. 't0' 
                and 'w0' will use values for given batch. For 't0cont' 
                and 'w0cont' will use extrapolated values to select topsus 
                values from.
            title_addendum: str, optional, default is an empty string, ''.
                Adds string to end of title.
            extrapolation_method: str, optional, method of selecting the
                extrapolation point to do the continuum limit. Method will be
                used on y values and tau int. Choices:
                    - plateau: line fits points neighbouring point in order to
                        reduce the error bars using y_raw for covariance
                        matrix.
                    - plateau_mean: line fits points neighbouring point in
                        order to reduce the error bars. Line will be weighted
                        by the y_err.
                    - nearest: line fit from the point nearest to what we seek
                    - interpolate: linear interpolation in order to retrieve
                        value and error. Does not work in conjecture with
                        use_raw_values.
                    - bootstrap: will create multiple line fits, and take
                        average. Assumes y_raw is the bootstrapped or
                        jackknifed samples.
            plateau_size: int, optional. Number of points in positive and
                negative direction to extrapolate fit target value from. This
                value also applies to the interpolation interval. Default is
                20.
            interpolation_rank: int, optional. Interpolation rank to use if
                extrapolation method is interpolation Default is 3.
            raw_func: function, optional, will modify the bootstrap data after
                samples has been taken by this function.
            raw_func_err: function, optional, will propagate the error of the
                bootstrapped line fitted data, raw_func_err(y, yerr).
                Calculated by regular error propagation.
        """

        if not self.check_continuum_extrapolation(): return

        self.extrapolation_method = extrapolation_method
        if not isinstance(self.reference_values, types.NoneType):
            if extrapolation_method in self.reference_values.keys():
                # Checking that the extrapolation method selected can be used.
                t0_values = (self.reference_values[self.extrapolation_method]
                             [self.analysis_data_type])
            else:
                # If the extrapolation method is not among the used methods.
                t0_values = self.reference_values.values()[
                    0][self.analysis_data_type]
        else:
            t0_values = None

        # Retrieves data for analysis.
        if fit_target == -1:
            fit_target = self.plot_values[max(self.plot_values)]["x"][-1]

        fit_targets = self.get_fit_targets(fit_target)
        if self.verbose:
            print "Fit targets: ", fit_targets

        a, a_err, a_norm_factor, a_norm_factor_err, obs, obs_raw, obs_err, \
            tau_int_corr = [], [], [], [], [], [], [], []

        for i, bn in enumerate(self.sorted_batch_names):
            if self.with_autocorr and not "blocked" in self.analysis_data_type:
                tau_int = self.plot_values[bn]["tau_int"]
                tau_int_err = self.plot_values[bn]["tau_int_err"]
            else:
                tau_int = None
                tau_int_err = None

            # plt.plot(self.plot_values[bn]["x"], self.chi[bn](
            #     np.mean(self.plot_values[bn]["y_raw"], axis=1)))
            # plt.plot(self.plot_values[bn]["x"],
            #          self.plot_values[bn]["y"], color="red")
            # plt.show()
            # exit("Good @ 255")

            # Extrapolation of point to use in continuum extrapolation
            res = extract_fit_target(
                fit_targets[i],  self.plot_values[bn]["x"],
                self.plot_values[bn]["y"],
                self.plot_values[bn]["y_err"],
                y_raw=self.plot_values[bn]["y_raw"],
                tau_int=tau_int, tau_int_err=tau_int_err,
                extrapolation_method=extrapolation_method,
                plateau_size=plateau_fit_size, interpolation_rank=3,
                plot_fit=plot_continuum_fit, raw_func=self.chi[bn],
                raw_func_err=self.chi_der[bn], plot_samples=False,
                verbose=False)

            _x0, _y0, _y0_error, _y0_raw, _tau_int0 = res

            # In case something is wrong -> skip
            if np.isnan([_y0, _y0_error]).any():
                print "NaN type detected: skipping calculation"
                return

            if self.verbose:
                msg = "Beta = %4.2f Topsus = %14.12f +/- %14.12f" % (
                    self.beta_values[bn], _y0, _y0_error)

            a.append(self.plot_values[bn]["a"])
            a_err.append(self.plot_values[bn]["a_err"])

            if isinstance(t0_values, types.NoneType):
                a_norm_factor.append(_x0)
                a_norm_factor_err.append(0)
            else:
                a_norm_factor.append(t0_values["t0cont"])
                a_norm_factor_err.append(t0_values["t0cont_err"])

                if self.verbose:
                    _tmp = t0_values["t0cont"]/(self.plot_values[bn]["a"]**2)
                    _tmp *= self.r0**2
                    msg += " t0 = %14.12f" % (_tmp)

            if self.verbose:
                print msg

            obs.append(_y0)
            obs_err.append(_y0_error)
            obs_raw.append(_y0_raw)
            tau_int_corr.append(_tau_int0)

        # Makes lists into arrays
        a = np.asarray(a)[::-1]
        a_err = np.asarray(a_err)[::-1]
        a_norm_factor = np.asarray(a_norm_factor)[::-1]
        a_norm_factor_err = np.asarray(a_norm_factor_err)[::-1]
        a_squared = a**2 / a_norm_factor
        a_squared_err = np.sqrt((2*a*a_err/a_norm_factor)**2
                                + (a**2*a_norm_factor_err/a_norm_factor**2)**2)
        obs = np.asarray(obs)[::-1]
        obs_err = np.asarray(obs_err)[::-1]

        # Continuum limit arrays
        N_cont = 1000
        a_squared_cont = np.linspace(-0.0025, a_squared[-1]*1.1, N_cont)

        # Fits to continuum and retrieves values to be plotted
        continuum_fit = LineFit(a_squared, obs, obs_err)

        y_cont, y_cont_err, fit_params, chi_squared = \
            continuum_fit.fit_weighted(a_squared_cont)
        self.chi_squared = chi_squared
        self.fit_params = fit_params

        # continuum_fit.plot(True)

        # Gets the continium value and its error
        y0_cont, y0_cont_err, _, _, = \
            continuum_fit.fit_weighted(0.0)

        # Matplotlib requires 2 point to plot error bars at
        a0_squared = [0, 0]
        y0 = [y0_cont[0], y0_cont[0]]
        y0_err = [y0_cont_err[0][0], y0_cont_err[1][0]]

        # Stores the chi continuum
        self.topsus_continuum = y0[0]
        self.topsus_continuum_error = (y0_err[1] - y0_err[0])/2.0

        y0_err = [self.topsus_continuum_error, self.topsus_continuum_error]

        # Sets of title string with the chi squared and fit target
        if isinstance(self.fit_target, str):
            title_string = r"$%s, \chi^2/\mathrm{d.o.f.} = %.2f$" % (
                self.fit_target, self.chi_squared)
        else:
            title_string = r"$\sqrt{8t_{f,\mathrm{extrap}}} = %.2f[fm], \chi^2 = %.2f$" % (
                self.fit_target, self.chi_squared)
        title_string += title_addendum

        # Creates figure and plot window
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plots an ax-line at 0
        ax.axvline(0, linestyle="dashed",
                   color=self.cont_axvline_color, linewidth=0.5)

        # Plots the fit
        ax.plot(a_squared_cont, y_cont, color=self.fit_color, alpha=0.5)
        ax.fill_between(a_squared_cont, y_cont_err[0], y_cont_err[1],
                        alpha=0.5, edgecolor='', facecolor=self.fit_fill_color)

        # Plot lattice points
        ax.errorbar(a_squared, obs, xerr=a_squared_err, yerr=obs_err, fmt="o",
                    capsize=5, capthick=1, color=self.lattice_points_color,
                    ecolor=self.lattice_points_color)

        # plots continuum limit, 5 is a good value for cap size
        ax.errorbar(a0_squared, y0,
                    yerr=y0_err, fmt="o", capsize=5,
                    capthick=1, color=self.cont_error_color,
                    ecolor=self.cont_error_color,
                    label=r"$\chi_{t_f}^{1/4}=%.3f\pm%.3f$" % (
                        self.topsus_continuum, self.topsus_continuum_error))

        ax.set_ylabel(self.y_label_continuum)
        ax.set_xlabel(self.x_label_continuum)
        if self.include_title:
            ax.set_title(title_string)
        ax.set_xlim(a_squared_cont[0], a_squared_cont[-1])
        ax.legend()
        ax.grid(True)

        if self.verbose:
            print "Target: %.16f +/- %.16f" % (self.topsus_continuum,
                                               self.topsus_continuum_error)

        # Saves figure
        fname = os.path.join(
            self.output_folder_path,
            "post_analysis_extrapmethod%s_%s_continuum%s_%s.pdf" % (
                extrapolation_method, self.observable_name_compact,
                str(fit_target).replace(".", ""), self.analysis_data_type))

        fig.savefig(fname, dpi=self.dpi)

        if self.verbose:
            print "Continuum plot of %s created in %s" % (
                self.observable_name_compact, fname)

        # plt.show()
        plt.close(fig)

        self.print_continuum_estimate()

    def get_linefit_parameters(self):
        """Returns the chi^2, a, a_err, b, b_err."""
        return self.chi_squared, self.fit_params, self.topsus_continuum, \
            self.topsus_continuum_error, self.NF, self.NF_error, \
            self.fit_target, self.intervals_str, self.descr, \
            self.extrapolation_method, self.obs_name_latex

    def print_continuum_estimate(self):
        """Prints the NF from the Witten-Veneziano formula."""
        self.NF, self.NF_error = witten_veneziano(self.topsus_continuum,
                                                  self.topsus_continuum_error)
        msg = "Observable: %s" % self.observable_name_compact
        if isinstance(self.fit_target, str):
            msg += "\n    Fit target: %s" % self.fit_target
        else:
            msg += "\n    Fit target: %.4f" % self.fit_target
        msg += "\n    Topsus = %.16f" % self.topsus_continuum
        msg += "\n    Topsus_error = %.16f" % self.topsus_continuum_error
        msg += "\n    N_f = %.16f" % self.NF
        msg += "\n    N_f_error = %.16f" % self.NF_error
        msg += "\n    Chi^2 = %.16f" % self.chi_squared
        msg += self.extra_continuum_msg
        msg += "\n"
        print msg


class Chi:
    def __init__(self, const):
        self.const = const

    def __call__(self, qq):
        return self.const*qq**(0.25)


class ChiDer:
    def __init__(self, const, const_err):
        self.const = const
        self.const_err = const_err

    def __call__(self, qq, qq_err):
        return np.sqrt((self.const_err*qq**0.25)**2 +
                       (0.25*self.const_err*qq_err/qq**(0.75))**2)


def main():
    exit("Exit: TopsusCore not intended to be a standalone module.")


if __name__ == '__main__':
    main()
