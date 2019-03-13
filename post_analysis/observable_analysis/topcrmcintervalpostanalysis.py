from post_analysis.core.postcore import PostCore
from post_analysis.core.multiplotcore import MultiPlotCore
from post_analysis.observable_analysis.topcrpostanalysis import \
    TopcRPostAnalysis
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
from tools.sciprint import sciprint
from tools.table_printer import TablePrinter
from collections import OrderedDict
import numpy as np
import os
import types


class TopcRMCIntervalPostAnalysis(TopcRPostAnalysis, MultiPlotCore, PostCore):
    """
    Post-analysis of the topc ratio with Q^4_C/Q^2. Requires that Q4 and Q2 
    has been imported.
    """

    observable_name = r"$R=\frac{\langle Q^4 \rangle_C}{\langle Q^2 \rangle}$"
    observable_name_compact = "topcrMC"
    x_label = r"$\sqrt{8t_{f}}$ [fm]"
    y_label = r"$R$"

    formula = r", $R = \langle Q^4_C \rangle = \langle Q^4 \rangle - $"
    formula += r"$3 \langle Q^2 \rangle^2$"

    print_latex = False

    dpi = 400

    lattice_sizes = {}

    def __init__(self, data, with_autocorr=True, figures_folder="../figures",
                 verbose=False, dryrun=False):
        """
        Initializes this specialized form of finding the ratio of different 
        topological charge definitions.
        """
        if with_autocorr:
            self.ac = "with_autocorr"
        else:
            self.ac = "without_autocorr"
        self.with_autocorr = with_autocorr
        observable = self.observable_name_compact

        self.verbose = verbose
        self.dryrun = dryrun
        
        self.beta_values = data.beta_values
        self.batch_names = data.batch_names
        self.colors = data.colors
        self.lattice_sizes = data.lattice_sizes
        self.lattice_volumes = data.lattice_volumes
        self.size_labels = data.labels
        self.reference_values = data.reference_values
        self._setup_analysis_types(data.analysis_types)
        self.print_latex = data.print_latex

        self.data = {atype: {b: {} for b in self.batch_names}
                     for atype in self.analysis_types}

        self.data_map = {bn: {"lattice_volume": self.lattice_volumes[bn],
                              "beta": self.beta_values[bn]}
                         for bn in self.batch_names}

        self.flow_epsilon = {bn: data.flow_epsilon[bn]
                             for bn in self.batch_names}

        self.sorted_batch_names = sorted(self.batch_names, key=lambda _k: (
            self.data_map[_k]["beta"], self.data_map[_k]["lattice_volume"]))

        self.observable_intervals = {bn: {} for bn in self.batch_names_sorted}

        # Q^2
        self.topc2 = {atype: {bn: {} for bn in self.batch_names}
                      for atype in self.analysis_types}

        # Q^4
        self.topc4 = {atype: {bn: {} for bn in self.batch_names}
                      for atype in self.analysis_types}

        # Q^4_C
        self.topc4C = {atype: {bn: {} for bn in self.batch_names}
                       for atype in self.analysis_types}

        # R = Q^4_C / Q^2
        self.topcR = {atype: {bn: {} for bn in self.batch_names}
                      for atype in self.analysis_types}

        # Data will be copied from R
        self.data = {atype: {bn: {} for bn in self.batch_names}
                     for atype in self.analysis_types}

        # Q^2 and Q^4 raw bs values
        self.topc2_raw = {}
        self.topc4_raw = {}
        self.topc4c_raw = {}
        self.topcR_raw = {}
        self.data_raw = {}
        self.ac_raw = {}

        for atype in data.raw_analysis:
            if atype == "autocorrelation":
                self.ac_raw["tau"] = data.raw_analysis[atype]
            elif atype == "autocorrelation_raw":
                self.ac_raw["ac_raw"] = data.raw_analysis[atype]
            elif atype == "autocorrelation_raw_error":
                self.ac_raw["ac_raw_error"] = data.raw_analysis[atype]
            else:
                self.data_raw[atype] = data.raw_analysis[atype]

        # First, gets the topc2, then topc4
        for atype in self.analysis_types:
            for bn in self.batch_names_sorted:
                self.observable_intervals[bn] = \
                    data.data_observables["topc2MC"][bn].keys()

                for subobs in data.data_observables["topc2MC"][bn]:

                    # Q^2
                    self.topc2[atype][bn][subobs] = \
                        data.data_observables["topc2MC"][bn][subobs][self.ac][atype]

                    # Q^4
                    self.topc4[atype][bn][subobs] = \
                        data.data_observables["topc4MC"][bn][subobs][self.ac][atype]

                    if self.with_autocorr:
                        self.topc2[atype][bn][subobs]["ac"] = \
                            data.data_observables["topc2MC"][bn][subobs]["with_autocorr"]["autocorr"]

                        self.topc4[atype][bn][subobs]["ac"] = \
                            data.data_observables["topc4MC"][bn][subobs]["with_autocorr"]["autocorr"]

        # Creates base output folder for post analysis figures
        self.figures_folder = figures_folder
        check_folder(self.figures_folder, dryrun=self.dryrun,
                     verbose=self.verbose)
        check_folder(os.path.join(self.figures_folder, data.batch_name),
                     dryrun=self.dryrun, verbose=self.verbose)

        # Creates output folder
        self.post_anlaysis_folder = os.path.join(self.figures_folder,
                                                 data.batch_name, "post_analysis")
        check_folder(self.post_anlaysis_folder, dryrun=self.dryrun,
                     verbose=self.verbose)

        # Creates observable output folder
        self.output_folder_path = os.path.join(self.post_anlaysis_folder,
                                               self.observable_name_compact)
        check_folder(self.output_folder_path, dryrun=self.dryrun,
                     verbose=self.verbose)

        self._setup_article_values()
        self._normalize_article_values()

        self._setup_volumes()
        self._normalize_Q()
        self._calculate_Q4C()
        self._calculate_R()

    def _normalize_Q(self):
        """Normalizes Q4 and Q2"""
        for atype in self.analysis_types:
            for bn in self.batch_names_sorted:
                for subobs in self.observable_intervals[bn]:
                    # self.topc2[atype][bn][subobs]["y"] /= self.V[bn]
                    # self.topc2[atype][bn][subobs]["y_error"] /= self.V[bn]
                    # self.topc4[atype][bn][subobs]["y"] /= self.V[bn]**2
                    # self.topc4[atype][bn][subobs]["y_error"] /= \
                    #     self.V[bn]**2

                    self.topc2[atype][bn][subobs]["y_error"] = np.sqrt(
                        (self.topc2[atype][bn][subobs]["y_error"]/self.V[bn])**2 +
                        (self.V_err[bn]*self.topc2[atype][bn][subobs]["y"]/self.V[bn]**2)**2)
                    self.topc2[atype][bn][subobs]["y"] /= self.V[bn]

                    self.topc4[atype][bn][subobs]["y_error"] = np.sqrt(
                        (self.topc4[atype][bn][subobs]["y_error"]/self.V[bn]**2)**2 +
                        (2*self.V_err[bn]*self.topc4[atype][bn][subobs]["y"]/self.V[bn]**3)**2)
                    self.topc4[atype][bn][subobs]["y"] /= self.V[bn]**2

    def _calculate_Q4C(self):
        """Caluclates the 4th cumulant for my data."""

        # Gets Q4C and R
        for atype in self.analysis_types:

            for bn in self.batch_names_sorted:

                for subobs in self.observable_intervals[bn]:
                    self.topc4C[atype][bn][subobs] = {

                        "x": self.topc2[atype][bn][subobs]["x"],

                        "y": self.Q4C(
                            self.topc4[atype][bn][subobs]["y"],
                            self.topc2[atype][bn][subobs]["y"]),

                        "y_error": self.Q4C_error(
                            self.topc4[atype][bn][subobs]["y"],
                            self.topc4[atype][bn][subobs]["y_error"],
                            self.topc2[atype][bn][subobs]["y"],
                            self.topc2[atype][bn][subobs]["y_error"]),
                    }

    def _calculate_R(self):
        """Calculates R = Q^4_C / Q^2 for my data."""

        # Gets Q4C and R
        for atype in self.analysis_types:

            for bn in self.batch_names_sorted:

                for subobs in self.observable_intervals[bn]:

                    self.topcR[atype][bn][subobs] = {

                        "x": self.topc2[atype][bn][subobs]["x"],

                        "y": self.R(
                            self.topc4C[atype][bn][subobs]["y"],
                            self.topc2[atype][bn][subobs]["y"]),

                        "y_error": self.R_error(
                            self.topc4C[atype][bn][subobs]["y"],
                            self.topc4C[atype][bn][subobs]["y_error"],
                            self.topc2[atype][bn][subobs]["y"],
                            self.topc2[atype][bn][subobs]["y_error"])
                    }
                    self.data[atype][bn][subobs] = \
                        self.topcR[atype][bn][subobs]

    def _setup_analysis_types(self, atypes):
        """
        Stores the number of analysis types from the data container, while 
        removing the autocorrelation one.
        """
        self.analysis_types = atypes
        if "autocorrelation" in self.analysis_types:
            self.analysis_types.remove("autocorrelation")

    def _check_plot_values(self):
        """Checks if we have set the analysis data type yet."""
        if not hasattr(self, "plot_values"):
            raise AttributeError(
                "set_analysis_data_type() has not been set yet.")

    def set_analysis_data_type(self, analysis_data_type="bootstrap"):
        """Sets the analysis type and retrieves correct analysis data."""

        # Makes it a global constant so it can be added in plot figure name
        self.analysis_data_type = analysis_data_type

        self.plot_values = {}  # Clears old plot values
        # print self.data.keys()
        # print self.data_raw.keys()
        # print analysis_data_type
        # self._initiate_plot_values(self.data[analysis_data_type],
        #     self.data_raw[analysis_data_type])

    def _initiate_plot_values(self, data, data_raw, interval_keys=None):
        """Sorts data into a format specific for the plotting method."""
        for ib, bn in enumerate(self.sorted_batch_names):
            values = {}
            if isinstance(interval_keys, types.NoneType):
                # Case where we have sub sections of observables, e.g. in
                # euclidean time.
                for sub_obs in self.observable_intervals[bn]:
                    sub_values = {}
                    sub_values["a"], sub_values["a_err"] = \
                        get_lattice_spacing(self.beta_values[bn])
                    sub_values["x"] = sub_values["a"] * \
                        np.sqrt(8*data[bn][sub_obs]["x"])
                    sub_values["y"] = data[bn][sub_obs]["y"]
                    sub_values["y_err"] = data[bn][sub_obs]["y_error"]
                    sub_values["label"] = r"%s, $\beta=%2.2f$, %s" % (
                        self.size_labels[bn], self.beta_values[bn],
                        self._convert_label(sub_obs))
                    values[sub_obs] = sub_values
            else:
                int_key = interval_keys[ib]
                values["a"], values["a_err"] = \
                    get_lattice_spacing(self.beta_values[bn])
                values["x"] = values["a"] * np.sqrt(8*data[bn][int_key]["x"])
                values["y"] = data[bn][int_key]["y"]
                values["y_err"] = data[bn][int_key]["y_error"]
                values["label"] = r"%s, $\beta=%2.2f$, %s" % (
                    self.size_labels[bn], self.beta_values[bn],
                    self._convert_label(int_key))
                values["interval"] = int_key
            self.plot_values[beta] = values

    def plot(self, *args, **kwargs):
        """Ensuring I am plotting with formule in title."""
        kwargs["plot_with_formula"] = True
        kwargs["error_shape"] = "band"
        kwargs["y_limits"] = [-1, 1]
        # kwargs["x_limits"] = [0.5, 0.6]
        super(TopcRMCIntervalPostAnalysis, self).plot(*args, **kwargs)

    @staticmethod
    def Q4C(q4, q2):
        """4th cumulant."""
        return q4 - 3 * q2**2

    @staticmethod
    def Q4C_error(q4, q4err, q2, q2err):
        """4th cumulant error."""
        return np.sqrt(q4err**2 + (6*q2err*q2)**2 - 12*q2*q4err*q2err)

    @staticmethod
    def R(q4c, q2):
        """Returns the ratio <Q^4>C/<Q^2>"""
        return q4c/q2

    @staticmethod
    def R_error(q4c, q4cerr, q2, q2err):
        """Returns the ratio <Q^4>C/<Q^2>"""
        return np.sqrt(
            (q4cerr/q2)**2 + (q4c*q2err / q2**2)**2 - 2*q4cerr*q4c*q2err/q2**3)

    @staticmethod
    def ratio_error(x, xerr, y, yerr):
        """Returns the ratio and and error between two quantities."""
        return x/y, np.sqrt((xerr/y)**2 + (x*yerr/y**2)**2)

    def _print_data(self, int_keys, atype="bootstrap"):
        """Prints data."""

        article_param_header = [r"Lattice", r"$\beta$", r"$L/a$", r"$L[\fm]$",
                                r"$a[\fm]$", r"$t_0/{a^2}$", r"$t_0/{r_0^2}$",
                                ]

        art_flat = self.article_flattened
        article_param_table = [
            art_flat.keys(),
            [art_flat[k]["beta"] for k in art_flat],
            [art_flat[k]["L"] for k in art_flat],
            [art_flat[k]["aL"] for k in art_flat],
            [art_flat[k]["a"] for k in art_flat],
            [sciprint(art_flat[k]["t0"], art_flat[k]["t0err"], prec=4)
                for k in art_flat],
            [sciprint(art_flat[k]["t0r02"], art_flat[k]["t0r02err"], prec=4,
                      force_prec=True) for k in art_flat],
        ]

        art_param_table_printer = TablePrinter(article_param_header,
                                               article_param_table)
        art_param_table_printer.print_table(width=15,
                                            row_seperator_positions=[5, 7, 9])

        article_values_header = [r"Lattice", r"$\langle Q^2 \rangle$",
                                 r"$\langle Q^4 \rangle$", r"$\langle Q^4 \rangle_C$", r"$R$"]
        article_values_table = [
            art_flat.keys(),
            [sciprint(art_flat[k]["Q2"], art_flat[k]["Q2Err"], prec=3)
                for k in art_flat],
            [sciprint(art_flat[k]["Q4"], art_flat[k]["Q4Err"], prec=2)
                for k in art_flat],
            [sciprint(art_flat[k]["Q4C"], art_flat[k]["Q4CErr"], prec=3)
                for k in art_flat],
            [sciprint(art_flat[k]["R"], art_flat[k]["RErr"], prec=3)
                for k in art_flat],
        ]

        art_values_table_printer = TablePrinter(article_values_header,
                                                article_values_table)
        art_values_table_printer.print_table(width=15,
                                             row_seperator_positions=[5, 7, 9])

        article_normed_header = [r"Lattice",
                                 r"$\langle Q^2 \rangle_\text{normed}$",
                                 r"$\langle Q^4 \rangle_\text{normed}$",
                                 r"$\langle Q^4 \rangle_{C,\text{normed}}$", r"$R_\text{normed}$"]
        article_normed_table = [
            art_flat.keys(),
            [sciprint(art_flat[k]["Q2_norm"], art_flat[k]["Q2Err_norm"], prec=3)
                for k in art_flat],
            [sciprint(art_flat[k]["Q4_norm"], art_flat[k]["Q4Err_norm"],
                      prec=3) for k in art_flat],
            [sciprint(art_flat[k]["Q4C_norm"], art_flat[k]["Q4CErr_norm"],
                      prec=3) for k in art_flat],
            [sciprint(art_flat[k]["R_norm"], art_flat[k]["RErr_norm"], prec=3)
                for k in art_flat],
        ]

        art_normed_table_printer = TablePrinter(article_normed_header,
                                                article_normed_table)
        art_normed_table_printer.print_table(width=15,
                                             row_seperator_positions=[5, 7, 9])

        beta_int_key_list = zip(self.batch_names_sorted, int_keys)

        values_header = [r"$\beta$", r"$L/a$", r"$t_0/a^2$",
                         r"$\langle Q^2 \rangle$", r"$\langle Q^4 \rangle$",
                         r"$\langle Q^4 \rangle_C$", r"$R$"]
        values_table = [
            [self.beta_values[bn] for bn in self.batch_names],
            ["{:.2f}".format(self.data_values[bn][k]["aL"])
                for bn, k in beta_int_key_list],
            [sciprint(self.data_values[bn][k]["Q2"],
                      self.data_values[bn][k]["Q2Err"])
                for bn, k in beta_int_key_list],
            [sciprint(self.t0[bn]["t0"], self.t0[bn]["t0err"])
                for bn, k in beta_int_key_list],
            [sciprint(self.data_values[bn][k]["Q4"],
                      self.data_values[bn][k]["Q4Err"])
                for bn, k in beta_int_key_list],
            [sciprint(self.data_values[bn][k]["Q4C"],
                      self.data_values[bn][k]["Q4CErr"])
                for bn, k in beta_int_key_list],
            [sciprint(self.data_values[bn][k]["R"],
                      self.data_values[bn][k]["RErr"])
                for bn, k in beta_int_key_list],
        ]

        values_table_printer = TablePrinter(values_header, values_table)
        values_table_printer.print_table(width=15)

        ratio_header = [r"Lattice", r"$\beta$",
                        r"$\text{Ratio}(\langle Q^2 \rangle)$",
                        r"$\text{Ratio}(\langle Q^4 \rangle)$",
                        r"$\text{Ratio}(\langle Q^4 \rangle_C)$", r"$\text{Ratio}(R)$"]
        ratio_table = []
        for fk in self.article_flattened:
            for bn, k in beta_int_key_list:
                sub_list = []

                sub_list.append(fk)
                sub_list.append(self.beta_values[bn])
                sub_list.append(sciprint(
                    self.data_ratios[fk][bn][k]["Q2"],
                    self.data_ratios[fk][bn][k]["Q2Err"]))
                sub_list.append(sciprint(
                    self.data_ratios[fk][bn][k]["Q4"],
                    self.data_ratios[fk][bn][k]["Q4Err"]))
                sub_list.append(sciprint(
                    self.data_ratios[fk][bn][k]["Q4C"],
                    self.data_ratios[fk][bn][k]["Q4CErr"]))
                sub_list.append(sciprint(
                    self.data_ratios[fk][bn][k]["R"],
                    self.data_ratios[fk][bn][k]["RErr"]))

                ratio_table.append(sub_list)

        ratio_table = np.asarray(ratio_table).T.tolist()
        ratio_tab_pos = (4*np.arange(len(self.article_flattened))) - 1
        ratio_table_printer = TablePrinter(ratio_header, ratio_table)
        ratio_table_printer.print_table(width=15,
                                        row_seperator_positions=ratio_tab_pos)

        print "Reference scale t0(my data): %s" % self.t0

    def _setup_comparison_values(self, int_keys, tf="t0beta_a2",
                                 atype="bootstrap"):
        """
        Sets up a new dictionary with my own values for comparing with the 
        article values.
        """
        self.data_values = {bn: {key: {} for key in int_keys}
                            for bn in self.batch_names_sorted}

        # Sets up the reference values
        try:
            ref_vals = self.reference_values[atype]["bootstrap"]
        except TypeError as type_error:
            print "Have missing 'reference_values'. Have 'energy' been run?"
            raise type_error
        except KeyError:
            fallback_key = self.reference_values[atype].keys()[0]
            print("Bootstrap line extrapolation not found."
                  " Falling back to %s" % fallback_key)
            ref_vals = self.reference_values[atype][fallback_key]
        self.t0 = {bn: {
            "t0": ref_vals[bn]["t0a2"],
            "t0err": ref_vals[bn]["t0a2err"]
        } for bn in self.batch_names_sorted}

        t0_indexes = [
            np.argmin(np.abs(
                self.topc2[atype][bn][int_keys[i]]["x"] - self.t0[bn]["t0"]))
            for i, bn in enumerate(self.sorted_batch_names)]

        t0_b_int_loop_list = zip(t0_indexes, self.batch_names_sorted, int_keys)
        for t0_index, bn, int_key in t0_b_int_loop_list:
            self.data_values[bn][int_key]["aL"] = ref_vals[b]["aL"]

            self.data_values[bn][int_key]["Q2"] = \
                self.topc2[atype][bn][int_key]["y"][t0_index]
            self.data_values[bn][int_key]["Q2Err"] = \
                self.topc2[atype][bn][int_key]["y_error"][t0_index]

            self.data_values[bn][int_key]["Q4"] = \
                self.topc4[atype][bn][int_key]["y"][t0_index]
            self.data_values[bn][int_key]["Q4Err"] = \
                self.topc4[atype][bn][int_key]["y_error"][t0_index]

            self.data_values[bn][int_key]["Q4C"] = \
                self.topc4C[atype][bn][int_key]["y"][t0_index]
            self.data_values[bn][int_key]["Q4CErr"] = \
                self.topc4C[atype][bn][int_key]["y_error"][t0_index]

            self.data_values[bn][int_key]["R"] = \
                self.topcR[atype][bn][int_key]["y"][t0_index]
            self.data_values[bn][int_key]["RErr"] = \
                self.topcR[atype][bn][int_key]["y_error"][t0_index]

        self.data_ratios = {k: {bn: {key: {} for key in int_keys}
                                for bn in self.batch_names_sorted}
                            for k in self.article_flattened.keys()}

        for flat_key in self.article_flattened:
            beta_article = self.article_flattened[flat_key]["beta"]
            for t0_index, bn, int_key in t0_b_int_loop_list:
                [self.data_ratios[flat_key][bn][int_key]["Q2"],
                    self.data_ratios[flat_key][bn][int_key]["Q2Err"]] = \
                    self.ratio_error(self.data_values[bn][int_key]["Q2"],
                                     self.data_values[bn][int_key]["Q2Err"],
                                     self.article_flattened[flat_key]["Q2_norm"],
                                     self.article_flattened[flat_key]["Q2Err_norm"])

                [self.data_ratios[flat_key][bn][int_key]["Q4"],
                    self.data_ratios[flat_key][bn][int_key]["Q4Err"]] = \
                    self.ratio_error(self.data_values[bn][int_key]["Q4"],
                                     self.data_values[bn][int_key]["Q4Err"],
                                     self.article_flattened[flat_key]["Q4_norm"],
                                     self.article_flattened[flat_key]["Q4Err_norm"])

                [self.data_ratios[flat_key][bn][int_key]["Q4C"],
                    self.data_ratios[flat_key][bn][int_key]["Q4CErr"]] = \
                    self.ratio_error(self.data_values[bn][int_key]["Q4C"],
                                     self.data_values[bn][int_key]["Q4CErr"],
                                     self.article_flattened[flat_key]["Q4C_norm"],
                                     self.article_flattened[flat_key]["Q4CErr_norm"])

                [self.data_ratios[flat_key][bn][int_key]["R"],
                    self.data_ratios[flat_key][bn][int_key]["RErr"]] = \
                    self.ratio_error(self.data_values[bn][int_key]["R"],
                                     self.data_values[bn][int_key]["RErr"],
                                     self.article_flattened[flat_key]["R_norm"],
                                     self.article_flattened[flat_key]["RErr_norm"])

        # arr = np.asarray(self.data_ratios)
        # print arr
        # print arr.shape
        # print arr.reshape(arr.shape[0]*arr.shape[1])

    def compare_lattice_values(self, interval, tf=None, atype="bootstrap"):
        """
        Compares values at flow times given by the data we are comparing against
        """
        if len(list(set(self.beta_values.values()))) != len(self.batch_names):
            print("Multiple values for a beta value: {} --> Skipping"
                  " continuum extrapolation".format(self.beta_values.values()))
            return

        x_pvals_article = []
        y_pvals_article = []

        x_pvals_me = []
        y_pvals_me = []

        def ratio_error(x, xerr, y, yerr):
            return x/y, np.sqrt((xerr/y)**2 + (x*yerr/y**2)**2)

        self._setup_comparison_values(interval, tf=tf, atype=atype)
        self._print_data(interval, atype=atype)

        # for size in sorted(self.article_name_size[data_set]):
        # for size in sorted(self.article_size_name):

        #   # Sets the t0 value to extract at.
        #   if tf == "article":
        #       self.t0 = {b: self.article_name_size["B"][size]["t0cont"]
        #           for bn in self.batch_names_sorted}
        #   else:
        #       tf = "t0beta_a2"
        #       self._get_tf_value(tf, atype, None)

        #   print "="*150
        #   print "Reference value type %s t0: %s" % (tf, self.t0)

        #   print "\nMy data:"
        #   for bn in self.batch_names_sorted:
        #       # Gets the approximate same t0 ref. value
        #       t0_index = np.argmin(np.abs(self.topc2[atype][beta]["x"] - self.t0[beta]))

        #       print ("Beta: %4.2f t0: %4.2f Q2: %10.5f Q2_err: %10.5f Q4: %10.5f \
        #           Q4_err: %10.5f Q4C: %10.5f Q4C_err: %10.5f R: %10.5f \
        #           R_err: %10.5f" % (beta, self.t0[beta],
        #           self.topc2[atype][beta]["y"][t0_index],
        #           self.topc2[atype][beta]["y_error"][t0_index],
        #           self.topc4[atype][beta]["y"][t0_index],
        #           self.topc4[atype][beta]["y_error"][t0_index],
        #           self.topc4C[atype][beta]["y"][t0_index],
        #           self.topc4C[atype][beta]["y_error"][t0_index],
        #           self.topcR[atype][beta]["y"][t0_index],
        #           self.topcR[atype][beta]["y_error"][t0_index]))

        #   print "\nArticle data(normalized by volume):"
        #   for data_set in sorted(article_data2[size]):

        #       print "Dataset: %s Beta: %2.2f Volume: %f t0: %s" % (
        #           data_set, self.article_name_size[data_set][size]["beta"],
        #           self.article_name_size[data_set][size]["V"],    self.t0)
        #       print "Q2:  %10.5f Q2_err:  %10.5f" % (
        #           self.article_name_size[data_set][size]["Q2_norm"],
        #           self.article_name_size[data_set][size]["Q2Err_norm"])
        #       print "Q4:  %10.5f Q4_err:  %10.5f" % (
        #           self.article_name_size[data_set][size]["Q4_norm"],
        #           self.article_name_size[data_set][size]["Q4Err_norm"])
        #       print "Q4C: %10.5f Q4C_err: %10.5f" % (
        #           self.article_name_size[data_set][size]["Q4C_norm"],
        #           self.article_name_size[data_set][size]["Q4CErr_norm"])
        #       print "R:   %10.5f R_err:   %10.5f" % (
        #           self.article_name_size[data_set][size]["R_norm"],
        #           self.article_name_size[data_set][size]["RErr_norm"])

        #       if size==1:
        #           x_pvals_article.append(self.t0)
        #           y_pvals_article.append((
        #               self.article_name_size[data_set][size]["R_norm"],
        #               self.article_name_size[data_set][size]["RErr_norm"]))

        #   print "\nRatios between me and article"
        #   for data_set in sorted(self.article_size_name[size]):
        #       # Compares values by dividing my values by article values
        #       for bn in self.batch_names_sorted:
        #           beta_article = self.article_name_size[data_set][size]["beta"]

        #           # Gets the approximate same t0 ref. value
        #           t0_index = np.argmin(
        #               np.abs(self.topc2[atype][beta]["x"] - self.t0[beta]["t0"]))
        #           print "Beta(me) %.2f Beta(article) %.2f Dataset %-s" % (
        #               beta, beta_article, data_set)

        #           print "Q2_me/Q2_article:   %10.5f +/- %10.5f" % (
        #               ratio_error(self.topc2[atype][beta]["y"][t0_index],
        #               self.topc2[atype][beta]["y_error"][t0_index],
        #               self.article_name_size[data_set][size]["Q2_norm"],
        #               self.article_name_size[data_set][size]["Q2Err_norm"]))
        #           print "Q4_me/Q4_article:   %10.5f +/- %10.5f" % (
        #               ratio_error(self.topc4[atype][beta]["y"][t0_index],
        #               self.topc4[atype][beta]["y_error"][t0_index],
        #               self.article_name_size[data_set][size]["Q4_norm"],
        #               self.article_name_size[data_set][size]["Q4Err_norm"]))
        #           print "Q4C_me/Q4C_article: %10.5f +/- %10.5f" % (
        #               ratio_error(self.topc4C[atype][beta]["y"][t0_index],
        #               self.topc4C[atype][beta]["y_error"][t0_index],
        #               self.article_name_size[data_set][size]["Q4C_norm"],
        #               self.article_name_size[data_set][size]["Q4CErr_norm"]))
        #           print "R_me/R_article:     %10.5f +/- %10.5f" % (
        #               ratio_error(self.topcR[atype][beta]["y"][t0_index],
        #               self.topcR[atype][beta]["y_error"][t0_index],
        #               self.article_name_size[data_set][size]["R_norm"],
        #               self.article_name_size[data_set][size]["RErr_norm"]))
        #   print ""

        # exit("Exits before exiting compare_lattice_values @ 565")

    def _normalize_article_values(self):
        """
        Normalizes values from article based on physical volume.
        """
        for data_set in sorted(self.article_name_size):
            for size in sorted(self.article_name_size[data_set]):
                # Set up volume in physical units
                L = self.article_name_size[data_set][size]["L"]
                a = self.article_name_size[data_set][size]["a"]
                aL = a*float(L)
                V = (aL)**4
                # self._add_article_dict_item(name, size, key, value)
                self._add_article_dict_item(data_set, size, "aL", aL)
                self._add_article_dict_item(data_set, size, "V", V)

                # Normalize Q^2 by V
                Q2 = self.article_name_size[data_set][size]["Q2"]
                Q2Err = self.article_name_size[data_set][size]["Q2Err"]
                Q2_norm = Q2/V
                Q2Err_norm = Q2Err/V
                self._add_article_dict_item(data_set, size, "Q2_norm", Q2_norm)
                self._add_article_dict_item(
                    data_set, size, "Q2Err_norm", Q2Err_norm)

                # Normalize Q^4 by V
                Q4 = self.article_name_size[data_set][size]["Q4"]
                Q4Err = self.article_name_size[data_set][size]["Q4Err"]
                Q4_norm = Q4/V**2
                Q4Err_norm = Q4Err/V**2
                self._add_article_dict_item(data_set, size, "Q4_norm", Q4_norm)
                self._add_article_dict_item(
                    data_set, size, "Q4Err_norm", Q4Err_norm)

                # Recalculates 4th cumulant
                Q4C_norm = self.Q4C(Q4_norm, Q2_norm)
                Q4CErr_norm = self.Q4C_error(Q4_norm, Q4Err_norm, Q2_norm,
                                             Q2Err_norm)
                self._add_article_dict_item(
                    data_set, size, "Q4C_norm", Q4C_norm)
                self._add_article_dict_item(
                    data_set, size, "Q4CErr_norm", Q4CErr_norm)

                # Recalculates R
                R_norm = self.R(Q4C_norm, Q2_norm)
                RErr_norm = self.R_error(Q4C_norm, Q4CErr_norm, Q2_norm,
                                         Q2Err_norm)
                self._add_article_dict_item(data_set, size, "R_norm", R_norm)
                self._add_article_dict_item(
                    data_set, size, "RErr_norm", RErr_norm)

        # for data_set in sorted(self.article_name_size):
        #   for size in sorted(self.article_name_size[data_set]):
        #       print "="*50
        #       print "Dataset: %s Size number: %s Volume: %f" % (
        #           data_set, size, self.article_name_size[data_set][size]["V"])
        #       print "Q2: %10.5f %10.5f" % (
        #           self.article_name_size[data_set][size]["Q2_norm"],
        #           self.article_name_size[data_set][size]["Q2Err_norm"])
        #       print "Q4: %10.5f %10.5f" % (
        #           self.article_name_size[data_set][size]["Q4_norm"],
        #           self.article_name_size[data_set][size]["Q4Err_norm"])
        #       print "Q4C: %10.5f %10.5f" % (
        #           self.article_name_size[data_set][size]["Q4C_norm"],
        #           self.article_name_size[data_set][size]["Q4CErr_norm"])
        #       print "R: %10.5f %10.5f" % (
        #           self.article_name_size[data_set][size]["R_norm"],
        #           self.article_name_size[data_set][size]["RErr_norm"])

    def _add_article_dict_item(self, name, size, key, value):
        """Small method for adding item to article data dictionaries."""
        self.article_size_name[size][name][key] = value
        self.article_name_size[name][size][key] = value
        flat_key = "{0:s}_{1:d}".format(name, size)
        self.article_flattened[flat_key][key] = value

    def _setup_article_values(self):
        """
        Sets up the article values from https://arxiv.org/abs/1506.06052

        Format:
            {Lattice type}/{Beta value}/{all other stuff}
        """

        self.article_name_size = {
            "A":
            {
                1: {
                    "beta": 5.96,
                    "t0cont": 2.79,  # t0/a^2
                    "t0": 2.995,  # t0/a^2
                    "t0err": 0.004,  # t0/a^2
                    "t0r02": 0.1195,
                    "t0r02err": 0.0009,
                    "L": 10,
                    # "aL": 1.0, # [fm]
                    "a": 0.102,  # [fm]
                    "Q2": 0.701,
                    "Q2Err": 0.006,
                    "Q4": 1.75,
                    "Q4Err": 0.04,
                    "Q4C": 0.273,
                    "Q4CErr": 0.020,
                    "R": 0.39,
                    "RErr": 0.03,
                },
            },

            "B":
            {
                1: {
                    "beta": 5.96,
                    "t0cont": 2.79,  # t0/a^2
                    "t0": 2.7984,
                    "t0err": 0.0009,
                    "t0r02": 0.1117,
                    "t0r02err": 0.0009,
                    "L": 12,
                    # "aL": 1.2, # [fm]
                    "a": 0.102,  # [fm]
                    "Q2": 1.617,
                    "Q2Err": 0.006,
                    "Q4": 8.15,
                    "Q4Err": 0.07,
                    "Q4C": 0.30,
                    "Q4CErr": 0.04,
                    "R": 0.187,
                    "RErr": 0.024,
                },
                2: {
                    "beta": 6.05,
                    "t0cont": 3.78,  # t0/a^2
                    "t0": 3.7960,
                    "t0err": 0.0012,
                    "t0r02": 0.1114,
                    "t0r02err": 0.0009,
                    "L": 14,
                    # "aL": 1.2, # [fm]
                    "a": 0.087,  # [fm]
                    "Q2": 1.699,
                    "Q2Err": 0.007,
                    "Q4": 9.07,
                    "Q4Err": 0.09,
                    "Q4C": 0.41,
                    "Q4CErr": 0.05,
                    "R": 0.24,
                    "RErr": 0.03,
                },
                3: {
                    "beta": 6.13,
                    "t0cont": 4.87,  # t0/a^2
                    "t0": 4.8855,
                    "t0err": 0.0015,
                    "t0r02": 0.1113,
                    "t0r02err": 0.0010,
                    "L": 16,
                    # "aL": 1.2, # [fm]
                    "a": 0.077,  # [fm]
                    "Q2": 1.750,
                    "Q2Err": 0.007,
                    "Q4": 9.58,
                    "Q4Err": 0.09,
                    "Q4C": 0.39,
                    "Q4CErr": 0.05,
                    "R": 0.22,
                    "RErr": 0.03,
                },
                4: {
                    "beta": 6.21,
                    "t0cont": 6.20,  # t0/a^2
                    "t0": 6.2191,
                    "t0err": 0.0020,
                    "t0r02": 0.1115,
                    "t0r02err": 0.0011,
                    "L": 18,
                    # "aL": 1.2, # [fm]
                    "a": 0.068,  # [fm]
                    "Q2": 1.741,
                    "Q2Err": 0.007,
                    "Q4": 9.44,
                    "Q4Err": 0.09,
                    "Q4C": 0.35,
                    "Q4CErr": 0.05,
                    "R": 0.20,
                    "RErr": 0.03,
                },
            },

            "C":
            {
                1: {
                    "beta": 5.96,
                    "t0cont": 2.79,  # t0/a^2
                    "t0": 2.7908,
                    "t0err": 0.0005,
                    "t0r02": 0.1114,
                    "t0r02err": 0.0009,
                    "L": 13,
                    # "aL": 1.3, # [fm]
                    "a": 0.102,  # [fm]
                    "Q2": 2.244,
                    "Q2Err": 0.006,
                    "Q4": 15.50,
                    "Q4Err": 0.10,
                    "Q4C": 0.40,
                    "Q4CErr": 0.05,
                    "R": 0.177,
                    "RErr": 0.023,
                },
            },

            "D": {
                1: {
                    "beta": 5.96,
                    "t0cont": 2.79,  # t0/a^2
                    "t0": 2.7889,
                    "t0err": 0.0003,
                    "t0r02": 0.1113,
                    "t0r02err": 0.0009,
                    "L": 14,
                    # "aL": 1.4, # [fm]
                    "a": 0.102,  # [fm]
                    "Q2": 3.028,
                    "Q2Err": 0.006,
                    "Q4": 28.14,
                    "Q4Err": 0.14,
                    "Q4C": 0.63,
                    "Q4CErr": 0.07,
                    "R": 0.209,
                    "RErr": 0.023,
                },
                2: {
                    "beta": 6.05,
                    "t0cont": 3.78,  # t0/a^2
                    "t0": 3.7825,
                    "t0err": 0.0008,
                    "t0r02": 0.1110,
                    "t0r02err": 0.0009,
                    "L": 17,
                    # "aL": 1.5, # [fm]
                    "a": 0.087,  # [fm]
                    "Q2": 3.686,
                    "Q2Err": 0.014,
                    "Q4": 41.6,
                    "Q4Err": 0.4,
                    "Q4C": 0.83,
                    "Q4CErr": 0.19,
                    "R": 0.22,
                    "RErr": 0.05,
                },
                3: {
                    "beta": 6.13,
                    "t0cont": 4.87,  # t0/a^2
                    "t0": 4.8722,
                    "t0err": 0.0011,
                    "t0r02": 0.1110,
                    "t0r02err": 0.0010,
                    "L": 19,
                    # "aL": 1.5, # [fm]
                    "a": 0.077,  # [fm]
                    "Q2": 3.523,
                    "Q2Err": 0.013,
                    "Q4": 37.8,
                    "Q4Err": 0.3,
                    "Q4C": 0.56,
                    "Q4CErr": 0.17,
                    "R": 0.16,
                    "RErr": 0.05,
                },
                4: {
                    "beta": 6.21,
                    "t0cont": 6.20,  # t0/a^2
                    "t0": 6.1957,
                    "t0err": 0.0014,
                    "t0r02": 0.1111,
                    "t0r02err": 0.0011,
                    "L": 21,
                    # "aL": 1.4, # [fm]
                    "a": 0.068,  # [fm]
                    "Q2": 3.266,
                    "Q2Err": 0.012,
                    "Q4": 32.7,
                    "Q4Err": 0.3,
                    "Q4C": 0.68,
                    "Q4CErr": 0.15,
                    "R": 0.21,
                    "RErr": 0.05,
                },
            },

            "E": {
                1: {
                    "beta": 5.96,
                    "t0cont": 2.79,  # t0/a^2
                    "t0": 2.78892,
                    "t0err": 0.00023,
                    "t0r02": 0.1113,
                    "t0r02err": 0.0009,
                    "L": 15,
                    # "aL": 1.5, # [fm]
                    "a": 0.102,  # [fm]
                    "Q2": 3.982,
                    "Q2Err": 0.006,
                    "Q4": 48.38,
                    "Q4Err": 0.18,
                    "Q4C": 0.81,
                    "Q4CErr": 0.09,
                    "R": 0.202,
                    "RErr": 0.023,
                },
            },

            "F": {
                1: {
                    "beta": 5.96,
                    "t0cont": 2.79,  # t0/a^2
                    "t0": 2.78867,
                    "t0err": 0.00016,
                    "t0r02": 0.1113,
                    "t0r02err": 0.0009,
                    "L": 16,
                    # "aL": 1.6, # [fm]
                    "a": 0.102,  # [fm]
                    "Q2": 5.167,
                    "Q2Err": 0.006,
                    "Q4": 80.90,
                    "Q4Err": 0.22,
                    "Q4C": 0.81,
                    "Q4CErr": 0.11,
                    "R": 0.157,
                    "RErr": 0.022,
                },
            },
        }

        self.article_size_name = {}
        for data_set in sorted(self.article_name_size):
            for size in sorted(self.article_name_size[data_set]):
                self.article_size_name[size] = {}

        for data_set in sorted(self.article_name_size):
            for size in sorted(self.article_name_size[data_set]):
                self.article_size_name[size][data_set] = \
                    self.article_name_size[data_set][size]

        # Usefull when printing out values. The two first layers of
        # dictionaries are merged.
        self.article_flattened = {}
        for data_set in sorted(self.article_name_size):
            for size in sorted(self.article_name_size[data_set]):
                name = "{0:s}_{1:d}".format(data_set, size)
                self.article_flattened[name] = \
                    self.article_name_size[data_set][size]

        self.article_flattened = OrderedDict(sorted(
            self.article_flattened.items(),
            key=lambda k: [k[0].split("_")[1], k[0].split("_")[0]]))


def main():
    exit("Exit: TopcRPostAnalysis not intended to be a standalone module.")


if __name__ == '__main__':
    main()
