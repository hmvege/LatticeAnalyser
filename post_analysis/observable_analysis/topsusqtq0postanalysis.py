from post_analysis.core.multiplotcore import MultiPlotCore
from post_analysis.core.topsuscore import TopsusCore
from tools.folderreadingtools import check_folder
import os


class TopsusQtQ0PostAnalysis(MultiPlotCore, TopsusCore):
    """Post-analysis of the topsus at a fixed flow time."""
    observable_name = \
        r"$\chi_{t_f}(\langle Q_{t_f} Q_{t_{f,0}} \rangle)^{1/4}$"
    observable_name_compact = "topsusqtq0"
    obs_name_latex = r"\chi_{t_f}^{1/4}\expect{Q_{t_f}Q_{t_{f,0}}}"
    x_label = r"$\sqrt{8t_{f}}$ [fm]"
    y_label = \
        r"$\chi_{t_f}(\langle Q_{t_f} Q_{t_{f,0}} \rangle)^{1/4}$ [GeV]"
    sub_obs = True
    descr = "One Q at fixed flow time"
    subfolder_type = "tf"

    # Continuum plot variables
    y_label_continuum = r"$\chi^{1/4}(\langle Q_{t} Q_{t_0} \rangle)[GeV]$"

    def _convert_label(self, label):
        """Short method for formatting time in labels."""
        try:
            return r"$\sqrt{8t_{f,0}}=%.2f$" % (float(label))
        except ValueError:
            return r"$%s$" % label

    def plot_series(self, indexes):
        """
        Method for plotting 4 axes together.

        Args:
                indexes: list containing integers of which intervals to 
                        plot together.
        """

        self.plot_values = {}
        self._initiate_plot_values(self.data[self.analysis_data_type],
                                   self.data_raw[self.analysis_data_type])

        for ib, bn in enumerate(self.sorted_batch_names):
            for sub_obs in self.observable_intervals[bn]:
                self.plot_values[bn][sub_obs]["label"] = \
                    self.ensemble_names[bn]

        _tmp_sub_values = sorted(self.observable_intervals.values()[0])
        sub_titles = [self._convert_label(_tsv) for _tsv in _tmp_sub_values]

        self._series_plot_core(indexes, legend_loc="upper right",
                               use_common_legend=True,
                               common_legend_anchor=(0.5, 0),
                               sub_adjust_bottom=0.15,
                               x_label_bottom_pos=(0.51, 0.09),
                               sub_titles=sub_titles)


def main():
    exit("Exit: TopsusQtQ0PostAnalysis "
         "not intended to be a standalone module.")


if __name__ == '__main__':
    main()
