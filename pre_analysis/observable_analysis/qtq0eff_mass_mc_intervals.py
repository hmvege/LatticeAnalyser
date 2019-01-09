from pre_analysis.observable_analysis import QtQ0EffectiveMassAnalyser
import copy
import numpy as np
import os
from tools.folderreadingtools import check_folder
import statistics.parallel_tools as ptools


class QtQ0EffectiveMassMCAnalyser(QtQ0EffectiveMassAnalyser):
    """Correlator of <QtQ0> in euclidean time analysis class."""
    observable_name = r""
    observable_name_compact = "qtq0effmc"
    x_label = r"$t_e[fm]$"
    y_label = r"$am_\textrm{eff} = \ln \frac{\langle Q_{t_e} Q_0 \rangle}{\langle Q_{t_e+1} Q_0 \rangle}$"
    mark_interval = 1
    error_mark_interval = 1

    def __str__(self):
        def info_string(s1, s2): return "\n{0:<20s}: {1:<20s}".format(s1, s2)
        return_string = ""
        return_string += "\n" + self.section_seperator
        return_string += info_string("Data batch folder",
                                     self.batch_data_folder)
        return_string += info_string("Batch name", self.batch_name)
        return_string += info_string("Observable",
                                     self.observable_name_compact)
        return_string += info_string("Beta", "%.2f" % self.beta)
        return_string += info_string("Flow time t0",
                                     "%.2f" % self.q0_flow_time)
        return_string += info_string("MC-interval: ",
                                     "[%d,%d)" % self.mc_interval)
        return_string += "\n" + self.section_seperator
        return return_string


def main():
    exit("Module QtQ0EffectiveMassAnalyser not intended for standalone usage.")


if __name__ == '__main__':
    main()
