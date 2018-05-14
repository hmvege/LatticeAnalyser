from post_analysis.core.flowgif import FlowGif
from post_analysis.observable_analysis.qtq0effectivemasspostanalysis \
    import QtQ0EffectiveMassPostAnalysis

class QtQ0EffPostGif(QtQ0EffectiveMassPostAnalysis, FlowGif):
    """Post-analysis of the effective mass."""
    observable_name_compact = "qtq0eff_gif"
    x_limits = [-0.1,4.7]
    y_limits = [-1,1]

    def data_setup(self):
        """Sets up the data in a format FlowGif can utilze."""
        self.data_backup = self.data

        _data = {atype: {b: {} for b in self.beta_values} \
            for atype in self.analysis_types}

        flow_times = sorted(self.data[self.analysis_types[0]] \
            [self.beta_values[0]].keys())

        # self.data[atype][beta][flow_time][euclidean_time]
        for atype in self.analysis_types:
            for beta in self.beta_values:
                for ftime in flow_times:

                    _y, _y_error = self.analyse_raw(
                        self.data[atype][beta][ftime],
                        self.data_raw[atype][beta] \
                            [self.observable_name_compact][ftime])

                    _data[atype][beta][ftime] = {
                        "x": self.data[atype][beta][ftime]["x"],
                        "y": _y, "y_error": _y_error}

        self.data = _data