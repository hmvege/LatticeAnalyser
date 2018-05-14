from post_analysis.core.flowgif import FlowGif
from post_analysis.observable_analysis.qtq0euclideanpostanalysis \
    import QtQ0EuclideanPostAnalysis

class QtQ0EPostGif(QtQ0EuclideanPostAnalysis, FlowGif):
    """Class for plotting different QteQte0 a specific flow time together."""
    observable_name_compact = "qtq0e_gif"
    y_limits = [-0.08, 0.6]
