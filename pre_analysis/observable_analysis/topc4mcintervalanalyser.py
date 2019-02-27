from pre_analysis.core.flowanalyser import FlowAnalyser

class Topc4MCIntervalAnalyser(FlowAnalyser):
    """Class for topological charge with quartic topological charge."""
    observable_name = r"$\langle Q^4 \rangle$"
    observable_name_compact = "topc4MC"
    x_label = r"$\sqrt{8t_{f}}$ [fm]"
    y_label = r"$\langle Q^4 \rangle$"

    def __init__(self, *args, **kwargs):
        super(Topc4MCIntervalAnalyser, self).__init__(*args, **kwargs)
        self.y **= 4

def main():
    exit("Module Topc4MCIntervalAnalyser not intended for standalone usage.")

if __name__ == '__main__':
    main()