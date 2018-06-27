class TablePrinter:
    """
    Class for generating a table from a header and table body. Prints in 
    either regular ascii text or in latex format.
    """

    def __init__(self, header, table):
        """
        Initializer for TablePrinter. Sets up a NxM table.

        Args:
            header: list of N str elements, one for each row in table.
            table: list of N lists of M column elements.

        Raises:
            AssertionError: if length of header differs from length of table.
        """

        assert len(header) == len(table), (
            "Header length is not equal number of table columns")

        self.header = header
        self.table = self._transpose_table(table)

    def _transpose_table(self, tab):
        """Transposes table list, tab."""
        self.N_cols = len(tab)
        self.N_rows = len(tab[0])
        assert len(set([len(row) for row in tab])) == 1, (
            "Length of all the table rows should be equal")

        new_tab = [[] for i in range(self.N_rows)]

        # Transforms table from [column][row] to [row][column]
        for ic, col in enumerate(tab):
            for ir, row in enumerate(col):
                new_tab[ir].append(row)

        return new_tab

    def _check_table_elem(self, elem, latex):
        """
        Checks and converts table element to string, and raises and error if 
        it is of an unknown type.
        """

        if isinstance(elem, (float, int)):
            elem = str(elem)
        elif isinstance(elem, str):
            pass
        else:
            raise TypeError(("Table element type "
                "%s not recognized in element: " % type(elem) + elem))

        if latex and len(elem)>0:
            return "${0:<s}$".format(elem)
        else:
            return elem

    def _generate_table(self, latex=True, width=10, row_seperator=r"\hline", 
        row_seperator_positions=[], ignore_latex_cols=[]):
        """
        Internal table generator.

        Args:
            latex: bool, optional, if true, will print in LaTeX format. Default
                is True.
            width: int or table of ints, optional. Width of spacing allocated 
                to table. Default is 10.
            row_seperator: str, optional. One will always be placed after 
                table header unless empty. Element to print between header and 
                table. Default is hline.
            row_seperator_positions: list of ints, optional. Will place a 
                row_seperator after each position in list.
            ignore_latex_cols: list of bools, optional, rows specified will 
                not be in LaTeX format if prompted.
        """

        if isinstance(width, list):
            assert len(width) == self.N_cols, ("number of spacings for "
                "columns in 'width' is not equal to the number of columns")
        elif isinstance(width, int):
            width = [width for icol in range(self.N_cols)]
        else:
            raise ValueError("%s is not allowed for 'width'. Type should be "
                "either 'int' or 'list'" % type(width))

        tab = "\n"

        # Printing header
        for icol, h in enumerate(self.header):
            tab += "{0:<{w}s}".format(h, w=width[icol])
            if latex and icol != (self.N_cols-1):
                tab += " & "

        if latex:
            tab += r" \\ "

        tab += "{0:<s}".format(row_seperator)

        # Printing table contents
        for ir, row in enumerate(self.table):

            tab += "\n"

            for icol, elem in enumerate(row):
                if icol in ignore_latex_cols:
                    elem = self._check_table_elem(elem, False)
                else:
                    elem = self._check_table_elem(elem, latex)

                tab += "{0:<{w}s}".format(elem, w=width[icol])

                if latex and icol != (self.N_cols-1):
                    tab += " & "

            if latex:
                tab += r" \\ "
                if ir in row_seperator_positions:
                    tab += row_seperator

        tab += "\n"

        return tab

    def print_table(self, latex=True, width=10, row_seperator=r"\hline", 
        row_seperator_positions=[], ignore_latex_cols=[]):
        """
        Prints a table in either LaTeX format or in regular ascii format.

        Args:
            latex: bool, optional, if true, will print in LaTeX format. Default
                is True. All cells will be converted to LaTeX equations unless 
                specified in ignore_latex_cols.
            width: int, optional. Width of spacing allocated to table. 
                Default is 10.
            row_seperator: str, optional. One will always be placed after 
                table header unless empty. Element to print between header and 
                table. Default is hline.
            row_seperator_positions: list of ints, optional. Will place a 
                row_seperator after each position in list.
            ignore_latex_cols: list of bools, optional, rows specified will 
                not be in LaTeX format if prompted.
        """

        print self._generate_table(latex=latex, width=width, 
            row_seperator=row_seperator, 
            row_seperator_positions=row_seperator_positions, 
            ignore_latex_cols=ignore_latex_cols)

    def __call__(self, latex=True, width=10, row_seperator=r"\hline",
        row_seperator_positions=[]):
        return self._generate_table(latex=latex, width=width, 
            row_seperator=row_seperator, 
            row_seperator_positions=row_seperator_positions)

    def __str__(self):
        return self._generate_table(latex=False, width=15, row_seperator="")


def main():
    # Testing printing function
    header = ["a", "b", "c", "d"]
    table = [
        [0.1, 2.3, 3.4], # Columns
        [0.3, 4.1, 4.5],
        [1.1, 1.3, 6.3],
        [1.3, 2.0, 4.3]
    ]

    ptab = TablePrinter(header, table)
    print ptab
    print "\nLATEX PRINTOUT:"
    ptab.print_table()

if __name__ == '__main__':
    main()
