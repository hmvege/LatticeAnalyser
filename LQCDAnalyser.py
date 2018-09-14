#!/usr/bin/env python2

from pre_analysis.pre_analyser import pre_analysis
from post_analysis.post_analyser import post_analysis
from tools.folderreadingtools import get_num_observables
import copy
import os

from analysis_batches.main_analysis import main_analysis
from analysis_batches.distribution_analysis import distribution_analysis
from analysis_batches.topc_modes_analysis import topc_modes_analysis
from analysis_batches.thermalization_comparison_analysis \
    import thermalization_analysis


def main():
    # Printing settings
    section_seperator = "="*160

    main_analysis()
    # distribution_analysis() # TODO: add different epsilon distributions to the same plotting window - possible in some sort of post analysis?
    # topc_modes_analysis() # TODO: add different modes to the same plotting window
    # thermalization_analysis() # TODO: complete thermalization analysis
    # strong_scaling_analysis() # TODO: complete strong scaling analysis
    # weak_scaling_analysis() # TODO: complete weak scaling analysis
    # lattice_updates_analysis() # TODO: complete lattice updates analysis


if __name__ == '__main__':
    main()
