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
from analysis_batches.b645_L32x32_analysis import beta645_L32_analysis
from analysis_batches.scaling_analysis import scaling_analysis
from analysis_batches.ABC_analysis import ABC_analysis
from analysis_batches.BCD_analysis import BCD_analysis


def main():
    # Printing settings
    section_seperator = "="*160

    # Default params
    run_pre_analysis = True
    run_post_analysis = True
    only_generate_data = False
    observables = None
    post_analysis_data_type = ["bootstrap", "unanalyzed", "blocked_bootstrap",
                               "jackknife", "blocked", "bootstrap_time_series"]

    # Overriding params for what to run
    run_pre_analysis = False
    # run_post_analysis = False
    only_generate_data = True

    # Observables selection, full
    observables = ["plaq", "topc", "topc2", "topc4", "topcr", "topsus",
                   "topsusqtq0", "qtq0e", "qtq0eff", "topcMC"]
    # The essential observables we are inspecting
    observables = ["topc", "topcr", "topsus", "topsusqtq0", "qtq0e", "qtq0eff"]
    # observables = ["topsusqtq0"]
    # observables = ["qtq0e"]
    observables = ["topcr"]
    # observables = ["topcMC"]
    # observables = ["energy"]
    # observables = ["qtq0eff"]
    observables = ["topc"]
    observables = ["topsus"]
    # observables = ["topcr"]
    observables = ["topsus", "topsusqtq0"]
    # observables = ["topsus"]

    # Sets the post analysis type to use
    post_analysis_data_type = ["bootstrap_time_series", "bootstrap"]

    # # Full analysis of *all* elements available
    # beta645_L32_analysis(run_pre_analysis=run_pre_analysis,
    #                      run_post_analysis=run_post_analysis,
    #                      only_generate_data=only_generate_data,
    #                      observables=observables,
    #                      post_analysis_data_type=post_analysis_data_type,
    #                      include_b645x48xx3x96=True)

    # main_analysis(run_pre_analysis=False,
    #               run_post_analysis=run_post_analysis,
    #               only_generate_data=only_generate_data,
    #               observables=observables,
    #               post_analysis_data_type=post_analysis_data_type)

    # beta645_L32_analysis(run_pre_analysis=False,
    #                      run_post_analysis=run_post_analysis,
    #                      only_generate_data=only_generate_data,
    #                      observables=observables,
    #                      post_analysis_data_type=post_analysis_data_type,
    #                      include_b645x48xx3x96=False)

    # Only run these in presence of topological susceptibility
    if "topsus" in observables:
        ABC_analysis(run_pre_analysis=False,
                     run_post_analysis=run_post_analysis,
                     only_generate_data=only_generate_data,
                     post_analysis_data_type=post_analysis_data_type)

        BCD_analysis(run_pre_analysis=False,
                     run_post_analysis=run_post_analysis,
                     only_generate_data=only_generate_data,
                     post_analysis_data_type=post_analysis_data_type,
                     include_b645x48xx3x96=False)

        BCD_analysis(run_pre_analysis=False,
                     run_post_analysis=run_post_analysis,
                     only_generate_data=only_generate_data,
                     post_analysis_data_type=post_analysis_data_type,
                     include_b645x48xx3x96=True)


    # distribution_analysis()
    # topc_modes_analysis()
    # lattice_updates_analysis()
    # thermalization_analysis()
    # scaling_analysis()


if __name__ == '__main__':
    main()
