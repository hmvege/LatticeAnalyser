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
    # run_pre_analysis = False
    # run_post_analysis = False
    # only_generate_data = True

    # Observables selection
    observables = ["plaq", "topc", "topc2", "topc4", "topcr", "topsus",
                   "topsusqtq0", "qtq0e", "qtq0eff"]
    # observables = ["topsusqtq0", "topsus"]
    # observables = ["qtq0eff"]
    # observables = ["qtq0e"]
    # observables = ["topcr"]
    # observables += ["energy"]
    # observables = ["topc"]

    # Sets the post analysis type to use
    post_analysis_data_type = ["bootstrap_time_series", "bootstrap"]

    main_analysis(run_pre_analysis=run_pre_analysis,
                  run_post_analysis=run_post_analysis,
                  only_generate_data=only_generate_data,
                  observables=observables,
                  post_analysis_data_type=post_analysis_data_type)

    # Full analysis of *all* elements available
    beta645_L32_analysis(run_pre_analysis=run_pre_analysis,
                         run_post_analysis=run_post_analysis,
                         only_generate_data=only_generate_data,
                         observables=observables,
                         post_analysis_data_type=post_analysis_data_type,
                         include_b645x48xx3x96=True)

    beta645_L32_analysis(run_pre_analysis=run_pre_analysis,
                         run_post_analysis=run_post_analysis,
                         only_generate_data=only_generate_data,
                         observables=observables,
                         post_analysis_data_type=post_analysis_data_type,
                         include_b645x48xx3x96=False)

    # distribution_analysis()
    # topc_modes_analysis()
    # lattice_updates_analysis()
    # thermalization_analysis()
    # scaling_analysis()


if __name__ == '__main__':
    main()
