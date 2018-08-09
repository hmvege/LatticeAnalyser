import numpy as np

def get_default_parameters(data_batch_folder=None, obs_exlusions=[]):
    """Container for available observables."""

    #######################################
    ######## Available observables ########
    #######################################
    observables = [
        "plaq", "energy",
        # Topological charge definitions
        # "topc2", "topc4",
        "topc", "topcMC", "topcr", "topcrMC",
        # Topological susceptibility definitions
        "topsus", "topsusMC", "topsusqtq0",
        # Other quantities 
        "topcr",
    ]
    observables_euclidean_time = [
        # Topological charge
        "topct", "topcte",
        # Topological susceptiblity
        "topsust", "topsuste",
        # Other quantities 
        "qtq0e",
        "qtq0eff",
    ]
    observables += observables_euclidean_time
    observables = list(set(set(observables) - set(obs_exlusions)))

    #######################################
    ########### Base parameters ###########
    #######################################
    N_bs = 500
    dryrun = False
    verbose = True
    print_latex = True
    parallel = True
    numprocs = 8
    # Try to load binary file(much much faster)
    load_file = True 
    # If we are to create per-flow datasets as opposite to per-cfg datasets
    create_perflow_data = False
    # Save binary file
    save_to_binary = True
    # Load specific parameters
    NFlows = 1000
    flow_epsilon = 0.01

    #######################################
    ######## Required parameters ##########
    #######################################
    if isinstance(data_batch_folder, type(None)):
        raise ValueError("data_batch_folder must be provided.")

    # Default figures folder
    figures_folder = "figures"

    gif_params = {
        # "gif_observables": ["qtq0e", "qtq0eff"],
        "gif_observables": [], # Uncomment to turn off
        "gif_euclidean_time": 0.5,
        "gif_flow_range": np.linspace(0, 0.6, 100),
        "betas_to_plot": "all",
        "plot_together": False,
        "error_shape": "band",
    }

    # Settings for Giovanni's data
    if "DataGiovanni" in data_batch_folder:
        observables = set(set(observables) - set(observables_euclidean_time))
        observables = list(observables)
        correct_energy = False
        load_file = True
        save_to_binary = False
    else:
        correct_energy = True

    # Number of different sectors we will analyse in euclidean time
    numsplits_eucl = 4

    # Number of different sectors we will analyse in monte carlo time
    MC_time_splits = 4

    # Extraction point in flow time a*t_f for q0 in qtq0
    q0_flow_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # [fermi]

    # Flow time indexes in percent to plot qtq0 in euclidean time at
    euclidean_time_percents = [0, 0.25, 0.50, 0.75, 1.00]

    #### Analysis batch setups
    default_params = {
        "N_bs": N_bs, 
        "dryrun": dryrun,
        "verbose": verbose, 
        "parallel": parallel,
        "numprocs": numprocs,
        "batch_folder": data_batch_folder,
        "figures_folder": figures_folder,
        "observables": observables,
        "load_file": load_file,
        "save_to_binary": save_to_binary, 
        # Flow parameters
        "create_perflow_data": create_perflow_data,
        "flow_epsilon": flow_epsilon, 
        "NFlows": NFlows,
        # Topc histogram parameters
        "topc_y_limits": None,
        "topc2_y_limits": None,
        "topc4_y_limits": None,
        "bin_range": [-10, 10],
        "num_bins_per_int": 4,
        "hist_flow_times": None,
        # Indexes to look at for topct.
        "num_t_euclidean_indexes": 5,
        # Interval selection parameters
        "q0_flow_times": q0_flow_times,
        "euclidean_time_percents": euclidean_time_percents,
        "numsplits_eucl": numsplits_eucl,
        "intervals_eucl": None,
        "MC_time_splits": MC_time_splits,
        "MCInt": None,
        # Various parameters
        "correct_energy": correct_energy,
        "print_latex": print_latex,
        # Gif smearing parameters in the qtq0e observable
        "gif": gif_params,
    }


    default_params["batch_folder"] = data_batch_folder
    return default_params

def main():
    raise ImportError("default_analysis_params not intended as a "
        "standalone module.")

if __name__ == '__main__':
    main()