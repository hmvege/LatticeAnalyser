# For Python 3 string features (much nicer than python 2)
from __future__ import unicode_literals

import sys
import os
import copy
import numpy as np
import re
import pandas as pd
import json
import types
import datetime
import calendar
import warnings

__all__ = ["FlowDataReader", "check_folder", "get_NBoots",
           "write_data_to_file", "write_raw_analysis_to_file",
           "load_observable", "check_relative_path"]


class _DirectoryTree:
    def __init__(self, batch_name, batch_folder, dryrun=False, verbose=True):
        self.flow_tree = {}
        self.obs_tree = {}
        self.CURRENT_FOLDER = os.getcwd()
        self.data_batch_folder = batch_folder
        self.observables_list = ["plaq", "topc", "energy", "topct"]
        self.found_flow_observables = []
        self.found_observables = []
        self.batch_name = batch_name
        self.dryrun = dryrun
        self.verbose = verbose

        # Checks that the output folder actually exist
        if not os.path.isdir(self.data_batch_folder):
            raise IOError(("No folder name output at location %s"
                           % self.data_batch_folder))

        # Retrieves folders and subfolders
        self.batch_name_folder = os.path.join(self.data_batch_folder,
                                              batch_name)

        # Retrieves potential .npy files
        self.observables_binary_files = {}
        for file in os.listdir(self.batch_name_folder):
            head, ext = os.path.splitext(file)
            if ext == ".npy":
                self.observables_binary_files[head] = os.path.join(
                    self.batch_name_folder, file)

        # Gets the regular configuration observables
        self.non_flow_obs_folder_exist = False
        obs_path = os.path.join(self.batch_name_folder, "observables")
        if os.path.isdir(obs_path) and len(os.listdir(obs_path)) != 0:
            self.observables_folder = obs_path
            self.non_flow_obs_folder_exist = True

            # Peformes sort on file paths, such that they correspond to
            # correct observable.
            _obs_file_paths = os.listdir(self.observables_folder)
            _tmp_list = []
            for _obs in self.observables_list:
                for _obs_path in _obs_file_paths:
                    if _obs in _obs_path:
                        _tmp_list.append(_obs_path)
            _obs_file_paths = _tmp_list

            for obs, file_name in zip(self.observables_list,
                                      _obs_file_paths):

                # Removes .DS_Store
                if obs.startswith("."):
                    continue

                obs_path = os.path.join(self.observables_folder, file_name)
                if os.path.isfile(obs_path):
                    self.obs_tree[obs] = obs_path
                    self.found_observables.append(obs)

        # Gets paths to flow observable
        # Creates the flow observables path
        self.flow_path = os.path.join(self.batch_name_folder,
                                      "flow_observables")

        # Checks that there exists a flow observable folder
        if os.path.isdir(self.flow_path):
            # Goes through the flow observables
            for flow_obs in self.observables_list:
                # Creates flow observables path
                obs_path = os.path.join(self.flow_path, flow_obs)

                # Checks if the flow observable path exists
                if os.path.isdir(obs_path):
                    # Finds and sets the observable file paths
                    flow_obs_dir_list = []

                    # In case batch contains an empty observable folder, it
                    # will be skipped and not added to the observable list.
                    if len(os.listdir(obs_path)) == 0:
                        continue

                    for obs_file in os.listdir(obs_path):
                        # Removes .DS_Store
                        if obs_file.startswith("."):
                            continue

                        flow_obs_dir_list.append(
                            os.path.join(obs_path, obs_file))

                    # Sorts list by natural sorting
                    self.flow_tree[flow_obs] = \
                        self.natural_sort(flow_obs_dir_list)

                    self.found_flow_observables.append(flow_obs)

    @staticmethod
    def natural_sort(l):
        # Natural sorting
        def convert(text): return int(text) if text.isdigit() else text.lower()

        def alphanum_key(key): return [convert(c)
                                       for c in re.split('(\d+)', key)]
        return sorted(l, key=alphanum_key)

    def get_flow(self, obs):
        """
        Retrieves flow observable files.
        """
        if obs in self.flow_tree:
            return self.flow_tree[obs]
        else:
            raise Warning(("Flow observable \"%s\" was not found in "
                           "possible observables: %s" %
                           (obs, ", ".join(self.flow_tree.keys()))))

    def get_obs(self, obs):
        """
        Retrieves observable files.
        """
        if obs in self.obs_tree:
            return self.obs_tree[obs]
        else:
            raise Warning(("Observable \"%s\" was not found in "
                           "possible observables: %s" %
                           (obs, ", ".join(self.flow_tree.keys()))))

    def get_found_flow_observables(self):
        """
        Returns list over all found observables.
        """
        return self.found_flow_observables

    def get_found_observables(self):
        """
        Returns list over all found observables.
        """
        return self.found_observables

    def __str__(self):
        """Prints the folder structure."""
        return_string = "Folder structure:"
        return_string += "\n{0:<s}".format(self.batch_name_folder)

        # Builds the non-flow observable file paths
        if self.non_flow_obs_folder_exist:

            return_string += "\n  {0:<s}/{1:<s}".format(
                self.data_batch_folder, "observables")

            for obs, file_name in zip(self.observables_list,
                                      os.listdir(self.observables_folder)):

                return_string += "\n    {0:<s}".format(
                    os.path.join(self.observables_folder, file_name))

        if len(self.observables_binary_files) != 0:

            for head in self.observables_binary_files:
                return_string += "\n  {0:<s}".format(
                    self.observables_binary_files[head].split("/")[-1])

        # Builds the flow observable file paths
        if os.path.isdir(self.flow_path):
            return_string += "\n  {0:<s}".format(self.flow_path)

            for flow_obs in self.observables_list:
                obs_path = os.path.join(self.flow_path, flow_obs)
                return_string += "\n    {0:<s}".format(obs_path)

                for obs_file in os.listdir(obs_path):
                    return_string += "\n      {0:<s}".format(
                        os.path.join(obs_path, obs_file))
        return return_string


class _FolderData:
    """
    Class for retrieving flow files.
    """

    def __init__(self, file_tree, observable, flow_cutoff=1000):
        """
        Loads the data immediately, and thus sets up a FolderData object
        containing data found in a folder.

        Args:
            file_tree: _DirectoryTree object containing a list over
                observables and their locations.
            observable: name string of observable we are loading.
            flow_cutoff: integer of what flow are to cut off at. Default is
                at a 1000 flows.

        """

        # Retrieves file from file tree
        files = file_tree.get_flow(observable)

        # Stores the file tree as a global constant for later use by settings
        # and perflow creator.
        self.file_tree = file_tree

        # Stores the observable name
        self.observable = observable

        if files == None:
            print("No observables of type %s found in folder: %s"
                  % (observable, folder))
            return

        # Booleans to ensure certain actions are only done once
        read_meta_data = True
        retrieved_sqrt8_flow_time = False
        retrieved_flow_time = False

        # Number of rows to skip after meta-data has been read
        N_rows_to_skip = 0

        # Long-term storage variables
        self.meta_data = {}
        self.data_y = []
        self.data_x = False

        # Ensures we handle the data as a folder
        if type(files) != list:
            self.files = [files]
        else:
            self.files = files

        # Number of files is the length of files in the the folder
        N_files = len(self.files)
        # Goes through files in folder and reads the contents into a file
        for i, file in enumerate(self.files):
            # Gets the metadata
            with open(file) as f:
                # Reads in meta data as long as the first element on the
                # line is a string.
                while read_meta_data:
                    line = f.readline().split(" ")
                    if line[0].isalpha():
                        self.meta_data[str(line[0])] = float(line[-1])
                        N_rows_to_skip += 1
                    else:
                        # Stores number of rows(if we are on old or new
                        # data reading).
                        N_rows = len(line)

                        # Exits while loop
                        read_meta_data = False

            # Loads the data and places it in a list
            if N_rows == 3:
                # Uses pandas to read data (quick!)
                data_frame = pd.read_csv(file, skiprows=N_rows_to_skip,
                                         sep=" ",
                                         names=["t", "sqrt8t",
                                                self.observable],
                                         header=None)

                # Only retrieves flow once
                if not retrieved_sqrt8_flow_time:

                    # This is the a*sqrt(8*t), kinda useless
                    # self.data_flow_time = _x

                    # Pandas, much faster than numpy for some reason
                    self.data_flow_time = \
                        data_frame["sqrt8t"].values[:flow_cutoff]

                    retrieved_sqrt8_flow_time = True

            elif N_rows == 2:
                # If it is new observables with no sqrt(8*t)
                # Uses pandas to read data (quick!)
                data_frame = pd.read_csv(file, skiprows=N_rows_to_skip,
                                         sep=" ",
                                         names=["t", self.observable],
                                         header=None)

            else:
                # If we have a topct-like variable, we will read in
                # NT rows as well. Sets up header names
                header_names = ["t"] + ["t%d" % j for j in range(N_rows-1)]
                data_frame = pd.read_csv(file, skiprows=N_rows_to_skip,
                                         sep=" ", names=header_names,
                                         header=None)

            # Only retrieves indexes/flow-time*1000 once
            if not retrieved_flow_time:
                self.data_x = data_frame["t"].values[:flow_cutoff]  # Pandas
                retrieved_flow_time = True

            # Appends observables
            if N_rows > 3:
                # Appends an array if we have data in more than one dimension
                self.data_y.append(
                    np.asarray([data_frame[iname].values[:flow_cutoff]
                                for iname in header_names[1:]]).T)
            else:
                self.data_y.append(
                    data_frame[self.observable].values[:flow_cutoff])

        self.data_y = np.asarray(self.data_y)

    def create_settings_file(self, dryrun=False, verbose=False):
        """
        Function for storing run info.
        """
        # Checking that settings file does not already exist
        setting_file_path = os.path.join(
            self.file_tree.batch_folder,
            "run_settings_%s.txt" % self.observable)

        # Creating string to be passed to info file
        info_string = ""
        info_string += "Batch %s" % self.file_tree.batch_name
        info_string += "\nObservables %s" % " ".join(
            self.file_tree.get_found_observables())
        info_string += "\nNConfigs %d" % self.data_y.shape[0]
        for key in self.meta_data:
            info_string += "\n%s %s" % (key, self.meta_data[key])

        # Creating file
        if not dryrun:
            with open(setting_file_path, "w") as f:
                # Appending batch name
                f.write(info_string)

        # Prints file if we have not dryrun or if verbose option is turned on
        if verbose:
            print "\nSetting file:", info_string, "\n"

        print "Setting file %s created." % setting_file_path


class FlowDataReader:
    """
    Class for reading all of the data from a batch.

    Modes:
    - Read all data to a single object and then write it to a single file
    - Load a single file

    Plan:
    1. Retrieve file paths.
    2. Retrieve data.
    3. Concatenate data to a single matrix
    4. Write data to a single file in binary

    """
    beta_to_spatial_size = {6.0: 24, 6.1: 28, 6.2: 32, 6.45: 48}
    fobs = ["plaq", "energy", "topc"]

    def __init__(self, batch_name, batch_folder, figures_folder,
                 load_binary_file=None, save_to_binary=False,
                 flow_epsilon=0.01, NCfgs=None, create_perflow_data=False,
                 correct_energy=False, N_spatial=None, N_temporal=None,
                 verbose=True, dryrun=False):
        """
        Class that reads and loads the observable data.

        Args:
                batch_name: string containing batch name.
                batch_folder: string containing location of batch.
                figures_folder: location of where to place figures from
                    analysis.
                load_binary_file: bool if we will try to look for a .npy file
                    in batch_folder/batch_name. Will look for the topct
                    file as well.
                save_to_binary: bool, optional, will try and save file to
                    binary for quicker loading.
                flow_epsilon: flow epsilon in flow of file we are loading.
                    Default is 0.01.
                create_perflow_data: boolean if we are to create a folder
                    containing per-flow data(as opposite of per-config).
                correct_energy: Optional, bool. If true, energy is by
                    dividing by 64.
                N_spatial: Optional, int. Spatial extent of lattice.
                N_temporal: Optional, int. Temporal extent of lattice.
                verbose: bool, a more verbose run. Default is True.
                dryrun: bool, dryrun option. Default is False.
        """

        self.verbose = verbose
        self.dryrun = dryrun
        self.data = {}
        self.correct_energy = correct_energy
        self.batch_name = batch_name
        self.batch_folder = batch_folder
        self.figures_folder = figures_folder
        self.N = N_spatial
        self.NT = N_temporal
        self.lattice_size = N_spatial**3 * N_temporal

        self.__print_load_info()

        self.file_tree = _DirectoryTree(self.batch_name, self.batch_folder,
                                        dryrun=dryrun, verbose=verbose)

        if NCfgs == None:
            raise ValueError(
                "missing number of observables associated with batch.")
        else:
            self.NCfgs = NCfgs

        if load_binary_file:
            # Retrieves binary file paths if they exist
            topct_fp = None
            obs_fp = None
            for bin_file_key in self.file_tree.observables_binary_files:
                if "topct" in bin_file_key:
                    topct_fp = \
                        self.file_tree.observables_binary_files[bin_file_key]
                else:
                    obs_fp = \
                        self.file_tree.observables_binary_files[bin_file_key]

            # Loads the topct data
            if topct_fp != None:
                # Loads binary
                self.__load_files(topct_fp, ["topct"],
                                  flow_epsilon=flow_epsilon)
            else:
                if os.path.isdir(self.file_tree.flow_path) and \
                        "topct" in self.file_tree.found_flow_observables:

                    # Loads in file from non binary
                    print("No binary file found for topct. "
                          "Loads from .dat files.")
                    self.__retrieve_observable_data(
                        ["topct"], create_perflow_data=create_perflow_data)

                    # Writes topct to binary for future fast loading
                    if "topct" in self.file_tree.get_found_flow_observables():
                        self.__write_file(["topct"])
                else:
                    print "No data found for topct"

            # Loads the other observable data
            if obs_fp != None:
                # Loads binary if it exists
                self.__load_files(obs_fp, self.fobs,
                                  flow_epsilon=flow_epsilon)
            else:
                if os.path.isdir(self.file_tree.flow_path):
                    # Loads in file from non binary
                    print("No binary file found for %s. Loads from .dat "
                          "files." % (", ".join(self.fobs)))
                    self.__retrieve_observable_data(
                        self.fobs, create_perflow_data=create_perflow_data)

                    # Writes observables to binary file for future loading.
                    self.__write_file(["plaq", "energy", "topc"])
                else:
                    print "No data found for %s" % (", ".join(self.fobs))

            if create_perflow_data:
                self._create_perflow_data()
        else:
            # Loads .dat files
            print("Retrieving data for batch %s from folder %s" %
                  (self.batch_name, self.batch_folder))

            observables_to_retrieve = self.file_tree.flow_tree
            self.__retrieve_observable_data(
                observables_to_retrieve,
                create_perflow_data=create_perflow_data)

            if save_to_binary:
                self.write_single_file()

        # Checks that provided folder exists
        check_folder(self.figures_folder, self.dryrun, verbose=self.verbose)

        # Sets the temporal dimension points
        if "topct" in self.data.keys():
            self.NT = self.data["topct"]["obs"].shape[-1]

        self.observables = self.data.keys()

    def __call__(self, obs):
        return copy.deepcopy(self.data[obs])

    def has_observable(self, obs):
        """Checks that the observable we are retrieving exists."""
        if not obs in self.observables:
            print "%s not found in data: %s" % (obs, self.observables)
            return False
        else:
            return True

    def write_parameter_file(self):
        """Writes a parameter file for the analysis of a given batch."""

        post_analysis_path = os.path.join(self.batch_folder, self.batch_name,
                                          "post_analysis_data")
        param_file_path = os.path.join(post_analysis_path, "params.json")
        json_dict = {}

        # Writes parameters to json dictionary
        json_dict["beta"] = self.beta
        json_dict["NFlows"] = self.NFlows
        json_dict["NCfgs"] = self.NCfgs

        # Prints configuration file content if verbose or dryrun is true
        if self.dryrun or self.verbose:
            print "Writing json parameter file at location {0:<s}:".format(
                param_file_path)
            print json.dumps(json_dict, indent=4, separators=(", ", ": "))

        # Checks if the post analysis folder exists.
        check_folder(post_analysis_path, self.dryrun, verbose=self.verbose)

        # Creates configuration file
        if not self.dryrun:
            with file(param_file_path, "w+") as json_file:
                json.dump(json_dict, json_file, indent=4)

    def __print_load_info(self):
        load_job_info = "="*160
        load_job_info += "\n" + "Data loader"
        load_job_info += "\n{0:<20s}: {1:<20s}".format("Batch name",
                                                       self.batch_name)
        load_job_info += "\n{0:<20s}: {1:<20s}".format("Batch folder",
                                                       self.batch_folder)
        print load_job_info

    def __retrieve_observable_data(self, observables_to_retrieve,
                                   create_perflow_data=False):
        """
        Retrieves observable data when there is no binary file to retrieve from.
        """

        _NFlows = []
        _beta_values = []
        for obs in observables_to_retrieve:
            # Creates a dictionary to hold data associated with an observable
            self.data[obs] = {}

            _data_obj = _FolderData(self.file_tree, obs)

            self.data[obs]["t"] = _data_obj.data_x
            self.data[obs]["obs"] = _data_obj.data_y
            self.data[obs]["beta"] = _data_obj.meta_data["beta"]
            self.data[obs]["lattice_size"] = self.lattice_size
            self.data[obs]["FlowEpsilon"] = _data_obj.meta_data["FlowEpsilon"]
            self.data[obs]["NFlows"] = _data_obj.meta_data["NFlows"]
            self.data[obs]["batch_name"] = self.file_tree.batch_name
            self.data[obs]["batch_data_folder"] = \
                self.file_tree.data_batch_folder

            if obs == "energy" and self.correct_energy:
                self.data[obs]["obs"] *= 1.0/64.0

            if create_perflow_data:
                _data_obj.create_perflow_data(verbose=self.verbose)

            # Stores all the number of flow values
            _NFlows.append(self.data[obs]["NFlows"])
            _beta_values.append(self.data[obs]["beta"])

            del _data_obj

            print "Retrieved %s. Size: %.2f MB" % (
                obs, sys.getsizeof(self.data[obs]["obs"])/1024.0/1024.0)

        # Checks that all values have been flowed for an equal amount of time
        assert len(set(_NFlows)) == 1, ("flow times differ for the different "
                                        "observables: %s" % (
                                            ", ".join(_NFlows)))
        self.NFlows = int(_NFlows[0])

        # Sets a global beta value for the batch
        assert np.asarray(_beta_values).all(), "beta values are not equal."
        self.beta = _beta_values[0]

    @staticmethod
    def __get_size_and_beta(input_file):
        """Gets size and beta value from binary file name."""
        _parts = input_file.split("/")[-1].split("_")
        N, beta = _parts[:2]
        beta = float(beta.strip(".npy"))
        return int(N), float(beta)

    def __load_files(self, input_file, obs_list, flow_epsilon):
        """Binary file loader for quicker loading of data."""
        raw_data = np.load(input_file)

        N, beta = self.__get_size_and_beta(input_file)

        if "topct" in obs_list:
            # Since there is only one observable, we split into number of
            # configs we have.
            num_splits = self.NCfgs
        else:
            # Only splits data into the number of observables we have
            num_splits = len(obs_list)

        raw_data_splitted = \
            np.array(np.split(raw_data[:, 1:], num_splits, axis=1))

        _NFlows = []
        _betas = []

        # Loads from the plaq, energy, topc and topct
        for i, obs in enumerate(obs_list):
            self.data[obs] = {}

            # Gets the flow time
            self.data[obs]["t"] = raw_data[:, 0]

            if obs == "topct":
                # Reshapes and roll axis to proper shape for later use
                self.data[obs]["obs"] = raw_data_splitted
            else:
                self.data[obs]["obs"] = raw_data_splitted[i].T

            # Fills in different parameters
            self.data[obs]["beta"] = beta
            self.data[obs]["FlowEpsilon"] = flow_epsilon
            self.data[obs]["NFlows"] = self.data[obs]["obs"].shape[1]
            self.data[obs]["batch_name"] = self.batch_name
            self.data[obs]["batch_data_folder"] = self.batch_folder
            self.data[obs]["lattice_size"] = self.lattice_size

            _NFlows.append(self.data[obs]["NFlows"])
            _betas.append(beta)

        # Checks that all values have been flowed for an equal amount of time
        assert len(set(_NFlows)) == 1, ("flow times differ for the different "
                                        "observables: %s" % (
                                            ", ".join(_NFlows)))
        self.NFlows = int(_NFlows[0])

        # Sets a global beta value for the batch
        assert np.asarray(_betas).all(), "beta values are not equal."
        self.beta = _betas[0]

        print "Loaded %s from file %s. Size: %.2f MB" % (
            ", ".join(obs_list), input_file, raw_data.nbytes/1024.0/1024.0)

    def write_single_file(self):
        self.__write_file(["plaq", "energy", "topc"])
        if "topct" in self.file_tree.get_found_flow_observables():
            self.__write_file(["topct"])

    def __write_file(self, obs_to_write):
        """
        Internal method for writing observable to file.
        """
        raw_data = self.data[obs_to_write[0]]["t"]

        for obs in obs_to_write:
            # Checks if we have an array of observables, e.g. topct
            if obs == "topct" and len(obs_to_write) == 1:
                # Rolls axis to make it on the correct format
                _temp_rolled_data = np.rollaxis(self.data[obs]["obs"].T, 0, 3)

                # Gets the axis shape in order to then flatten the data
                _shape = _temp_rolled_data.shape
                _temp_rolled_data = _temp_rolled_data.reshape(
                    _shape[0], _shape[1]*_shape[2])

                raw_data = np.column_stack((raw_data, _temp_rolled_data))
            else:
                raw_data = np.column_stack((raw_data, self.data[obs]["obs"].T))

        beta_value = self.data[obs_to_write[0]]["beta"]

        # Sets up file name. Format {N}_{beta}.npy and {N}_{beta}_topct.npy
        if len(obs_to_write) == 1 and obs_to_write[0] == "topct":
            file_name = "%d_%1.2f_topct" % (self.N, beta_value)
        else:
            file_name = "%d_%1.2f" % (self.N, beta_value)

        file_path = os.path.join(self.file_tree.batch_name_folder, file_name)

        # Saves as binary
        np.save(file_path, raw_data)

        print "%s written to a single file at location %s.npy." % (
            ", ".join(obs_to_write), file_path)

        return file_path + ".npy"

    def _create_perflow_data(self):
        """Function for creating per-flow data, as opposed to per-config."""

        for observable in self.data:
            obs_data = self.data[observable]

            # Creating per flow folder
            per_flow_folder = os.path.join("..", obs_data["batch_data_folder"],
                                           obs_data["batch_name"], "perflow")
            check_folder(per_flow_folder, self.dryrun, verbose=self.verbose)

            # Creates observable per flow folder
            per_flow_observable_folder = os.path.join(
                per_flow_folder, observable)
            check_folder(per_flow_observable_folder, self.dryrun,
                         verbose=self.verbose)

            # Retrieving number of configs and number of flows
            NConfigs = len(obs_data["obs"])
            NFlows = obs_data["NFlows"]

            # Re-storing files in a per flow format
            for iFlow in xrange(NFlows):
                # Setting up new per-flow file
                flow_file = os.path.join(
                    per_flow_folder, observable,
                    obs_data["batch_name"] + "_flow%05d.dat" % iFlow)

                # Saving re-organized data to file
                if not self.dryrun:
                    np.savetxt(flow_file, obs_data["obs"][:, iFlow],
                               fmt="%.16f",
                               header="t %f \n%s" % (
                        iFlow*obs_data["FlowEpsilon"], observable))

                # Prints message regardless of dryrun and iff
                if self.verbose:
                    print "%s created." % flow_file

            print "Per flow data for observable %s created." % observable


def load_observable(params):
    """
    Method for loading observable .dat files in the 'observables' folder.

    Args:
        params: dict, takes a parameter dictionary, which contains entries 
            batch_name, batch_folder, dryrun, verbose

    Returns: 
        dictionary with entry 'meta' containing metadata from the run, and
            and entry 'obs' that contains the different observables
    """
    data = {
        "meta": {},
        "obs": {},
    }

    file_tree = _DirectoryTree(params["batch_name"], params["batch_folder"],
                               dryrun=params["dryrun"],
                               verbose=params["verbose"])

    # Loops over found observables
    for obs, obs_path in file_tree.obs_tree.items():

        # Counter for the number of meta data lines
        _num_meta_lines = 0

        with open(obs_path, "r") as f:

            # Reads meta data
            for l in f:

                # Will continue to retrieve meta data while possible
                _lvals = l.split(" ")

                # Checks if first element is string, descriptor
                if _lvals[0].isalpha():

                    # Conerts element to float
                    data["meta"][_lvals[0]] = float(_lvals[1])
                    _num_meta_lines += 1
                else:

                    # Meta data done when first element is not alpha,
                    # i.e. is a integer.
                    break

        data["obs"][obs] = np.loadtxt(obs_path, skiprows=_num_meta_lines)

    if params["verbose"]:
        _data_size = list(map(sys.getsizeof, data["obs"].values()))
        print "Retrieved {0:s}. Size: {1:.2f} MB".format(
            ", ".join(file_tree.get_found_observables()),
            sum(_data_size)/1024.0/1024.0)

    return data


class SlurmDataReader:
    """Class for reading in slurm data to a dictionary."""
    max_small_iter = 100
    max_large_iter = int(1e6)

    def __init__(self, p):
        """
        Loads slurm data and places it in an dictionary.

        Args:
            p: slurm .out-file path

        Returns:
            dict, dictionary with slurm data
        """
        self.p = p
        self.data = {}
        self.months_to_int = {v: k for k, v in enumerate(calendar.month_abbr)}

    @staticmethod
    def _key_cleaner(k):
        """
        Minor function for cleaning a key to dictionary,
        in case it contains parenthesis.""
        """
        return str(k.split("(")[0].replace(" ", "_"))

    @staticmethod
    def _clean_str_list(l):
        """
        Cleans string list by stripping spaces, trims whitespace and ensure 
        ascii.
        """

        # Splits into observable header
        _lsplit = list(map(lambda _ll: _ll.strip(" "), l.split(" ")))

        # Ensures ascii
        _lsplit = list(map(str, _lsplit))

        # Removes everything of length 0
        _lsplit = filter(lambda _ll: False if len(_ll) == 0 else True,
                         _lsplit)

        return _lsplit

    def _get_timestamp(self, l):
        """Gets the time stamp from line l that is always printed the 
        beginning and end of output file."""

        _tmp_start_time = l.split(" at ")[-1]
        _, _month, _day, _time_stamp, _, _year = \
            _tmp_start_time.split(" ")
        _time_stamp = list(map(int, _time_stamp.split(":")))

        _time = datetime.datetime(
            int(_year), self.months_to_int[_month], int(_day),
            hour=_time_stamp[0], minute=_time_stamp[1],
            second=_time_stamp[2])

        return _time

    def _get_starting_job(self):
        """Searches for 'Starting job' string."""
        for i in range(self.max_small_iter):
            l = self.f.readline()

            if "Starting job" in l:
                # Gets job id
                _job_id = re.findall(r"Starting job ([\d]+)", l)[0]
                self.data["job_id"] = int(_job_id)

                # Gets job name
                _job_name = re.findall(r'\("([\w\._]+)"\)', l)[0]
                self.data["job_name"] = _job_name

                # Gets start time
                self.data[str("start_time")] = self._get_timestamp(l)
                break
        else:
            warnings.warn("Max iter reached for _get_starting_job()")

    def _read_parameters(self):
        """Reads the first '=' block which contains run parameters."""
        start_line_counted = False

        for i in range(self.max_small_iter):
            l = self.f.readline()

            # Skipps line to begin reading
            if not start_line_counted and l[0] == "=":
                start_line_counted = True
                continue

            # If we encounter one more "=", then break
            if start_line_counted and l[0] == "=":
                break

            # Splits parameters
            _lparams = l.strip("\n").split(":")

            # In case we are not in the parameter section yet
            if len(_lparams) != 2:
                continue

            _key = self._key_cleaner(_lparams[0])
            _val = _lparams[1].lstrip(" ").rstrip(" ")

            if _val[0].isnumeric():

                # If right hand side is a number
                if len(_val.split(" ")) > 1:
                    # In case we are dealing with multiple numbers
                    self.data[_key] = \
                        list(map(int, _val.split(" ")))

                else:

                    # For single number cases
                    if _val.isdigit():
                        self.data[_key] = int(_val)
                    else:
                        self.data[_key] = float(_val)

            else:

                if len(_val.split(",")) > 1:
                    # For observables list
                    self.data[_key] = \
                        list(map(str, _val.split(",")))

                else:

                    # For regular string parameter
                    self.data[_key] = str(_val)
        else:
            warnings.warn("Max iter reached for _read_parameters()")

    def _read_config_generation(self):
        """
        Reads in configuration generation till an "=" is encountered.
        """

        # Retrieves either thermalized done or configuration
        for i in range(self.max_small_iter):
            _l = self.f.readline()
            if "Configuration" in _l:
                self.data["start_config"] = \
                    _l.split(" ")[1].lstrip(" ").rstrip(" ")
                break

            if "Thermalization complete." in _l or "Termalization complete." in _l:
                self.data["therm_accept_rate"] = float(_l.split(" ")[-1])
                break

        else:
            warnings.warn("Max iter reached for thermalization/start config "
                          "loading _read_config_generation()")

        # Reads in parameter heading
        for i in range(self.max_small_iter):
            _l = self.f.readline()

            husk å fikse slik at en kan lese in loaded files hvis en ikke genererer!
            _lsplit = self._clean_str_list(_l)

            if _lsplit[0] == "i":
                self.data["printed_obs_header"] = _lsplit[1:-3]
                break
        else:
            warnings.warn("Max iter reached for header loading in "
                          "_read_config_generation()")

        _obs_dict = {k: [] for k in self.data["printed_obs_header"]}
        # _loaded_observables = []
        _written_observables = []

        # Reads in value/config loaded
        for i in range(self.max_large_iter):
            _l = self.f.readline()
            _lsplit = self._clean_str_list(_l)

            # If we encounter summary section, break
            if _l[0] == "=":
                break

            # If we have generated a config, store that
            if _lsplit[0].isdigit():
                for j, _obs in enumerate(self.data["printed_obs_header"]):
                    _obs_dict[_obs] = float(_lsplit[j+1])

                _written_observables = _lsplit[-2]
        else:
            warnings.warn("Max iter reached for obs loading in"
                          " _read_config_generation()")

        # self.data["loaded_obs"] = _loaded_observables
        self.data["obs"] = _obs_dict
        self.data["written_obs"] = _written_observables

    def _read_final_output(self):
        """
        Reads final slurm output.
        """

        for i in range(self.max_small_iter):
            _l = self.f.readline()
            _lsplit = _l.strip("\n").split(" ")

            if "Acceptancerate" in _lsplit[0]:
                self.data["accept_rate"] = float(_lsplit[1])
                continue

            if "Average update time" in _l:
                self.data["avg_update_time"] = float(_lsplit[-2])
                continue

            if "Total update time for" in _l:
                self.data["num_updates"] = float(_lsplit[4])
                self.data["update_time"] = float(_lsplit[-2])
                continue

            if "=" == _l[0]:
                break
        else:
            warnings.warn("Max iter reached for first '=' block in "
                          "_read_final_output()")

        # List for storing .dat file output locations
        _dat_obs_files = []

        # Dictionary for storing final output values
        _obs_final_output = {_obs: {}
                             for _obs in self.data["printed_obs_header"]}

        for i in range(self.max_small_iter):
            _l = self.f.readline()
            _lsplit = self._clean_str_list(_l)

            # Stores .dat files
            if _l.startswith("/"):
                _dat_obs_files.append(_lsplit[0])

            # Stores final obs output
            if len(_lsplit) > 1 and \
                    _lsplit[0] in self.data["printed_obs_header"]:

                _obs_final_output[_lsplit[0]] = {
                    "mean": float(_lsplit[1]),
                    "var": float(_lsplit[2]),
                    "std": float(_lsplit[3]),
                }

            # Stores program time
            if "Program complete" in _l:
                _times = re.findall(r"([\d]+\.[\d]+)", _l)
                self.data["program_times"] = {
                    "hours": float(_times[0]),
                    "seconds": float(_times[1]),
                }

            # Stores loaded module files if 'Currently Loaded Modulefiles'
            # is in _l.
            if "Currently Loaded Modulefiles" in _l:
                _l = self.f.readline()
                # 1) gcc/5.2.0            2) openmpi.gnu/1.10.2
                _found_modeles = re.findall(r"\d{1}\) ([\./\w]+)", _l)
                self.data["job_modules"] = _found_modeles

            # If first element is job id, store total time spent
            if len(_lsplit) > 1 and _lsplit[0] == str(self.data["job_id"]):
                _elapsed_time = _lsplit[-3]
                if "-" in _elapsed_time:

                    # Splits {days}-{hours}:{minutes}:{seconds}
                    _elapsed_time = _elapsed_time.split("-")

                    # Converts {hours}:{minutes}:{seconds} to integers
                    _tmp = list(map(int, _elapsed_time[-1].split(":")))

                    # Gathers results into a list
                    _elapsed_time = [int(_elapsed_time[0])*24 + _tmp[0],
                                     _tmp[1], _tmp[2]]
                else:
                    _tmp = list(map(int, _elapsed_time.split(":")))
                    _elapsed_time = [_tmp[0], _tmp[1], _tmp[2]]

                self.data["elapsed_time"] = _elapsed_time

            # If first element is 'job', store get end time. Then break loop
            if len(_lsplit) > 1 and _lsplit[0] == "Job" and \
                    _lsplit[1] == str(self.data["job_id"]):
                self.data["end_time"] = self._get_timestamp(_l)
                break

        else:
            warnings.warn("Max iter reached for final output stats searching "
                          "in _read_final_output().")

        self.data["obs_dat_files"] = _dat_obs_files
        self.data["obs_final_output"] = _obs_final_output

    def read(self, verbose=False):
        """Method for reading the slurm output file.

        Args:
            verbose: bool, default False. More verbose output.

        Returns:
            dictionary containing slurm file contents.
        """

        with open(self.p, "r") as self.f:

            # Searches for starting string
            self._get_starting_job()

            # Searches for "="
            self._read_parameters()

            # Searches for configuration output
            self._read_config_generation()

            # Reads final values
            self._read_final_output()

        return self.data

    def __str__(self):
        """Pretty prints data."""
        _str = "Slurm data for {}".format(self.p)
        for key in self.data:
            _str += str("\n{:40}: ".format(key)) + str(self.data[key])
        return _str


def check_folder(folder_name, dryrun=False, verbose=False):
    """
    Checks that figures folder exist, and if not will create it.

    Args:
        folder_name: str, folder to check if exist.
        dryrun: bool, default is False.
        verbose: bool, default is False.
    """
    if not os.path.isdir(folder_name):
        if dryrun or verbose:
            print "> mkdir %s" % folder_name
        if not dryrun:
            os.mkdir(folder_name)


def check_relative_path(p):
    """
    Function for ensuring are in the correct relative path.

    Args:
        p: str, path we are checking

    Return:
        relative path to p
    """
    if not os.path.isdir(p):
        p = os.path.join("..", p)
        return check_relative_path(p)
    else:
        return p


def get_NBoots(raw):
    """Recursive method for getting the number of bootstraps."""
    if isinstance(raw, dict):
        return get_NBoots(raw.values()[0])
    else:
        return raw.shape[-1]


def write_data_to_file(analysis_object, save_as_txt=False):
    """
    Function that write data to file for the post analysis class

    Args:
            analysis_object: object of FlowAnalyser class
            post_analysis_folder: optional string output folder, default 
                    is ../output/analyzed_data
    """

    dryrun = analysis_object.dryrun
    verbose = analysis_object.verbose
    post_analysis_folder = analysis_object.post_analysis_folder

    # Retrieves beta value and makes it into a string
    beta_string = str(analysis_object.beta).replace(".", "_")

    # Ensures that the post analysis data folder exists
    check_folder(post_analysis_folder, dryrun, verbose=verbose)

    # Retrieves analyzed data
    x = copy.deepcopy(analysis_object.x)
    y_org = copy.deepcopy(analysis_object.unanalyzed_y)
    y_err_org = copy.deepcopy(
        analysis_object.unanalyzed_y_std *
        analysis_object.autocorrelation_error_correction)
    y_bs = copy.deepcopy(analysis_object.bs_y)
    y_err_bs = copy.deepcopy(
        analysis_object.bs_y_std *
        analysis_object.autocorrelation_error_correction)
    y_jk = copy.deepcopy(analysis_object.jk_y)
    y_err_jk = copy.deepcopy(
        analysis_object.jk_y_std *
        analysis_object.autocorrelation_error_correction)

    # Stacks data to be written to file together
    if not analysis_object.autocorrelation_performed:
        data = np.stack((x, y_org, y_err_org, y_bs, y_err_bs,
                         y_jk, y_err_jk), axis=1)
    else:
        tau_int = copy.deepcopy(
            analysis_object.integrated_autocorrelation_time)
        tau_int_err = copy.deepcopy(
            analysis_object.integrated_autocorrelation_time_error)
        sqrt2tau_int = copy.deepcopy(
            analysis_object.autocorrelation_error_correction)
        data = np.stack((x, y_org, y_err_org, y_bs, y_err_bs,
                         y_jk, y_err_jk, tau_int, tau_int_err,
                         sqrt2tau_int), axis=1)

    # Retrieves compact analysis name
    observable = analysis_object.observable_name_compact

    # Sets up file name and file path
    if not analysis_object.autocorrelation_performed:
        fname = "%s_no_autocorr" % observable
        header_string = ("observable %s beta %s\nt original original_error bs"
                         " bs_error jk jk_error" % (observable, beta_string))
    else:
        fname = "%s" % observable
        header_string = ("observable %s beta %s\nt original original_error bs "
                         "bs_error jk jk_error tau_int tau_int_err "
                         "sqrt2tau_int" % (observable, beta_string))

    fname_path = os.path.join(post_analysis_folder, fname)

    # Saves data to file
    if not dryrun:
        if save_as_txt:
            np.savetxt(fname_path + ".txt", data, fmt="%.16f",
                       header=header_string)
        else:
            np.save(fname_path, data)

    if analysis_object.verbose or analysis_object.dryrun:
        print "Data for the post analysis written to %s" % fname_path


def write_raw_analysis_to_file(raw_data, analysis_type, observable,
                               post_analysis_folder, dryrun=False,
                               verbose=False):
    """
    Function that writes raw analysis data to file, either bootstrapped or
            jackknifed data.

    Args:
            raw_data: numpy float array, NFlows x NBoot, contains the data
                    from the analysis.
            analysis_type: string of the type of analysis we have performed,
                    e.g. bootstrap, jackknife, autocorrelation ect.
            observable: name of the observable
            post_analysis_folder: post analysis folder. Should be on the form 
                of {data_batch_folder}/{beta_folder}/post_analysis_data.
            dryrun: optional dryrun argument, for testing purposes
            verbose: a more verbose output
    """

    # Ensures that the post analysis data folder exists
    check_folder(post_analysis_folder, dryrun, verbose=verbose)

    # Checks that a folder exists for the statistical method output
    post_obs_folder = os.path.join(post_analysis_folder, analysis_type)
    check_folder(post_obs_folder, dryrun, verbose=verbose)

    # Creates file name
    file_name = observable
    file_name_path = os.path.join(post_obs_folder, file_name)

    # Stores data as binary output
    if not dryrun:
        np.save(file_name_path, raw_data)
    if verbose:
        print("Analysis %s for observable %s stored as binary data "
              "at %s.npy" % (analysis_type, observable, file_name_path))


def get_num_observables(batch_folder, beta_folder=None):
    """Gets the number of observable in a folder."""
    if isinstance(beta_folder, types.NoneType):
        flow_path = os.path.join(batch_folder, "flow_observables")
    else:
        flow_path = os.path.join(batch_folder, beta_folder, "flow_observables")
    num_obs = []

    # If flow path do not exist, then we return
    if not os.path.isdir(flow_path):
        return 0

    # Loops over flow obs folders
    for flow_obs in os.listdir(flow_path):
        # Skips all hidden files, e.g. .DS_Store
        if flow_obs.startswith("."):
            continue

        # Gets flow observable files, should be equal to number of observables
        flow_obs_path = os.path.join(flow_path, flow_obs)
        flow_obs_files = os.listdir(flow_obs_path)

        # In case observable folder is empty
        if len(flow_obs_files) == 0:
            continue

        # Removes all hidden files, e.g. .DS_Store
        flow_obs_files = [f for f in flow_obs_files if not f.startswith(".")]

        num_obs.append(len(flow_obs_files))

    assert not sum([i - num_obs[0] for i in num_obs]), \
        "number of flow observables in each flow observable differ"

    return num_obs[0]


if __name__ == '__main__':
    exit("Exiting: folderreadingtools not intended to be run as a standalone"
         " module.")
