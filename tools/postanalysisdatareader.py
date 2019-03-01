from folderreadingtools import check_folder
import os
import re
import numpy as np
import copy
import json
import types

__all__ = ["PostAnalysisDataReader", "getLatticeSpacing"]


class PostAnalysisDataReader:
    """
    Small class for reading post analysis data
    """

    def __init__(self, batch_parameters, observables_to_load=None,
                 verbose=False):
        """
        Class for loading the post analysis data.

        Args:
                batch_folders: list of strings of beta folder paths.
                observables_to_load: list of str, optional. Will only load those
                        specified.
                verbose: optional more verbose output. Default is False.
        """
        self._set_batch_name(batch_parameters)
        self._set_lattice_parameters(batch_parameters)

        self.verbose = verbose

        # Dictionary variable to hold all the data sorted by batches
        self.data_batches = {}

        # Dictionaries to hold the raw bootstrapped/jackknifed
        # data and autocorrelation data
        self.data_raw = {}

        # Binary folder types available
        self.analysis_types = []

        # Observable list
        self.observable_list = []

        # Observables to load, enables fast load times when only analysing few
        self.observables_to_load = [o.lower() for o in observables_to_load]

        # Number of betas variable
        self.beta_values = []

        # Reference Scale, must be set after initial loading if it to be used
        self.reference_values = None

        # Iterates over the different beta value folders
        for beta_folder in self.beta_folders:
            # Construct beta post analysis folder path
            beta_dir_path = os.path.join(beta_folder, "post_analysis_data")

            observables_data = {}
            obs_data_raw = {}

            _beta_values = []

            # Tries to load the parameter file
            try:
                param_file = os.path.join(beta_folder, "post_analysis_data",
                                          "params.json")
                self.param_beta = self._get_parameter_data(param_file)["beta"]
                _beta_values.append(self.param_beta)
            except IOError:
                print "No parameter file found."

            for obs in self._get_dir_content(beta_dir_path):
                obs_dir_path = os.path.join(beta_dir_path, obs)

                if os.path.splitext(obs_dir_path)[-1] == ".json":
                    continue

                if obs not in self.observable_list:
                    self.observable_list.append(obs)

                if not isinstance(self.observables_to_load, types.NoneType):
                    if not obs.lower() in self.observables_to_load:
                        self.observable_list.remove(obs)
                        if self.verbose:
                            print "Skipping loading observable %s" % obs
                        continue

                if self._check_is_content_dir(obs_dir_path):
                    # In case we have an observable that contains sub folders
                    # with the same observable but at different points in
                    # time or similiar.
                    obs_data = {}
                    sub_obs_raw = {}

                    for sub_obs in self._sort_folder_list(obs_dir_path):
                        sub_obs_path = os.path.join(obs_dir_path, sub_obs)

                        if self._check_is_content_dir(sub_obs_path):
                            # Checks if we have another nested folder, which
                            # is the case for the qtq0e quantity.
                            ss_obs_data = {}
                            ss_obs_raw = {}

                            for ss_obs in self._sort_folder_list(sub_obs_path):
                                ss_path = os.path.join(sub_obs_path, ss_obs)

                                # Retrieves sub-sub observable folder data
                                _data, _raw, _beta = self._get_obs_data(
                                    sub_obs, sub_obs_path, sub_obs=ss_obs,
                                    sub_obs_path=ss_path)

                                # Appends and populates dictionaries
                                _beta_values.append(_beta)
                                ss_obs_data[ss_obs] = _data
                                ss_obs_raw[ss_obs] = _raw

                            # Populates dictionaries
                            obs_data[sub_obs] = ss_obs_data
                            sub_obs_raw[sub_obs] = ss_obs_raw

                        else:
                            # Retrieves sub observable folder data
                            _data, _raw, _beta = self._get_obs_data(
                                obs, obs_dir_path, sub_obs=sub_obs,
                                sub_obs_path=sub_obs_path)

                            # Appends and populates dictionaries
                            obs_data[sub_obs] = _data
                            sub_obs_raw[sub_obs] = _raw
                            _beta_values.append(_beta)

                    # Places the retrieved sub-folder observables into
                    # dictionary.
                    obs_data_raw[obs] = sub_obs_raw
                else:
                    # In case of regular observables, that is no sub-folders.
                    _data, _raw, _beta = self._get_obs_data(
                        obs, obs_dir_path)

                    obs_data = _data
                    obs_data_raw[obs] = _raw

                observables_data[obs] = obs_data

            assert np.asarray(_beta_values).all(), "betas not equal."
            beta = _beta_values[0]
            self.beta_values.append(beta)

            # Stores batch data
            self.data_batches[beta] = copy.deepcopy(observables_data)

            # Stores the binary data
            self.data_raw[beta] = obs_data_raw

            # Frees memory
            del observables_data

        # Reorganizes data to more ease-of-use type of data set
        self._reorganize_data()
        self._reorganize_raw_data()

    def __call__(self, observable):
        """Returns observable with all relevant data."""
        return self.data_observables[observable]

    def __getitem__(self, observable):
        """Returns observable with all relevant data."""
        return self.data_observables[observable]

    def get_observables(self):
        """Returns list of observables."""
        return self.observable_list

    def set_reference_values(self, reference_values):
        """
        Sets the reference scale usually obtained from the energy.

        Args:
                reference_values: dict. 

                        Structure:
                        reference_values[extrapolation_method][analysis_type]
                                ["t0_cont", "t0_cont_err", beta]

                        Beta dictionary contains:
                                [beta]["t0", "t0err"]
        """
        self.reference_values = reference_values

    def _set_batch_name(self, batch_parameters):
        """Sets batch name and batch folder."""
        self.beta_folders = [os.path.join(b["batch_folder"], b["batch_name"])
                             for b in batch_parameters]

        # Splits all of the paths
        _folders = [os.path.split(b["batch_folder"]) for b in batch_parameters]

        # Gets the batch folder names
        _bfolders = set([b["batch_folder"] for b in batch_parameters])
        assert len(_bfolders) == 1, "Multiple batch folders detected: %s" \
            % ", ".join(list(_bfolders))

        # Gets the batch folder name
        _bnames = set([os.path.split(_f)[-1] for _f in _bfolders])
        assert len(_bnames) == 1, "Multiple batches detected: %s" \
            % ", ".join(list(_bnames))

        # Sets the final results
        self.batch_name = list(_bnames)[0]
        self.batch_folder = list(_bfolders)[0]

    def _set_lattice_parameters(self, batch_parameters):
        """Sets the correct labels, colors and lattice sizes."""
        self.lattice_sizes = {}
        self.lattice_volumes = {}
        self.N = {}
        self.NT = {}
        self.flow_epsilon = {}
        self.colors = {}
        self.labels = {}
        _print_latex_list = []
        for b in batch_parameters:
            self.N[b["beta"]] = b["N"]
            self.NT[b["beta"]] = b["NT"]
            self.lattice_sizes[b["beta"]] = [b["N"], b["NT"]]
            self.lattice_volumes[b["beta"]] = b["N"]**3*b["NT"]
            self.flow_epsilon[b["beta"]] = b["flow_epsilon"]
            self.colors[b["beta"]] = b["color"]
            self.labels[b["beta"]] = r"$%d^3 \times %d$" % (b["N"], b["NT"])
            _print_latex_list.append(b["print_latex"])

        assert len(set(_print_latex_list)), (
            "check input parameters: print_latex differs for parameters.")

        self.print_latex = _print_latex_list[0]

    def _get_obs_data(self, obs, obs_path, sub_obs=None, sub_obs_path=None):
        """Method for retrieving data associated with an observable."""
        if sub_obs == None:
            sub_obs = obs
        if sub_obs_path == None:
            sub_obs_path = obs_path

        # Retrieves folder file lists
        binary_data_folders, observable_files = self._get_obs_dir_paths(
            sub_obs, sub_obs_path)

        # Retrieve observable data and beta
        _beta, sub_obs_data = self._get_obs_dict(obs, observable_files)

        # Retrieves the raw binary data into dictionary
        analysis_raw = {}
        for binary_analysis_folder in binary_data_folders:
            analysis_type = os.path.basename(binary_analysis_folder)
            analysis_raw[analysis_type] = self._get_bin(binary_analysis_folder)
            self._add_analysis(analysis_type)

        return sub_obs_data, analysis_raw, _beta

    def _check_is_content_dir(self, folder):
        """Returns True if all of the contents are folders."""
        for f in self._get_dir_content(folder):
            if f.startswith("."):  # If we have a .DS_Store file or similar
                continue
            if not os.path.isdir(os.path.join(folder, f)):
                return False
        else:
            return True

    def _add_analysis(self, atype):
        """Adds new analysis if we don't have it."""
        if atype not in self.analysis_types:
            self.analysis_types.append(atype)

    def _sort_folder_list(self, folder_path):
        """
        Sorts the folders depending on if we have intervals or specific points.
        """

        folders = self._get_dir_content(folder_path)

        if len(folders) == 0:
            return []

        if "-" in folders[0]:  # For MC and euclidean intervals
            def sort_key(s): return int(s.split("-")[-1])
        else:
            def alphanum(s): return re.findall('(\d+)', s)[0]

            def sort_key(s): return int(alphanum(s))

        return sorted(folders, key=sort_key)

    def _get_obs_dir_paths(self, obs, obs_folder):
        """
        Internal method for retrieving all of the observable data within a 
        folder.
        """

        # Sorts into two lists, one with .txt extensions, for retrieving the
        # beta value. The other for retrieving the analysis types.
        binary_data_folders = []
        observable_files = []

        # Function for checking if we have a binary data folder
        def _chk_obs_f(f, fp): return os.path.isdir(
            fp) and not f.startswith(".")

        for _f in self._get_dir_content(obs_folder):
            # Temporary path to observable sub folder
            _f_path = os.path.join(obs_folder, _f)

            # If we have a folder containing binary raw data
            if _chk_obs_f(_f, _f_path):
                binary_data_folders.append(_f_path)
            else:
                observable_files.append(_f_path)

        return binary_data_folders, observable_files

    def _get_obs_dict(self, obs, observable_files):
        """
        Gets the observable data dictionaries.
        """
        observable_data = {}

        # Temporary beta list for cross checking
        _beta_values = []

        # Gets the analyzed data for each observable
        for obs_file_path in observable_files:
            # Retrieves the observable data
            if "no_autocorr" in obs_file_path:
                observable_data["without_autocorr"] = self._get_obs_data_dict(
                    obs_file_path, autocorr=False)

                _beta_values.append(
                    observable_data["without_autocorr"]["beta"])

            else:
                observable_data["with_autocorr"] = self._get_obs_data_dict(
                    obs_file_path, autocorr=True)

                _beta_values.append(observable_data["with_autocorr"]["beta"])

        assert np.asarray(_beta_values).all(), "betas not equal."

        return _beta_values[0], observable_data

    def _get_obs_data_dict(self, observable_file, autocorr=False):
        """
        Internal function for retrieving observable data.

        Args:
            observable_file: string file path of a .txt-file containing
                relevant information

        Returns:
            obs_data: dictionary with y, y_error, beta, unanalyzed, bootstrap,
                jackknife and optional tau_int, tau_int_err, sqrt2tau_int.
        """

        # Retrieves meta data
        # Make it so one can retrieve the key as meta_data[i] and then value as
        # meta_data[i+1]
        if os.path.splitext(observable_file)[-1] == ".txt":
            meta_data = self._get_meta_data(observable_file)

            # Loads data into temporary holder
            retrieved_data = np.loadtxt(observable_file)
        else:
            meta_data = {"beta": self.param_beta}

            # Loads data into temporary holder
            retrieved_data = np.load(observable_file)

        # Temporary methods for getting observable name and beta value, as this
        # will be put into the meta data
        obs = os.path.split(os.path.splitext(observable_file)[0])[-1]

        # Dictionary to store all observable data in
        obs_data = {}

        # Puts data into temporary holding facilities
        t = retrieved_data[:, 0]
        y = retrieved_data[:, 1]
        y_error = retrieved_data[:, 2]
        bs_y = retrieved_data[:, 3]
        bs_y_error = retrieved_data[:, 4]
        jk_y = retrieved_data[:, 5]
        jk_y_error = retrieved_data[:, 6]

        # Stores data into dictionaries
        unanalyzed_data = {"y": y, "y_error": y_error, "x": t}
        bs_data = {"y": bs_y, "y_error": bs_y_error, "x": t}
        jk_data = {"y": jk_y, "y_error": jk_y_error, "x": t}

        # Stores flow time

        # Stores observable data
        obs_data["beta"] = copy.deepcopy(meta_data["beta"])
        obs_data["unanalyzed"] = copy.deepcopy(unanalyzed_data)
        obs_data["bootstrap"] = copy.deepcopy(bs_data)
        obs_data["jackknife"] = copy.deepcopy(jk_data)
        # obs_data["flow_time"]     = copy.deepcopy(t)

        if autocorr:
            tau_int = retrieved_data[:, 7]
            tau_int_err = retrieved_data[:, 8]
            sqrt2tau_int = retrieved_data[:, 9]
            ac_data = {
                "tau_int": tau_int,
                "tau_int_err": tau_int_err,
                "sqrt2tau_int": sqrt2tau_int
            }
            obs_data["autocorr"] = copy.deepcopy(ac_data)

        if self.verbose:
            print "Data retrieved from %s" % observable_file

        # Frees memory
        del retrieved_data

        return obs_data

    def _get_bin(self, folder):
        """Gets binary data."""
        assert len(os.listdir(folder)) == 1, (
            "unrecognized/multiple files in binary folder %s: %s" % (
                folder, ", ".join(os.listdir(folder))))
        return np.load(os.path.join(folder, self._get_dir_content(folder)[0]))

    @staticmethod
    def _get_parameter_data(file):
        # Retrieves meta data from header or file
        return_dict = {}
        with open(file, "r") as f:
            return_dict = json.load(f)
        return return_dict

    @staticmethod
    def _get_meta_data(file):
        """Retrieves meta data from header or file."""
        meta_data = {}
        with open(file) as f:
            header_content = f.readline().split(" ")[1:]
            meta_data[header_content[0]] = header_content[1]
            meta_data[header_content[2]] = \
                float(header_content[3].replace("_", "."))
        return meta_data

    def _reorganize_data(self):
        """Reorganizes the data into beta-values and observables sorting."""
        self.data_observables = {}

        # Sets up new dictionaries by looping over batch names
        for beta in self.data_batches:
            # Loops over observable names
            for observable_name in self.data_batches[beta]:
                # Creates new sub-dictionary ordered by the observable name
                self.data_observables[observable_name] = {}

        # Places data into dictionaries
        for beta in self.data_batches:
            # Loops over the batch observable
            for observable_name in self.data_batches[beta]:
                # Stores the batch data in a sub-dictionary
                self.data_observables[observable_name][beta] = \
                    self.data_batches[beta][observable_name]

    def _reorganize_raw_data(self):
        """Reorganizes the data into beta-values and observables sorting."""
        self.raw_analysis = \
            {analysis_type: {} for analysis_type in self.analysis_types}

        # Sets up new beta value dictionaries
        for beta in self.data_raw:
            # self.raw_analysis[beta] = {}

            # Loops over observable names and sets up dicts
            for observable_name in self.data_raw[beta]:
                # self.raw_analysis[beta][observable_name] = {}

                for sub_elem in self.data_raw[beta][observable_name]:

                    if not sub_elem in self.analysis_types:
                        for analysis_type in self.analysis_types:
                            self._check_raw_bin_dict_keys(
                                analysis_type, beta, observable_name, sub_elem)
                    else:
                        self._check_raw_bin_dict_keys(
                            sub_elem, beta, observable_name)

        # Populates dictionaries
        for beta in self.data_raw:
            # Loops over observable names
            for observable_name in self.data_raw[beta]:
                # Loops over analysis types contained in observable name,
                # unless it is a split observable
                for sub_elem in self.data_raw[beta][observable_name]:
                    if sub_elem in self.analysis_types:
                        atype = sub_elem
                        self.raw_analysis[atype][beta][observable_name] = \
                            self.data_raw[beta][observable_name][atype]

                    else:
                        self._reorganize_raw_sub_sub(beta,
                                                     sub_elem, observable_name)

    def _reorganize_raw_sub_sub(self, beta, sub_elem, obs_name):
        """Internal method for re-organizing the raw data."""

        # Loops over sub-sub element
        for ss in self.data_raw[beta][obs_name][sub_elem]:

            # Checks if we have an raw analysis dictionary
            if ss in self.analysis_types:
                atype = ss
                self.raw_analysis[atype][beta][obs_name][sub_elem] = \
                    self.data_raw[beta][obs_name][sub_elem][atype]

            else:
                for atype in self.data_raw[beta][obs_name][sub_elem][ss]:
                    self.raw_analysis[atype][beta][obs_name][sub_elem][ss] = \
                        self.data_raw[beta][obs_name][sub_elem][ss][atype]

    def _check_raw_bin_dict_keys(self, analysis_type, beta, observable_name,
                                 sub_obs=None):
        """
        Internal method for setting up dictionaries in case they do not exist.
        """

        if beta not in self.raw_analysis[analysis_type]:
            self.raw_analysis[analysis_type][beta] = {}

        if observable_name not in self.raw_analysis[analysis_type][beta]:
            self.raw_analysis[analysis_type][beta][observable_name] = {}

        if sub_obs != None:
            if sub_obs not in \
                    self.raw_analysis[analysis_type][beta][observable_name]:

                self.raw_analysis[analysis_type][
                    beta][observable_name][sub_obs] = {}

    def _retrieve_sub_sub(self, obs_dir_path):
        """
        Internal method for retrieving observable qtq0e, as that it contains 
        nested folders.
        """

        print "subsub is used!"

        # print obs_dir_path
        for flow_folder in self._get_dir_content(obs_dir_path):
            flow_folder_path = os.path.join(obs_dir_path, flow_folder)

            for eucl_folder in self._get_dir_content(flow_folder_path):
                eucl_folder_path = os.path.join(flow_folder_path,
                                                eucl_folder)

            exit(1)
        raise NotImplementedError("qtq0e not completely implemented")

    @staticmethod
    def _get_dir_content(folder):
        if not os.path.isdir(folder):
            raise IOError("No folder by the name %s found." % folder)
        else:
            return [f for f in os.listdir(folder) if not f.startswith(".")]


if __name__ == '__main__':
    import sys
    sys.exit("Error: PostAnalysisDataReader is not a standalone program.")
