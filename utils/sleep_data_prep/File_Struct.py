from utils.sleep_data_prep.edf_reader import EDF_Reader
from utils.sleep_data_prep.mne_reader import MNE_Reader
from utils.sleep_data_prep.sleep_data_utils import load_pyedf, rearrange_headers, resample, create_windows
from pathlib import Path
import os
import csv
import re
import sys
import json
import shutil
import numpy as np
import time
import pickle
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy import signal as sg
import math
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
from os import path
import zarr
from collections.abc import Iterable
from scipy.io import loadmat, savemat
import warnings


class File_Struct():
    def __init__(self, config):
        self.config = config
        self.root = config.data_root
        self.ending_data = self.config.ending_data
        self.ending_label = self.config.ending_label

        self.doctor = self._Node(data_file=self.root, name = "doctor", parent= None)
        # This dict includes the mapping between object nodes' name and the data folders
        self.doctor.names_dict = {}
        self._create_patients(self._load_dirs(self.root))

    # def _finding_data_ending(self, patient_name):
    #     for ending in self.ending_data:
    #         if ending in patient_name:
    #             patient_name = patient_name.split(ending)[0]
    #             right_ending = ending
    #             return right_ending
    #     return False
    def _load_dirs(self, filename):
        """
        Loads only the directories of the path self.root

        :return: List of paths
        """
        list_of_dirs = []
        for fname in os.listdir(filename):
            path = os.path.join(filename, fname)
            if os.path.isdir(path):
                new_list = self._load_dirs(path)
                new_list_r = [fname+"/"+i for i in new_list]
                if new_list:
                    list_of_dirs += new_list_r
            else:
                list_of_dirs.append(fname)
        return list_of_dirs

    def _create_patients(self, dir_patients):
        """
        Creates a child on the central node for every patient (file in the root)
        and a child on the patient node for every file with the data and label path.

        :param dir_patients: List of directories, each one is a patient.
        :return: Objects reporting the file system under the central node.
        """
        k=0

        # Get the names of all the patients in the dataset. Each dataset requires a new way to do it,
        # according to the chosen file organization
        patient_files = []
        for patient_name in dir_patients:
            for ending in self.ending_data:
                if ending in patient_name:
                    # SHHS1
                    if self.config.dataset_type == "shhs":
                        patient_files.append(patient_name.split(".")[0])
                    elif self.config.dataset_type == "nch" or self.config.dataset_type == "nch_total":
                        patient_files.append(patient_name.split(".")[0])
                    # SeizIT1
                    elif self.config.dataset_type == "seizit1":
                        actual_name = patient_name.split("/")[0]
                        if actual_name in self.config.train_patients or  actual_name in self.config.test_patients:
                            patient_files.append(actual_name)
                    # SLEEP_EDF
                    elif self.config.dataset_type == "sleep_edf":
                        patient_files.append(patient_name.split("/")[0])
                    # Neonatal
                    elif self.config.dataset_type == "neonatal":
                        patient_files.append(patient_name.split("/")[0])


        patient_files = list(dict.fromkeys(patient_files))



        #Add each patient and its files as children to our top node the doctor.
        files_num = 0
        for patient_name in patient_files:

            self.doctor.names_dict[patient_name] = "patient_" + str(f'{k:04}')
            self.doctor.add_child("patient_" + str(f'{k:04}'), self._Node(parent=self.doctor, data_file =self.root+"/"+patient_name,name=patient_name ))
            patient = getattr(self.doctor, "patient_" + str(f'{k:04}'))
            patient.name = patient_name
            filenames_data, filenames_label = [], []
            for i in dir_patients:
                if patient_name in i:
                    flag_data = False
                    flag_label = False
                    for m in self.ending_data:
                        if m in i:
                            flag_data = True
                    for m in self.ending_label:
                        if m in i:
                            flag_label = True
                    if flag_data:
                        filenames_data.append(self.root+"/"+i)
                    if flag_label:
                        filenames_label.append(self.root+"/"+i)

            filenames_data.sort()
            if self.config.dataset_type == "seizit1":
                #Eliminate some of the label filenames, since we have more than one labeler
                filenames_label = [s for s in filenames_label if "a1" in s]
            filenames_label.sort()
            if filenames_data or filenames_label:
                if len(filenames_data) != len(filenames_label) or len(filenames_label) == 0:
                    try:
                        filenames_data.remove(patient_name+".edf")
                        filenames_label.remove(patient_name+".tsv")
                    except:
                        pass
                    warnings.warn("Patient {} has different number of data and label files".format(patient_name))
                    continue
            print(filenames_data)
            for i, files in enumerate(filenames_data):

                patient.add_child("file_" + str(f'{patient.num_children:02}'), self._Node(parent=self.doctor, data_file=files, name= "file_" + str(f'{patient.num_children:02}') ))
                file_obj = getattr(patient, "file_" + str(f'{patient.num_children-1:02}'))
                file_obj.label_file = filenames_label[i] #self._file_pair(files, filenames_label[i])
            files_num += len(filenames_data)
            k+=1

        print("Total patient number: {} and file number {}".format(len(patient_files),files_num))

    def _create_dirs(self):
        inv_dict = {v: k for k, v in self.doctor.names_dict.items()}
        if os.path.isfile(self.config.save_dir + '/patient_map.json'):
            with open(self.config.save_dir + '/patient_map.json', 'w+') as fp:
                json.dump(inv_dict, fp)
            fp.close()
        else:
            self._create_n_save_patient_map()
            print("We dont have a patient map")

    def _create_n_save_patient_map(self):
        patient_map = {}
        for patient_num in range(self.doctor.num_children):
            patient = getattr(self.doctor, "patient_{}".format(f'{patient_num:04}'))
            patient_map["patient_{}".format(f'{patient_num:04}')] = patient.name

        try:
            patient_map_file = open(self.config.save_dir + '/patient_map.pkl', "wb")
            self.patient_map = patient_map
            pickle.dump(patient_map, patient_map_file)
            patient_map_file.close()
            print("Patient map created and saved on {}".format(self.config.save_dir + '/patient_map.pkl'))
        except Exception as e:
            print(e)
            raise RuntimeError("Error on saving patient map")

    def _merge_two_dicts(self, x, y):
        z = x.copy()
        z.update(y)
        return z

    # def _file_pair(self, data_file, filenames_label):
    #     """
    #     Find for the data_file the corresponding label file. Raise exception in case none or more than one fit.
    #
    #     :param data_file: Data file path
    #     :param filenames_label: List of all label files in the folder
    #     :return: Path to the label
    #     """
    #     cmn_phrase = data_file.split(".")[0]
    #     count = 0
    #     for i in filenames_label:
    #         x = re.search(cmn_phrase, i)
    #         if x:
    #             count +=1
    #             label_file = i
    #     if count == 1:
    #         return label_file
    #     else:
    #         raise Exception("We cannot find or there are more than one valid label files for {}".format(data_file))

    def _load_files(self, root, ending):
        """
        Find the available data or label paths.

        :param root: The full path of the data we want to load
        :param ending: The .sth of the data files
        :return: List of paths
        """
        file_paths = [os.path.join(root, file) for root, _, files in os.walk(root) for file in files if
                      file.endswith(ending)]
        return file_paths

    def _load_metrics(self):
        """
        Metrics_dir needs to be a pickle file.
        Format of the metrics is as follows:
            metrics[part][type][modalities][channels] where
                            part = train (necessary) or test (optional)
                            type = Depending on whether you are using time signals or stft we got
                                    time -> "mean_time" and "std_time" are expected
                                    stft -> "mean_stft" and "std_stft" are expected
                            modalities = int index of the modality
                            channels = int index of the channel of each modality

            For type="time" each metrics[part][type][modalities][channels] will be float
            For type="stft" each metrics[part][type][modalities][channels] will be np.array(dtype="float").shape = STFT feature bins

        :return: metrics object
        """
        try:
            metrics_file = open(self.config.metrics_dir, "rb")
            metrics = pickle.load(metrics_file)
            print("Metrics loaded from {}".format(self.config.metrics_dir))
            return metrics
        except Exception as e:
            print(e)
            raise RuntimeError("Error on loading metrics")

    def _save_metrics(self, metrics):
        """
        Metrics_dir needs to be a pickle file.
        Format of the metrics should be as follows:
            metrics[part][type][modalities][channels] where
                            part = train (necessary) or test (optional)
                            type = Depending on whether you are using time signals or stft we got
                                    time -> "mean_time" and "std_time" are expected
                                    stft -> "mean_stft" and "std_stft" are expected
                            modalities = int index of the modality
                            channels = int index of the channel of each modality

            For type="time" each metrics[part][type][modalities][channels] will be float
            For type="stft" each metrics[part][type][modalities][channels] will be np.array(dtype="float").shape = STFT feature bins
        """
        try:
            metrics_file = open(self.config.metrics_dir, "wb")
            pickle.dump(metrics, metrics_file)
            metrics_file.close()
            print("Metrics saved on {}".format(self.config.metrics_dir))
        except Exception as e:
            print(e)
            raise RuntimeError("Error on saving metrics")

    def _get_mean_std(self):

        if self.config.normalization == "all":
            norm_axis = None
        elif self.config.normalization == "freq_bin":
            norm_axis = 1

        metrics = self._initialize_metrics()
        if self.config.parallel:
            num_cores = multiprocessing.cpu_count()
            metrics_scrumble = Parallel(n_jobs=num_cores)(delayed(self.parallel_file_mean_std)(patient_num, metrics, norm_axis) for patient_num in tqdm(range(self.doctor.num_children), "Mean-STD Calculating "))
            metrics = self.gather_results(metrics_scrumble, metrics)

        else:
            for patient_num in tqdm(range(12,14)):
                self._parallel_file_mean_std(patient_num, metrics, norm_axis)

        metrics = self._calculate_weighted_means(metrics)
        return metrics

    def _get_metrics(self):
        if self.config.load_metrics == True:
            metrics = self._load_metrics()
        elif self.config.load_metrics == "after":
            metrics = self._initialize_metrics(make_none=True)
            print("Not normalizing data")
        else:
            metrics = self._get_mean_std()
            self._save_metrics(metrics)
        return metrics

    def _get_file_save_dir(self, real_p_name, patient_num, file_num):
        """
        This function is used to get the proper savedir if we want to split our data initially on train and test set.
        It is mainly used to avoid touching test set on data preparation.

        :param real_p_name: string real patient name (according to the edf file name)
        :param patient_num: int corresponding patient number
        :param file_num: int corresponding file number
        :return: string root_save_dir/train or test/patient_{padded 4 patient_num}/file_{padded 2 file_num}
        """

        # Get the file_save_dir
        if real_p_name in self.config.test_patients:
            file_save_dir = self.config.save_dir + "/test" + "/patient_" + str(
                f'{patient_num:04}') + "/" + "file_" + str(f'{file_num:02}')
        else:
            file_save_dir = self.config.save_dir + "/train" + "/patient_" + str(
                f'{patient_num:04}') + "/" + "file_" + str(f'{file_num:02}')
        # We dont use an assert here if it doesnt belong to either test or train set,
        # because we want to be flexible on not defining every training set.

        return file_save_dir

    def _initialize_metrics(self, make_none=False):
        metrics = {}
        metrics_list = ["mean_time","mean_sq_time", "std_time", "sum_time", "mean_stft", "mean_sq_stft", "std_stft", "sum_stft"]
        file_saving_list = ["files_time","files_stft"]
        for set in ["train", "test"]:
            metrics[set] = {}
            for m in file_saving_list:
                metrics[set][m] = {}
                if make_none:
                    metrics[set][m] = None
                    continue
                for mod in self.config.modality_end:
                    metrics[set][m][mod] = []

            for m in metrics_list:
                metrics[set][m] = {}
                if make_none:
                    metrics[set][m] = None
                    continue
                for mod in self.config.modality_end:
                    metrics[set][m][mod] = {}
                    for ch in range(self.config.ch_per_mod[mod]):
                        metrics[set][m][mod][ch] = np.zeros(129) if "stft" in m and "sum" not in m else 0

        return metrics

    def _calculate_weighted_means(self, metrics):
        for part in ["train", "test"]:
            for key in ["mean_time","std_time"]:
                for mod in range(len(self.config.modality_end)):
                    for ch in range(len(metrics[part][key][mod].keys())):
                        if  isinstance(metrics[part][key][mod][ch],(np.ndarray, np.generic)) and metrics[part]["sum_time"][mod][ch] != 0 :
                            metrics[part][key][mod][ch] = np.array(metrics[part][key][mod][ch])/metrics[part]["sum_time"][mod][ch]
            for key in ["mean_stft", "std_stft"]:
                for mod in range(len(self.config.modality_end)):
                    for ch in range(len(metrics[part][key][mod].keys())):
                        if isinstance(metrics[part][key][mod][ch],(np.ndarray, np.generic))  and metrics[part]["sum_stft"][mod][ch] != 0 :
                            metrics[part][key][mod][ch] = np.array(metrics[part][key][mod][ch])/metrics[part]["sum_stft"][mod][ch]
        return metrics

    def _gather_metrics(self, metrics_scramble):
        metrics = self._initialize_metrics()
        for metrics_p in metrics_scramble:
            if metrics_p == []:
                continue
            metrics_p = metrics_p[0]
            real_p_name = metrics_p["patient_name"]
            part = "test" if real_p_name in self.config.test_patients else "train"

            for mod in metrics_p["sum_stft"].keys():
                for ch in metrics_p["sum_stft"][mod].keys():
                    # Only numbers represent channels
                    if isinstance(ch,(int,float)):
                        if metrics[part]["sum_stft"][mod][ch] == 0:
                            for key in ["mean_stft", "mean_sq_stft"]:
                                metrics[part][key][mod][ch] += metrics_p[key][mod][ch]
                            metrics[part]["sum_stft"][mod][ch] += metrics_p["sum_stft"][mod][ch]
                        else:
                            for key in ["mean_stft", "mean_sq_stft"]:
                                metrics[part][key][mod][ch] = (metrics_p[key][mod][ch] * metrics_p["sum_stft"][mod][ch] + metrics[part][key][mod][ch] * metrics[part]["sum_stft"][mod][ch]) / (metrics_p["sum_stft"][mod][ch] + metrics[part]["sum_stft"][mod][ch])
                            metrics[part]["sum_stft"][mod][ch] += metrics_p["sum_stft"][mod][ch]


        for part in ["train", "test"]:
            for mod in metrics[part]["sum_stft"].keys():
                for ch in metrics_p["sum_stft"][mod].keys():

                    varX = -np.multiply(metrics[part]["mean_time"][mod][ch], metrics[part]["mean_time"][mod][ch]) + metrics[part]["mean_sq_time"][mod][ch]
                    metrics[part]["std_time"][mod][ch] = np.sqrt(varX*metrics[part]["sum_time"][mod][ch]/(metrics[part]["sum_time"][mod][ch]-1)) if metrics[part]["sum_time"][mod][ch]!=0 else 0
                    metrics[part]["mean_time"][mod][ch] = metrics[part]["mean_time"][mod][ch]/metrics[part]["sum_time"][mod][ch] if metrics[part]["sum_time"][mod][ch]!=0 else 0
                    metrics[part]["mean_sq_time"][mod][ch] = metrics[part]["mean_sq_time"][mod][ch]/metrics[part]["sum_time"][mod][ch] if metrics[part]["sum_time"][mod][ch]!=0 else 0

                    varX = -np.multiply(metrics[part]["mean_stft"][mod][ch], metrics[part]["mean_stft"][mod][ch]) + metrics[part]["mean_sq_stft"][mod][ch]
                    metrics[part]["std_stft"][mod][ch] = np.sqrt(varX*metrics[part]["sum_stft"][mod][ch]/(metrics[part]["sum_stft"][mod][ch]-1)) if metrics[part]["sum_stft"][mod][ch]!=0 else 0
                    metrics[part]["mean_stft"][mod][ch] = metrics[part]["mean_stft"][mod][ch]/metrics[part]["sum_stft"][mod][ch] if metrics[part]["sum_stft"][mod][ch]!=0 else 0
                    metrics[part]["mean_sq_stft"][mod][ch] = metrics[part]["mean_sq_stft"][mod][ch]/metrics[part]["sum_stft"][mod][ch] if metrics[part]["sum_stft"][mod][ch]!=0 else 0


        return metrics

    def _gather_filenames(self, metrics_scramble, types):

        """
        Gather and save the filenames of the data and the windows each one contains. We save for each part = "train" or "test" for each type (stft, time) and for each modality.
        Save them based on config.name_dict on a *_file_map.txt file.

        :param metrics_scramble: list of metrics (that include filenames) from each patient separately.
        :param types: list of types, for example ["time", "stft"]
        :return:
        """

        metrics = self._initialize_metrics()
        #Gather filenames and number of windows from the different patients.
        for metrics_p in metrics_scramble:
            if metrics_p == []:
                continue
            for metric_f in metrics_p:
                part = "test" if metric_f["patient_name"] in self.config.test_patients else "train"
                for type in types:
                    for mod in metric_f["files_{}".format(type)].keys():
                        for file in metric_f["files_{}".format(type)][mod]:
                            metrics[part]["files_{}".format(type)][mod].append(file)

        #Save filenames and number of windows.
        for part in ["train", "test"]:
            for type in types:
                for mod in metrics[part]["files_{}".format(type)].keys():
                    if len(metrics[part]["files_{}".format(type)][mod]) == 0 : continue
                    total_windows = np.array([i[1] for i in metrics[part]["files_{}".format(type)][mod]]).sum()
                    _, patient_counts = np.unique(np.array([i[0] for i in metrics[part]["files_{}".format(type)][mod]]), return_counts=True)
                    print("{}-{}-{} has {} patients with {} windows".format(part, type, self.config.name_dict[str(mod)][type], patient_counts.sum(), total_windows))
                    with open(self.config.save_dir + "/" + part + '/{}_file_map.txt'.format(self.config.name_dict[str(mod)][type]), 'w+') as fp:
                        for file in metrics[part]["files_{}".format(type)][mod]:
                            len_file = file[1]
                            fp.write("{}-{}\n".format(file[0],len_file))

    def _parallel_file_mean_std(self, patient_num, metrics, norm_axis=1):
        patient = getattr(self.doctor, "patient_{}".format(f'{patient_num:04}'))
        real_p_name = patient.name
        # print("{}_{}".format(real_p_name,patient.num_children))
        for file_num in range(patient.num_children):
            file = getattr(patient, "file_" + str(f'{file_num:02}'))
            try:
                if not hasattr(file, 'reader'):
                    setattr(file, 'reader', EDF_Reader(edf_file=file.data_file, name="file_" + str(f'{file_num:02}')))
                for type in self.config.types:
                    file.reader.check_edf_labels(self.config.headers_file, type)
                if not file.reader.verify_edf_channels(self.config.required_channels):
                    print("Missing Channel!")
                file.reader_enabled = True
            except Warning as err:
                print("In {}, {}".format(file.data_file, err))
                file.reader_enabled = False
            except OSError as err:
                print("In {}, {}".format(file.data_file, err))
                file.reader_enabled = False

            if file.reader_enabled:
                file.reader.load_signals()
                if self.config.types[0] == "seizit1":
                    file.reader.longtitudinal_montage(self.config.montage)
                file.reader.extract_labels(file.label_file, self.config.label_dict)
                file.reader.discard_parts(patient_num=patient_num)
                file.reader.filter_signals(self.config.filter_type, self.config.filter_order, self.config.filter_cutoff)
                file.reader.resample(self.config.sampling_rate)

                if file.parent.name in self.config.test_patients:
                    if not self.config.metrics_on_test:
                        # np.array([file.reader.signals[modality][ch] for modality in range(file.reader.signals["modalities"]) for ch in range(file.reader.signals[modality]["num_channels"])])
                        metrics["mean_test"].append(np.array([file.reader.signals[modality][ch].mean() * len(file.reader.signals[modality][ch]) for modality in
                                                              range(file.reader.signals["modalities"]) for ch in
                                                              range(file.reader.signals[modality]["num_channels"])]))
                        metrics["std_test"].append(np.array([file.reader.signals[modality][ch].std() * len(file.reader.signals[modality][ch]) for modality in
                                                             range(file.reader.signals["modalities"]) for ch in
                                                             range(file.reader.signals[modality]["num_channels"])]))
                        mean_test_stft, std_test_stft = [], []
                        for modality in range(file.reader.signals["modalities"]):
                            for ch in range(file.reader.signals[modality]["num_channels"]):
                                _, _, Zxx = sg.stft(file.reader.signals[modality][ch],
                                                    file.reader.signals[modality]["sampling_rate"], nperseg=200,
                                                    noverlap=100, nfft=256)
                                Zxx = 20 * np.log10(np.abs(Zxx))
                                mean_test_stft.append(Zxx.mean(axis=norm_axis))
                                std_test_stft.append(Zxx.std(axis=norm_axis))
                        metrics["mean_test_stft"].append(np.array(mean_test_stft))
                        metrics["std_test_stft"].append(np.array(std_test_stft))
                    else:
                        # These are added to avoid any additional checks below, they do not have a particular meaning
                        metrics["mean_test_stft"].append(np.array([0, 0, 0]))
                        metrics["std_test_stft"].append(np.array([0, 0, 0]))
                else:
                    for modality in range(file.reader.signals["modalities"]):
                        for ch in range(file.reader.signals[modality]["num_channels"]):
                            metrics["mean_train"][modality][ch] += [file.reader.signals[modality][ch].mean()*len(file.reader.signals[modality][ch])]
                            metrics["std_train"][modality][ch] += [file.reader.signals[modality][ch].std()*len(file.reader.signals[modality][ch])]
                            metrics["sum_train"][modality][ch] += len(file.reader.signals[modality][ch])
                            _, _, Zxx = sg.stft(file.reader.signals[modality][ch],
                                                file.reader.signals[modality]["sampling_rate"],
                                                nperseg=2 * file.reader.signals[modality]["sampling_rate"],
                                                noverlap=file.reader.signals[modality]["sampling_rate"],
                                                nfft=self.config.nfft_points)
                            Zxx = np.abs(Zxx)
                            Zxx[Zxx == 0] = 1e-5
                            Zxx = 20 * np.log10(Zxx)
                            if (Zxx != Zxx).any():
                                print(Zxx.min())
                                print(np.isnan(Zxx).any())
                            metrics["mean_train_stft"][modality][ch].append(Zxx.mean(axis=norm_axis)*Zxx.shape[norm_axis])
                            metrics["std_train_stft"][modality][ch].append(Zxx.std(axis=norm_axis)*Zxx.shape[norm_axis])
                            metrics["sum_train_stft"][modality][ch] += Zxx.shape[norm_axis]
                file.reader.sign_mem_free()
                file.reader.close()
                del file.reader
        return metrics

    def _parallel_file_processing(self, patient_num, patient, metrics, types):

        metrics_for_files = []
        for file_num in range(patient.num_children):
            file = getattr(patient, "file_" + str(f'{file_num:02}'))
            # file_save_dir = self._get_file_save_dir(real_p_name=patient.name, patient_num=patient_num, file_num=file_num)
            if ("use_library" in self.config and self.config.use_library == "edf_reader") or "use_library" not in self.config:
                setattr(file, 'reader', EDF_Reader(config=self.config, edf_file=file.data_file, name=patient.name,
                                                       label_file=file.label_file, patient_num=patient_num,
                                                       file_num=file_num, mod_names=self.config.modality_end))

            elif "use_library" in self.config and self.config.use_library == "mne_reader":
                setattr(file, 'reader', MNE_Reader(config=self.config, edf_file=file.data_file, name=patient.name,
                                                       label_file=file.label_file, patient_num=patient_num,
                                                       file_num=file_num, mod_names=self.config.modality_end))

            file.reader.check_edf_labels(self.config.headers_file, self.config.dataset_type)
            if not file.reader.verify_edf_channels(self.config.required_channels): assert "Missing Channel!"
            file.reader.load_signals()
            file.reader.extract_transform_labels()
            if not file.reader.discard_parts():
                warnings.warn("Patient {} file {} has been totally discarded".format(patient_num, file_num))
                continue

            file.reader.additional_dataset_operations()
            file.reader.resample(self.config.sampling_rate)
            file.reader.filter_signals(self.config.filter_type, self.config.filter_order, self.config.filter_cutoff)
            file.metrics = {}
            if "time" in types:

                time_in_minutes, file_metrics_time = file.reader.create_windows(
                                                                     enhancement=self.config.enhancement,
                                                                     mean=metrics['train']["mean_time"], std=metrics['train']["std_time"])
                file_metrics_time = file.reader.save_windows(view="time", metrics=file_metrics_time)
                file.metrics = self._merge_two_dicts(file.metrics, file_metrics_time)
                file.reader.plot_data("time", mean=metrics['train']["mean_stft"], std=metrics['train']["std_stft"])
                st = "Patient {}/{} file {}/{} ".format(patient_num+1, self.doctor.num_children, file_num+1, patient.num_children)
                for i in range(self.config.num_classes): st = st + "Label {}: {} ".format(i, int(file_metrics_time["count_labels"][i]))
                st = st + "Class Borders: {} ".format(int(file_metrics_time["count_labels"][-1]))
                st = st + "Total: {} ".format(int(file_metrics_time["count_labels"][:-1].sum()))

            if "stft" in types:
                file_metrics_stft = file.reader.create_sfft_windows(
                                                enhancement=self.config.enhancement,
                                                mean=metrics['train']["mean_stft"], std=metrics['train']["std_stft"])
                file_metrics_stft = file.reader.save_windows(view="stft", metrics=file_metrics_stft)
                file.metrics = self._merge_two_dicts(file.metrics, file_metrics_stft)
                file.reader.plot_data("stft", mean=metrics['train']["mean_stft"], std=metrics['train']["std_stft"])
            print(st)
            file.metrics["patient_name"] = patient.name
            metrics_for_files.append(file.metrics)
            file.reader.sign_mem_free()
            # file.reader.close()
        return metrics_for_files

    def process_patients(self,types):

        self._create_dirs()
        metrics = self._get_metrics()

        if self.config.parallel:
            num_cores = self.config.num_cores
            print("Number of cores are {}".format(num_cores))
            metrics_scramble = Parallel(n_jobs=num_cores)(delayed(self._parallel_file_processing)(patient_num, getattr(self.doctor, "patient_{}".format(f'{patient_num:04}')), metrics, types) for patient_num in tqdm(range(self.doctor.num_children)))
        else:
            metrics_scramble = []
            for patient_num in tqdm(range(5)):
            # for patient_num in tqdm(range(self.doctor.num_children)):
                # print(self.patient_map["patient_{}".format(f'{patient_num:04}')])
                metrics_scramble.append(self._parallel_file_processing(patient_num, getattr(self.doctor, "patient_{}".format(f'{patient_num:04}')),metrics, types))
        self._gather_filenames(metrics_scramble, types)
        if self.config.load_metrics == "after":
            #TODO: On release this save metrics could be erased or be commented.
            self._save_metrics(metrics_scramble) # Double saving to avoid double effort
            # metrics_scramble = self._load_metrics()
            print("We got metrics from {} patients".format(len(metrics_scramble)))
            metrics = self._gather_metrics(metrics_scramble)
            self._save_metrics(metrics)


    class _Node(object):
        """
        Inner class to allow the parent-children structure
        """
        def __init__(self, name, parent, data_file ):
            self.parent = parent
            self.set_parent(parent, "parent")
            self.data_file = data_file
            self.num_children = 0
            self.name = name

        def set_parent(self, parent, name):
            if self.parent and self.parent is not parent:
                try:
                    children = getattr(self.parent, name)
                    children.remove(self)
                except:
                    pass
            self.parent = parent
        def set_reader(self, reader):
            self.reader = reader

        def add_child(self, name, node):
            self.__setattr__(name, node)
            node.set_parent(self, name)
            self.num_children += 1
