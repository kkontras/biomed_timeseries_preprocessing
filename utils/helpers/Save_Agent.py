import os
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET

import einops
import pandas as pd
import mne
import h5py
import hdf5storage
import numpy as np
from scipy.io import savemat
import csv
from os import path
from pathlib import Path
import zarr
import copy
import shutil
import pickle
import matlab.engine
import torch
import warnings

# eng = matlab.engine.start_matlab()

def Get_Savior(save_type, dataset_name, savedir, test_patients, patient_num, file_num, patient_name):

    if save_type == "npz":
        return Save_Agent_NPZ(save_type=save_type, savedir=savedir, test_patients=test_patients,
                              patient_num=patient_num, file_num=file_num, patient_name=patient_name)

    elif save_type == "mat":
        return Save_Agent_MAT(save_type=save_type, savedir=savedir, test_patients=test_patients,
                                   patient_num=patient_num, file_num=file_num, patient_name=patient_name)

    elif save_type == "hdf5" and (dataset_name == "nch" or dataset_name=="nch_total"):
        return Save_Agent_HDF5_NCH(save_type=save_type, savedir=savedir, test_patients=test_patients,
                                   patient_num=patient_num, file_num=file_num, patient_name=patient_name)

    elif save_type == "hdf5":
        return Save_Agent_HDF5(save_type=save_type, savedir=savedir, test_patients=test_patients,
                                   patient_num=patient_num, file_num=file_num, patient_name=patient_name)

    elif save_type == "zarr_one":
        return Save_Agent_ZARR_One(save_type=save_type, savedir=savedir, test_patients=test_patients,
                               patient_num=patient_num, file_num=file_num, patient_name=patient_name)
    elif save_type == "npz_small":
        return Save_Agent_NPZ_Small(save_type=save_type, savedir=savedir, test_patients=test_patients,
                               patient_num=patient_num, file_num=file_num, patient_name=patient_name)

    elif save_type == "zarr":
        return Save_Agent_Zarr(save_type=save_type, savedir=savedir, test_patients=test_patients,
                               patient_num=patient_num, file_num=file_num, patient_name=patient_name)

    elif save_type == "pkl":
        return Save_Agent_Pickle(save_type=save_type, savedir=savedir, test_patients=test_patients,
                               patient_num=patient_num, file_num=file_num, patient_name=patient_name)

    else:
        assert "Saving agent {} does not exist options are npz, mat, hdf5, zarr, pkl, zarr_one, npz_small".format(save_type)


class Save_Agent(ABC):

    def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
        self.save_type = save_type
        self.savedir = savedir
        self.test_patients = test_patients
        self.patient_num = patient_num
        self.file_num = file_num
        self.patient_name = patient_name

    @abstractmethod
    def save_windows(self, data, metrics):
        raise NotImplementedError()

    @abstractmethod
    def save_windows_stft(self, data, metrics):
        raise NotImplementedError()

    def create_files(self):
        if self.patient_name in self.test_patients:
            file_save_dir = self.savedir + "/test" + "/patient_{}".format(f'{self.patient_num:04}')
        else:
            file_save_dir = self.savedir + "/train" + "/patient_{}".format(f'{self.patient_num:04}')
        Path(file_save_dir).mkdir(parents=True, exist_ok=True)
        file_save_dir = file_save_dir + "/file_" + str(f'{self.file_num:02}')
        Path(file_save_dir).mkdir(parents=True, exist_ok=True)
        return file_save_dir

    def save_windows(self, data, metrics):
        metrics["files_time"] = {}
        file_save_dir = self.create_files()
        for modality in data["mod_name"]:
            save_filenames = []
            file_name = file_save_dir + '/n{}_f{}_{}.{}'.format(f'{self.patient_num:04}', f'{self.file_num:04}',
                                                                modality, self.save_type)

            save_response = self.save_specific_window(data[modality], file_name=file_name)
            if save_response:
                save_filenames.append([file_name, len(data[modality]["windows"][0])])
                metrics["files_time"][modality] = save_filenames
            else:
                warnings.warn("File {} was not saved due to its len been {}".format(file_name, len(data[modality]["windows"][0])))

        # TODO: On release you should uncomment this to make sure that modalities are correctly and alignly saved.

        # for modality_2 in data["modalities"]:
        #     if isinstance(data[modality]["w_labels"],(np.ndarray, np.generic)) and  isinstance(data[modality]["w_labels"],(np.ndarray, np.generic)):
        #         if len(data[modality]["w_labels"])==len(data[modality_2]["w_labels"]) and (data[modality]["w_labels"] != data[modality_2]["w_labels"]).any():
        #             try:
        #                 raise Exception ("Labels on different modalities do not comply")
        #             except:
        #                 print("a")
        #     else:
        #         raise Warning ("Something strange happens here")
        return metrics

    def save_windows_stft(self, data, metrics):
        metrics["files_stft"] = {}
        file_save_dir = self.create_files()
        for modality in data["mod_name"]:
            save_filenames = []
            file_name = file_save_dir + '/n{}_f{}_{}_stft.{}'.format(f'{self.patient_num:04}', f'{self.file_num:02}',
                                                                     modality, self.save_type)
            self.save_specific_window_stft(data[modality], file_name=file_name)

            save_response = self.save_specific_window_stft(data[modality], file_name=file_name)
            if save_response:
                save_filenames.append([file_name, len(data[modality]["windows_stft"][0])])
                metrics["files_stft"][modality] = save_filenames
            else:
                warnings.warn("File {} was not saved due to its len been {}".format(file_name, len(data[modality]["windows_stft"][0])))



        return metrics

    def save_specific_window(self):
        raise NotImplementedError("This function should have been inherited.")
    def save_specific_window_stft(self):
        raise NotImplementedError("This function should have been inherited.")

class Save_Agent_NPZ_Small(Save_Agent):
    def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
        super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)

    def save_windows(self, data, metrics):

        metrics["files_time"] = {}
        file_save_dir = self.create_files()
        for modality in data["modalities"]:
            save_filenames = []
            # print("{}-{}".format(len(data[modality]["windows"]),len(data[modality]["w_labels"])))
            for i in range(len(data[modality]["windows"])):
                file_name = file_save_dir + '/img_{}_{}.npz'.format(f'{i:05}', modality)
                save_filenames.append([file_name, 1])  # Save name of file and how many windows it contains
                np.savez_compressed(file_name, data[modality]["windows"][i])
            metrics["files_time"][modality] = save_filenames

            for modality_2 in data["modalities"]:
                if isinstance(data[modality]["w_labels"], (np.ndarray, np.generic)) and isinstance(
                        data[modality]["w_labels"], (np.ndarray, np.generic)):
                    if len(data[modality]["w_labels"]) == len(data[modality_2]["w_labels"]) and (
                            data[modality]["w_labels"] != data[modality_2]["w_labels"]).any():
                        try:
                            raise Exception("Labels on different modalities do not comply")
                        except:
                            print("a")
                else:
                    raise Warning("Something strange happens here")

        with open(self.savedir + "_labels.csv", "wt") as csvfile:
            writer = csv.writer(csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            for i in range(len(data[modality]["windows"])):
                writer.writerow(["{}".format(data[modality]["w_labels"][i][0]),
                                 "{}".format(data[modality]["w_labels"][i][1])])
        return metrics

    def save_windows_stft(self, data, metrics):
        metrics["files_stft"] = {}
        file_save_dir = self.create_files()
        for modality in data["modalities"]:
            save_filenames = []
            for i in range(len(data[modality]["windows_stft"])):
                filename = file_save_dir + '/img_{}_{}_stft.npz'.format(f'{i:05}', modality)
                save_filenames.append([filename, 1])  # Save name of file and how many windows it contains
                np.savez_compressed(filename, data[modality]["windows_stft"][i])
            metrics["files_stft"][modality] = save_filenames
        if not path.exists(self.savedir + "_labels.csv"):
            with open(self.savedir + "{}_labels_stft.csv".format(modality), "wt") as csvfile:
                writer = csv.writer(csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
                for i in range(len(data[modality]["windows_stft"])):
                    writer.writerow(["{}".format(data[modality]["w_labels"][i][0]),
                                     "{}".format(data[modality]["w_labels"][i][1])])
        return metrics

class Save_Agent_ZARR_One(Save_Agent):

    def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
        super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)

    def create_zarr(self, zarr_dir):
        if not path.exists(zarr_dir):
            self.zarr_file = zarr.open(zarr_dir, 'w')
        else:
            self.zarr_file = zarr.open(zarr_dir, 'r+')

    def create_zarr_patient_files_datasets(self):

        if not "datasets" in self.zarr_file:
            datasets_grp = self.zarr_file.create_group("datasets")
        else:
            datasets_grp = self.zarr_file["/datasets"]

        if not "patients" in self.zarr_file:
            patients_grp = self.zarr_file.create_group("patients")
        else:
            patients_grp = self.zarr_file["/patients"]

        if not "patient_{}".format(f'{self.patient_num:04}') in patients_grp:
            patient_num_grp = patients_grp.create_group("patient_{}".format(f'{self.patient_num:04}'))
        else:
            patient_num_grp = self.zarr_file["/patients/patient_{}".format(f'{self.patient_num:04}')]

        if not "file_{}".format(f'{self.file_num:02}') in patient_num_grp:
            file_grp = patient_num_grp.create_group("file_{}".format(f'{self.file_num:02}'))
        else:
            file_grp = self.zarr_file[
                "/patients/patient_{}/file_{}".format(f'{self.patient_num:04}', f'{self.file_num:02}')]

        return file_grp, datasets_grp

    def save_windows(self, data, metrics):

        zarr_dir = self.savedir + "/dataset.zarr"
        self.create_zarr(zarr_dir)
        file_grp, datasets_grp = self.create_zarr_patient_files_datasets()
        view = "time"
        metrics["files_{}".format(view)] = {}
        for mod in data["mod_name"]:

            if not "{}_{}".format(mod, view) in file_grp:
                file_grp.create_dataset("{}_{}".format(mod, view), data=data[mod]["windows"], dtype='i')
            else:
                mod_view_dataset = self.zarr_file[
                    "/patients/patient_{}/file_{}/{}_{}".format(f'{self.patient_num:04}', f'{self.file_num:02}', mod,
                                                                view)]
                mod_view_dataset[:] = data[mod]["windows"]

            path_name_wlength = np.array([["/patients/patient_{}/file_{}/{}_{}".format(f'{self.patient_num:04}',
                                                                                       f'{self.file_num:02}', mod,
                                                                                       view),
                                           len(data[mod]["windows"])]])

            if not "{}_{}_name_list".format(mod, view) in datasets_grp:
                datasets_grp.create_dataset('{}_{}_name_list'.format(mod, view), data=path_name_wlength,
                                            synchronizer=zarr.ThreadSynchronizer(), dtype='<U62')
            else:
                pathnames = datasets_grp["{}_{}_name_list".format(mod, view)]
                pathnames.resize((pathnames.shape[0] + path_name_wlength.shape[0], 2))
                pathnames[-1:] = path_name_wlength
                print(pathnames[:])
            metrics["files_{}".format(view)][mod] = path_name_wlength[0]

        if not "labels" in file_grp:
            file_grp.create_dataset("labels", data=data[mod]["w_labels"], dtype='i')
        else:
            mod_view_dataset = self.zarr_file[
                "/patients/patient_{}/file_{}/labels".format(f'{self.patient_num:04}', f'{self.file_num:02}')]
            mod_view_dataset[:] = data[mod]["w_labels"]
        return metrics

    def save_windows_stft(self, data, metrics):
        zarr_dir = self.savedir + "/dataset.zarr"
        self.create_zarr(zarr_dir)
        file_grp, datasets_grp = self.create_zarr_patient_files_datasets()
        view = "stft"
        metrics["files_{}".format(view)] = {}
        for mod in data["modalities"]:

            if not "{}_{}".format(mod, view) in file_grp:
                file_grp.create_dataset("{}_{}".format(mod, view), data=data[mod]["windows"], dtype='i')
            else:
                mod_view_dataset = self.zarr_file[
                    "/patients/patient_{}/file_{}/{}_{}".format(f'{self.patient_num:04}', f'{self.file_num:02}', mod,
                                                                view)]
                mod_view_dataset[:] = data[mod]["windows"]

            path_name_wlength = np.array([["/patients/patient_{}/file_{}/{}_{}".format(f'{self.patient_num:04}',
                                                                                       f'{self.file_num:02}', mod,
                                                                                       view),
                                           len(data[mod]["windows"])]])

            if not "{}_{}_name_list".format(mod, view) in datasets_grp:
                datasets_grp.create_dataset('{}_{}_name_list'.format(mod, view), data=path_name_wlength,
                                            synchronizer=zarr.ThreadSynchronizer(), dtype='<U62')
            else:
                pathnames = datasets_grp["{}_{}_name_list".format(mod, view)]
                pathnames.resize((pathnames.shape[0] + path_name_wlength.shape[0], 2))
                pathnames[-1:] = path_name_wlength
                print(pathnames[:])
            metrics["files_{}".format(view)][mod] = path_name_wlength[0]

        if not "labels" in file_grp:
            file_grp.create_dataset("labels", data=data[mod]["w_labels"], dtype='i')
        else:
            mod_view_dataset = self.zarr_file[
                "/patients/patient_{}/file_{}/labels".format(f'{self.patient_num:04}', f'{self.file_num:02}')]
            mod_view_dataset[:] = data[mod]["w_labels"]
        return metrics

# class Save_Agent_MAT(Save_Agent):
#     def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
#         super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)
#
#     def save_specific_window(self, data, file_name):
#
#         # d ={u"{}".format(i):data["windows"][i].astype(np.float32) for i in data["windows"].keys()}
#         # l ={u"{}".format(i):data["w_labels"][i].astype(np.int) for i in data["w_labels"].keys()}
#         # hdf5storage.write( {u"X2": d, u"labels": l}, '.', file_name, matlab_compatible=True, store_python_metadata=True)
#
#         #Correct version
#         # hf = h5py.File(file_name, 'w')
#         # x2 = hf.create_group("X2")
#         # for i in data["windows"].keys():
#         #     x2.create_dataset('ch_{}'.format(i), data=data["windows"][i].astype(np.float32))
#         # hf.create_dataset('labels', data=data["w_labels"][0].astype(np.int))
#         # hf.close()
#
#
#         #Matlab version
#         for i in data["windows"].keys():
#             eng.workspace['X2_ch_{}'.format(i)] = matlab.double(data["windows"][i].transpose().tolist())
#         eng.workspace['labels'] = matlab.double(data["w_labels"][0].transpose().astype(np.int).tolist())
#         eng.workspace['inits'] = matlab.double(data["w_inits"][0].astype(np.int).tolist())
#         if len(data["windows"].keys()) == 1:
#             eng.save(file_name, 'X2_ch_0', 'labels', '-v7.3', nargout=0)
#         elif len(data["windows"].keys()) == 2:
#             eng.save(file_name, 'X2_ch_0', 'X2_ch_1', 'labels', 'inits', '-v7.3', nargout=0)
#
#
#         return True
#
#     def save_specific_window_stft(self, data, file_name):
#         # d ={u"{}".format(i):data["windows_stft"][i].astype(np.float32) for i in data["windows_stft"].keys()}
#         # l ={u"{}".format(i):data["w_labels_stft"][i].astype(np.int) for i in data["w_labels_stft"].keys()}
#         # hdf5storage.write( {u"X2": einops.rearrange(data["windows_stft"][0].astype("f4"), "a b c -> b c a"), u"labels":  data["w_labels_stft"][0].astype(int)}, '.', file_name, matlab_compatible=False, store_python_metadata=False)
#
#         #Correct Version
#         # hf = h5py.File(file_name, 'w')
#         # x2 = hf.create_group("X2")
#         # for i in data["windows_stft"].keys():
#         #     x2.create_dataset('ch_{}'.format(i), data=data["windows_stft"][i].astype(np.float32))
#         # hf.create_dataset('labels', data=data["w_labels_stft"][0].astype(np.int))
#         # hf.close()
#
#         #MAtlab version
#         for i in data["windows_stft"].keys():
#             eng.workspace['X2_ch_{}'.format(i)] = matlab.double(data["windows_stft"][i].transpose().tolist())
#         eng.workspace['labels'] = matlab.double(data["w_labels_stft"][0].astype(np.int).transpose().tolist())
#         eng.workspace['inits'] = matlab.double(data["w_inits_stft"][0].astype(np.int).tolist())
#         if len(data["windows_stft"].keys()) == 1:
#             eng.save(file_name, 'X2_ch_0', 'labels', 'inits', '-v7.3', nargout=0)
#         elif len(data["windows_stft"].keys()) == 2:
#             eng.save(file_name, 'X2_ch_0', 'X2_ch_1', 'labels', 'inits', '-v7.3', nargout=0)
#
#         return True

class Save_Agent_NPZ(Save_Agent):
    def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
        super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)

    def save_specific_window(self, data, file_name):

        data_to_be_saved = {"X2": data["windows"], "labels": data["w_labels"]}
        np.savez(file_name, **data_to_be_saved)
        return True

    def save_specific_window_stft(self, data, file_name):

        data_to_be_saved = {"X2": data["windows_stft"], "labels": data["w_labels_stft"]}
        np.savez(file_name, **data_to_be_saved)
        return True

class Save_Agent_HDF5_NCH(Save_Agent):
    def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
        super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)

    def save_specific_window(self, data, file_name):

        sum = 0
        for i in data["windows"].keys():
            sum += len(data["windows"][i])
        if sum<10:
            return False

        hf = h5py.File(file_name, 'w')
        for i in data["windows"].keys():
            hf.create_dataset('X2_ch_{}'.format(i), data=data["windows"][i])
        hf.create_dataset('labels', data=data["w_labels"][0])
        hf.create_dataset('inits', data=data["w_inits"][0])
        hf.close()

        return True

    def save_specific_window_stft(self, data, file_name):

        sum = 0
        for i in data["windows"].keys():
            sum += len(data["windows"][i])
        if sum<10:
            return False

        hf = h5py.File(file_name, 'w')

        for i in data["windows_stft"].keys():
            hf.create_dataset('X2_ch_{}'.format(i), data=data["windows_stft"][i])

        # hf.create_dataset('X2', data=data["windows_stft"])
        hf.create_dataset('labels', data=data["w_labels_stft"][0])
        hf.create_dataset('inits', data=data["w_inits_stft"][0])
        hf.close()

        return True

class Save_Agent_HDF5(Save_Agent):
    def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
        super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)

    def save_specific_window(self, data, file_name):

        hf = h5py.File(file_name, 'w')
        for i in data["windows"].keys():
            hf.create_dataset('X2_ch_{}'.format(i), data=data["windows"][i])
        hf.create_dataset('labels', data=data["w_labels"][0])
        hf.create_dataset('inits', data=data["w_inits"][0])
        hf.close()

        return True

    def save_specific_window_stft(self, data, file_name):

        hf = h5py.File(file_name, 'w')

        for i in data["windows_stft"].keys():
            hf.create_dataset('X2_ch_{}'.format(i), data=data["windows_stft"][i])

        # hf.create_dataset('X2', data=data["windows_stft"])
        hf.create_dataset('labels', data=data["w_labels_stft"][0])
        hf.create_dataset('inits', data=data["w_inits_stft"][0])
        hf.close()

        return True

class Save_Agent_Pickle(Save_Agent):
    def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
        super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)

    def save_specific_window(self, data, file_name):

        with open(file_name, 'wb') as handle:
            pickle.dump({"X2": data["windows"], "labels": data["w_labels"]}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def save_specific_window_stft(self, data, file_name):

        with open(file_name, 'wb') as handle:
            pickle.dump({"X2": data["windows_stft"], "labels": data["w_labels_stft"]}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

class Save_Agent_Zarr(Save_Agent):
    def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
        super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)

    def save_specific_window(self, data, file_name):
        if path.exists(file_name):
            shutil.rmtree(file_name)
        zarr_file = zarr.open(file_name, 'w')
        zarr_file.create_dataset("X2", data=data["windows"], dtype='i',
                                 synchronizer=zarr.ThreadSynchronizer())
        zarr_file.create_dataset("labels", data=data["w_labels"], dtype='i',
                                 synchronizer=zarr.ThreadSynchronizer())
        return True

    def save_specific_window_stft(self, data, file_name):
        if path.exists(file_name):
            shutil.rmtree(file_name)
        zarr_file = zarr.open(file_name, 'w')
        zarr_file.create_dataset("X2", data=data["windows_stft"], dtype='i',
                                 synchronizer=zarr.ThreadSynchronizer())
        zarr_file.create_dataset("labels", data=data["w_labels_stft"], dtype='i',
                                 synchronizer=zarr.ThreadSynchronizer())
        return True

#
# class Save_Agent_MAT_HDF5(Save_Agent):
#     def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
#         super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)
#
#     def σ(self, data, metrics):
#         metrics["files_time"] = {}
#         file_save_dir = self.create_files()
#         for modality in data["mod_name"]:
#             save_filenames = []
#             # print("{}-{}".format(len(data[modality]["windows"]),len(data[modality]["w_labels"])))
#
#             file_name = file_save_dir + '/n{}_f{}_{}.{}'.format(f'{self.patient_num:04}', f'{self.file_num:04}',
#                                                                 modality, self.save_type)
#             save_filenames.append([file_name, len(data[modality]["windows"])])
#             # savemat(file_name, {"X2": data[modality]["windows"], "labels": data[modality]["w_labels"]})
#
#             hdf5storage.write( {u"X2": data[modality]["windows"], u"labels": data[modality]["w_labels"]}, '.', file_name, matlab_compatible=True, store_python_metadata=False)
#
#             # TODO: to add the v7.3 mat files I changed the function that we saved, so check if the new one is compatible for
#
#             # hdf5storage.savemat(mdict={"X2": data[modality]["windows"], "labels": data[modality]["w_labels"]}, file_name=file_name, store_python_metadata=False, matlab_compatible=False)
#             # hdf5storage.write(data={u"X2": data[modality]["windows"], u"labels": data[modality]["w_labels"]}, path=file_save_dir, filename=file_name, matlab_compatible=True)
#             metrics["files_time"][modality] = save_filenames
#
#             # TODO: On release you should uncomment this to make sure that modalities are correctly and alignly saved.
#
#             # for modality_2 in data["modalities"]:
#             #     if isinstance(data[modality]["w_labels"],(np.ndarray, np.generic)) and  isinstance(data[modality]["w_labels"],(np.ndarray, np.generic)):
#             #         if len(data[modality]["w_labels"])==len(data[modality_2]["w_labels"]) and (data[modality]["w_labels"] != data[modality_2]["w_labels"]).any():
#             #             try:
#             #                 raise Exception ("Labels on different modalities do not comply")
#             #             except:
#             #                 print("a")
#             #     else:
#             #         raise Warning ("Something strange happens here")
#
#         return metrics
#
#     def save_windows_stft(self, data, metrics):
#         metrics["files_stft"] = {}
#         file_save_dir = self.create_files()
#         for modality in data["mod_name"]:
#             save_filenames = []
#             file_name = file_save_dir + '/n{}_f{}_{}_stft.{}'.format(f'{self.patient_num:04}', f'{self.file_num:02}',
#                                                                      modality, self.save_type)
#             save_filenames.append([file_name, len(data[modality]["windows_stft"])])
#             savemat(file_name, {"X2": data[modality]["windows_stft"], "labels": data[modality]["w_labels_stft"]})
#             # hdf5storage.savemat(mdict={u"X2": data[modality]["windows_stft"], u"labels": data[modality]["w_labels_stft"]}, file_name=file_name, store_python_metadata=False, matlab_compatible=False)
#             # if path.exists(file_name):
#             #     os.remove(file_name)
#             # zarr_file = zarr.open(file_name, 'w')
#             # zarr_file.create_dataset("X2", data=data[modality]["windows_stft"], dtype='i', synchronizer=zarr.ThreadSynchronizer())
#             # zarr_file.create_dataset("labels", data=data[modality]["w_labels_stft"], dtype='i', synchronizer=zarr.ThreadSynchronizer())
#
#             metrics["files_stft"][modality] = save_filenames
#
#         return metrics
#
# class Save_Agent_ZARR(Save_Agent):
#     def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
#         super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)
#
#     def save_windows(self, data, metrics):
#         metrics["files_time"] = {}
#         file_save_dir = self.create_files()
#         for modality in data["mod_name"]:
#             save_filenames = []
#             # print("{}-{}".format(len(data[modality]["windows"]),len(data[modality]["w_labels"])))
#
#             file_name = file_save_dir + '/n{}_f{}_{}.{}'.format(f'{self.patient_num:04}', f'{self.file_num:04}',
#                                                                 modality, self.save_type)
#             save_filenames.append([file_name, len(data[modality]["windows"])])
#
#             metrics["files_time"][modality] = save_filenames
#
#             if path.exists(file_name):
#                 shutil.rmtree(file_name)
#             zarr_file = zarr.open(file_name, 'w')
#             zarr_file.create_dataset("X2", data=data[modality]["windows"], dtype='i', synchronizer=zarr.ThreadSynchronizer())
#             zarr_file.create_dataset("labels", data=data[modality]["w_labels"], dtype='i', synchronizer=zarr.ThreadSynchronizer())
#
#             # TODO: On release you should uncomment this to make sure that modalities are correctly and alignly saved.
#
#             # for modality_2 in data["modalities"]:
#             #     if isinstance(data[modality]["w_labels"],(np.ndarray, np.generic)) and  isinstance(data[modality]["w_labels"],(np.ndarray, np.generic)):
#             #         if len(data[modality]["w_labels"])==len(data[modality_2]["w_labels"]) and (data[modality]["w_labels"] != data[modality_2]["w_labels"]).any():
#             #             try:
#             #                 raise Exception ("Labels on different modalities do not comply")
#             #             except:
#             #                 print("a")
#             #     else:
#             #         raise Warning ("Something strange happens here")
#
#         return metrics
#
#     def save_windows_stft(self, data, metrics):
#         metrics["files_stft"] = {}
#         file_save_dir = self.create_files()
#         for modality in data["mod_name"]:
#             save_filenames = []
#             file_name = file_save_dir + '/n{}_f{}_{}_stft.{}'.format(f'{self.patient_num:04}', f'{self.file_num:02}',
#                                                                      modality, self.save_type)
#             save_filenames.append([file_name, len(data[modality]["windows_stft"])])
#
#             if path.exists(file_name):
#                 shutil.rmtree(file_name)
#             zarr_file = zarr.open(file_name, 'w')
#             zarr_file.create_dataset("X2", data=data[modality]["windows_stft"], dtype='i', synchronizer=zarr.ThreadSynchronizer())
#             zarr_file.create_dataset("labels", data=data[modality]["w_labels_stft"], dtype='i', synchronizer=zarr.ThreadSynchronizer())
#
#             metrics["files_stft"][modality] = save_filenames
#
#         return metrics
#
#
# class Save_Agent_parquet(Save_Agent):
#     def __init__(self, save_type, savedir, test_patients, patient_num, file_num, patient_name):
#         super().__init__(save_type, savedir, test_patients, patient_num, file_num, patient_name)
#
#     def save_windows(self, data, metrics):
#         metrics["files_time"] = {}
#         file_save_dir = self.create_files()
#         for modality in data["mod_name"]:
#             save_filenames = []
#             # print("{}-{}".format(len(data[modality]["windows"]),len(data[modality]["w_labels"])))
#
#             file_name = file_save_dir + '/n{}_f{}_{}.{}'.format(f'{self.patient_num:04}', f'{self.file_num:04}',
#                                                                 modality, self.save_type)
#             save_filenames.append([file_name, len(data[modality]["windows"])])
#             # savemat(file_name, {"X2": data[modality]["windows"], "labels": data[modality]["w_labels"]})
#
#
#             # TODO: to add the v7.3 mat files I changed the function that we saved, so check if the new one is compatible for
#             hdf5storage.write( {u"X2": data[modality]["windows"], u"labels": data[modality]["w_labels"]}, '.', file_name, matlab_compatible=True, store_python_metadata=False)
#
#             # hdf5storage.savemat(mdict={"X2": data[modality]["windows"], "labels": data[modality]["w_labels"]}, file_name=file_name, store_python_metadata=False, matlab_compatible=False)
#             # hdf5storage.write(data={u"X2": data[modality]["windows"], u"labels": data[modality]["w_labels"]}, path=file_save_dir, filename=file_name, matlab_compatible=True)
#             metrics["files_time"][modality] = save_filenames
#
#             # TODO: On release you should uncomment this to make sure that modalities are correctly and alignly saved.
#
#             # for modality_2 in data["modalities"]:
#             #     if isinstance(data[modality]["w_labels"],(np.ndarray, np.generic)) and  isinstance(data[modality]["w_labels"],(np.ndarray, np.generic)):
#             #         if len(data[modality]["w_labels"])==len(data[modality_2]["w_labels"]) and (data[modality]["w_labels"] != data[modality_2]["w_labels"]).any():
#             #             try:
#             #                 raise Exception ("Labels on different modalities do not comply")
#             #             except:
#             #                 print("a")
#             #     else:
#             #         raise Warning ("Something strange happens here")
#
#         return metrics
#
#     def save_windows_stft(self, data, metrics):
#         metrics["files_stft"] = {}
#         file_save_dir = self.create_files()
#         for modality in data["mod_name"]:
#             save_filenames = []
#             file_name = file_save_dir + '/n{}_f{}_{}_stft.{}'.format(f'{self.patient_num:04}', f'{self.file_num:02}',
#                                                                      modality, self.save_type)
#             save_filenames.append([file_name, len(data[modality]["windows_stft"])])
#             savemat(file_name, {"X2": data[modality]["windows_stft"], "labels": data[modality]["w_labels_stft"]})
#             # hdf5storage.savemat(mdict={u"X2": data[modality]["windows_stft"], u"labels": data[modality]["w_labels_stft"]}, file_name=file_name, store_python_metadata=False, matlab_compatible=False)
#             # if path.exists(file_name):
#             #     os.remove(file_name)
#             # zarr_file = zarr.open(file_name, 'w')
#             # zarr_file.create_dataset("X2", data=data[modality]["windows_stft"], dtype='i', synchronizer=zarr.ThreadSynchronizer())
#             # zarr_file.create_dataset("labels", data=data[modality]["w_labels_stft"], dtype='i', synchronizer=zarr.ThreadSynchronizer())
#
#             metrics["files_stft"][modality] = save_filenames
#
#         return metrics



# np_arr = np.array([1.3, 4.22, -5], dtype=np.float32)
# pa_table = pa.table({"data": np_arr})
# pa.parquet.write_table(pa_table, "test.parquet")