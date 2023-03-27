
import time
import h5py
import pickle
import zarr
import numpy as np

f1 = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/VDec_shhs1_mat/train/patient_0001/file_00/n0001_f0000_eog.mat"
f2 = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/VDec_shhs1_mat/train/patient_0001/file_00/n0001_f0000_eog.hdf5"
f3 = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/VDec_shhs1_mat/train/patient_0001/file_00/n0001_f0000_eog.npz"
f4 = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/VDec_shhs1_mat/train/patient_0001/file_00/n0001_f0000_eog.pkl"
f5 = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/VDec_shhs1_mat/train/patient_0001/file_00/n0001_f0000_eog.zarr"

data_idx = 0
end_idx = 1

print("Mat v7.3", end="    ")
start = time.time()
f = h5py.File(f1, 'r')
signal = f["X2"][data_idx:end_idx]
label = f["labels"][data_idx:end_idx]
f.close()
print("{0:.6f} sec".format(time.time() - start))
print("NPZ", end="         ")
start = time.time()
f = dict(np.load(f3))
signal = f["X2"][data_idx:end_idx]
label = f["labels"][data_idx:end_idx]
print("{0:.6f} sec".format(time.time() - start))
print("Zarr", end="        ")
start = time.time()
f = zarr.open(f5, 'r')
signal = f["X2"][data_idx:end_idx]
label = f["labels"][data_idx:end_idx]
print("{0:.6f} sec".format(time.time() - start))
print("Pickle", end="      ")
start = time.time()
with open(f4, 'rb') as handle:
    f = pickle.load(handle)
    signal = f["X2"][data_idx:end_idx]
    label = f["labels"][data_idx:end_idx]
print("{0:.6f} sec".format(time.time() - start))
print("HDF5", end="        ")
start = time.time()
f = h5py.File(f2, 'r')
signal = f["X2"][data_idx:end_idx]
label = f["labels"][data_idx:end_idx]
f.close()
print("{0:.6f} sec".format(time.time() - start))