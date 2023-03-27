import os
import h5py
import numpy as np

def list_dirs(filename):
    list_of_dirs = []
    for fname in os.listdir(filename):
        path = os.path.join(filename, fname)
        if os.path.isdir(path):
            new_list = list_dirs(path)
            new_list_r = [fname + "/" + i for i in new_list]
            if new_list:
                list_of_dirs += new_list_r
        else:
            list_of_dirs.append(fname)
    return list_of_dirs
path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_NCH/Version_1/train"
a = list_dirs(path)
pat_eog, pat_eog_stft, pat_eeg, pat_emg, pat_emg_stft, pat_eeg_stft = [],[],[],[],[],[]
for i in a:
    if "patient" not in i:
        continue
    if "eeg_stft" in i:
        pat_eeg_stft.append(i)
    elif "eeg" in i:
        pat_eeg.append(i)
    elif "eog_stft" in i:
        pat_eog_stft.append(i)
    elif "eog" in i:
        pat_eog.append(i)
    elif "emg_stft" in i:
        pat_emg_stft.append(i)
    elif "emg" in i:
        pat_emg.append(i)

list_of_mods = [pat_eog, pat_eog_stft, pat_eeg, pat_emg, pat_emg_stft, pat_eeg_stft]
dts = ["time_eog", "stft_eog", "time_eeg", "time_emg", "stft_emg", "stft_eeg"]

# for j, ll in enumerate(list_of_mods):
#     dt = dts[j]
#     ll.sort()
#     with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_NCH/Version_1/" + "/train" + '/{}_file_map.txt'.format(dt), 'w+') as fp:
#         for i in ll:
#             f = h5py.File("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_NCH/Version_1/"+i, 'r', swmr=True)
#             m = len(f["labels"])
#             f.close()
#             fp.write("{}-{}\n".format(i, m))
#     fp.close()
#     print("done {}".format(dt))

list_of_mods = [pat_eog_stft, pat_emg_stft, pat_eeg_stft]
dts = ["stft_eog", "stft_emg", "stft_eeg"]

# list_of_mods = [pat_eog, pat_emg, pat_eeg]
# dts = ["time_eog", "time_emg", "time_eeg"]
totals = {}
for j, ll in enumerate(list_of_mods):
    dt = dts[j]
    ll.sort()
    totals[dts[j]] = {"dirs": [], "lens": []}
    for i in ll:
        f = h5py.File(path + i, 'r', swmr=True)
        m = len(f["labels"])
        f.close()
        totals[dts[j]]["dirs"].append(i)
        totals[dts[j]]["lens"].append(m)
        print("{}-{}".format(i, m))
    print("done {}".format(dt))

patient_names = []
patient_lens = []
for part in totals:
    for i, name in enumerate(totals[part]["dirs"]):
        this_name = name.split("/")[-1].split("_")[0] + "_" + name.split("/")[-1].split("_")[1]
        if this_name not in patient_names:
            patient_names.append(this_name)
            patient_lens.append(totals[part]["lens"][i])
        else:
            pos_name = patient_names.index(this_name)
            if totals[part]["lens"][i] != patient_lens[pos_name] and totals[part]["lens"][i] != 0:
                print(patient_names[pos_name])
                print(patient_lens[pos_name])
                print(totals[part]["dirs"][i])
                print(totals[part]["lens"][i])
                print("Lens are not the same for the same file between modalities")

temp = np.array([[x,i] for i, x in sorted(zip(patient_names, patient_lens))])
patient_lens, patient_names = temp[:,0], temp[:,1]


to_write = {}
for part in totals:
    totals[part]["totals"] = []

for i, name in enumerate(patient_names):
    for part in totals:
        absense_of_name = True
        for ndir in totals[part]["dirs"]:
            if name in ndir:
                absense_of_name = False
                totals[part]["totals"].append("{}-{}".format(ndir, patient_lens[i]))
        if absense_of_name:
            totals[part]["totals"].append("{}-{}".format("empty", patient_lens[i]))

for part in totals:
    with open(path + '{}_file_map_empties.txt'.format(part), 'w+') as fp:
        for i in totals[part]["totals"]:
            fp.write(path+"{}\n".format(i))