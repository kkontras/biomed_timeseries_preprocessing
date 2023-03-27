from scipy.stats.stats import pearsonr
import numpy as np
import pickle
import matplotlib.pyplot as plt

# with open('/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patient_data.pkl', 'wb') as f:
#     pickle.dump({"patient_map": patient_map, "data": r, "headers": header}, f)
with open('/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patient_data.pkl', 'rb') as f:
    data = pickle.load(f)

r = data["data"]
header = data["headers"]
r = np.array(r)
r = r.transpose()

desired_headers = ["nsrrid","bmi_s1","bmi_s2","height","weight","fev1","fvc","afibincident","afibprevalent","chol","diasbp","systbp","avg23bpd_s2","avg23bps_s2","ethnicity","gender","race","educat","age_category_s1","age_s1","smokstat_s1","smokstat_s2","ace1","aced1","anar1a1","anar1b1","anar1c1","anar31","benzod1","insuln1","tca1"]
var_list = []
chosen_headers = []
for h in desired_headers:
    try:
        h_i = header.index(h)
        chosen_headers.append(h)
    except:
        print("{} is not on the list".format(h))
        continue
    print(h)
    print(h_i)
    h_i_list = []
    for numeric_string in r[h_i]:
        if numeric_string == "":
            h_i_list.append(0)
        elif type(numeric_string) == np.str_ and numeric_string.isnumeric():
            h_i_list.append(float(numeric_string))
        elif type(numeric_string) != tuple:
            if type(numeric_string[0]) == str and numeric_string[0].isnumeric():
                h_i_list.append(float(numeric_string[0]))
            else:
                print("here")
                print(type(numeric_string))
                print(numeric_string.isnumeric())
                print(numeric_string)
        else:
            print(numeric_string)
            print(type(numeric_string))

    print(len(h_i_list))
    var_list.append(h_i_list)
var_list = np.array(var_list)

import copy
import pickle


# with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patient_data.pkl", "rb") as f:
#     results_shhs1 = pickle.load(f)

# with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS_2/f1kentrpy_results_shhs2.pkl", "rb") as f:
#     results_shhs2 = pickle.load(f)
# with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS_2/f1kentrpy_results_shhs1.pkl", "rb") as f:
#     results = pickle.load(f)
# with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS_2/f1kentrpy_results_shhs1.pkl", "rb") as f:
#     results_merged = pickle.load(f)

with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patient_perf_eeg_shhs1.pkl", "rb") as f:
    results_eeg = pickle.load(f)
with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patient_perf_merged_shhs1.pkl", "rb") as f:
    results_merged = pickle.load(f)
filen_merged = './configs/shhs/myprepro/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json'
filen_merged = './configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json'
filen_eeg = './configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json'

with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_NCH/Version_1/f1kentrpy_results_nch_eeg.pkl", "rb") as f:
    results_eeg = pickle.load(f)
with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_NCH/Version_1/f1kentrpy_results_nch_merged.pkl", "rb") as f:
    results_merged = pickle.load(f)
filen_merged = './configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv_temp.json'
filen_eeg = './configs/nch/single_channel/fourier_transformer_eeg_mat_emphasisonN1.json'
x1 = []
x2 = []
nri = []
for shhs1_merged_i in results_merged["f1"][filen_merged].keys():
    for shhs1_eeg_i in results_eeg["f1"][filen_eeg].keys():
        if shhs1_merged_i == shhs1_eeg_i:
            print("{0:.2f}  {1:.2f}".format(results_merged["f1"][filen_merged][shhs1_merged_i], results_eeg["f1"][filen_eeg][shhs1_eeg_i]))
            x1.append(results_eeg["f1"][filen_eeg][shhs1_eeg_i])
            x2.append(results_merged["f1"][filen_merged][shhs1_merged_i])
            nri.append(shhs1_merged_i)

x1 = []
x2 = []
nri = []
for shhs1_merged_i in results_merged["f1"][filen_merged].keys():
    for shhs1_eeg_i in results_eeg["f1"][filen_eeg].keys():
        if shhs1_merged_i.split("-")[1] == shhs1_eeg_i.split("-")[1]:
            print("{0:.2f}  {1:.2f}".format(results_merged["f1"][filen_merged][shhs1_merged_i], results_eeg["f1"][filen_eeg][shhs1_eeg_i]))
            x1.append(results_eeg["f1"][filen_eeg][shhs1_eeg_i])
            x2.append(results_merged["f1"][filen_merged][shhs1_merged_i])
            nri.append(shhs1_merged_i.split("-")[1])

patient_sortargs = np.argsort(x1)
x1 = [x1[i] for i in patient_sortargs]
x2 = [x2[i] for i in patient_sortargs]

colors = np.array(x1) > np.array(x2)
colors = ["orange" if i else "lightblue" for i in colors]
plt.figure(figsize=(25, 5))
x = np.linspace(0, len(x1) - 1, len(x1))
plt.xlabel("Patient")
plt.ylabel("F1")
plt.title("F1 Comparison")
plt.plot(x, x1, 'o', color='orange', label="EEG")
plt.plot(x, x2, 'o', color='lightblue', label="EEG-EOG")
# for i in range(len(x)):
#     plt.axvline(x[i], 0, 1, color=colors[i])
plt.legend()
plt.show()

with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_NCH/Version_1/f1kentrpy_results_nch_eeg.pkl", "rb") as f:
    results_eeg = pickle.load(f)
with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_NCH/Version_1/f1kentrpy_results_nch_merged.pkl", "rb") as f:
    results_merged = pickle.load(f)



filen = './configs/shhs/myprepro/multi_modal/eeg_eog/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json'
x1 = []
x2 = []
nri = []
for shhs2_i in results_shhs2["f1"][filen].keys():
    for shhs1_i in results["f1"][filen].keys():
        if shhs2_i.split("-")[1] == shhs1_i.split("-")[1]:
            print("{0:.2f}  {1:.2f}".format(results_shhs2["f1"][filen][shhs2_i], results["f1"][filen][shhs1_i]))
            x1.append(results["f1"][filen][shhs1_i])
            x2.append(results_shhs2["f1"][filen][shhs2_i])
            nri.append(shhs2_i.split("-")[1])



with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Sleep_SHHS/Version_1/patient_perf_eeg_shhs1.pkl",
          "rb") as f:
    eeg_shhs1 = pickle.load(f)
filen = './configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json'
x_eeg = []
nri = []
for shhs1_i in eeg_shhs1["f1"][filen].keys():
        x_eeg.append(eeg_shhs1["f1"][filen][shhs1_i])
        nri.append(shhs1_i.split("-")[1])


patient_sortargs = np.argsort(x1)

x1_s = [x1[i] for i in patient_sortargs]
x2_s = [x2[i] for i in patient_sortargs]

colors = np.array(x1_s) > np.array(x2_s)
colors = ["orange" if i else "lightblue" for i in colors]
plt.figure(figsize=(25, 5))
x = np.linspace(0, len(x1_s) - 1, len(x1_s))
plt.xlabel("Patients", fontdict={'fontsize': 36})
plt.ylabel("MF1", fontdict={'fontsize': 36})
plt.title("MF1 Comparison", fontdict={'fontsize': 36})
plt.plot(x, x1_s, 'o', color='orange', label="SHHS1")
plt.plot(x, x2_s, 'o', color='lightblue', label="SHHS2")
# for i in range(len(x)):
#     plt.axvline(x[i], 0, 1, color=colors[i])
plt.legend()
plt.show()

total_data = []
for i in var_list.transpose():
    for j, ni in enumerate(nri):
        if int(i[0]) == int(ni):
            m = copy.deepcopy(list(i))
            m.append(x1[j])
            m.append(x2[j])
            print(len(m))
            total_data.append(m)

R1 = np.corrcoef(var_list)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(R1,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(chosen_headers),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(chosen_headers)
ax.set_yticklabels(chosen_headers)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(R1[25:],cmap='hot', vmin=-1, vmax=1)
for (i, j), z in np.ndenumerate(R1[25:]):
    if np.absolute(z)>0.2:
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize=4, fontweight=800)
fig.colorbar(cax)
ticks = np.arange(0,len(chosen_headers),1)
ax.set_yticks(np.arange(0,2,1))
plt.xticks(rotation=90)
ax.set_xticks(ticks)
ax.set_xticklabels(chosen_headers)
ax.set_yticklabels(["F1 SHHS1","F1 SHHS2"])
plt.show()
# pearsonr(var1, var2)
# np.corrcoef(weight, height)