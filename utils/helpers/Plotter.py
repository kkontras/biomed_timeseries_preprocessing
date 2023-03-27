import matplotlib.pyplot as plt
import numpy as np
import einops
from scipy import signal as sg


class Plotter():
    def __init__(self, file_duration, config):
        self.file_duration = file_duration
        self.config = config

    def plot_stft(self, data, mean, std):
        print("We are in plot stft")
        fig = plt.figure()
        num_rows = 1
        for modality in range(data["modalities"]):
            num_rows += data[modality]["num_channels"]
        num_cols = 1
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(wspace=0.05, hspace=0.15)
        ax = [plt.subplot(gs[i]) for i in range(num_rows * num_cols)]
        count_mod = 0
        for modality in range(data["modalities"]):
            srate = data[modality]["sampling_rate"]
            signals = np.array([data[modality][ch] for ch in range(data[modality]["num_channels"])])
            amp = 2 * np.sqrt(2)
            f, t, Zxx = sg.stft(signals, srate, nperseg=2 * srate, noverlap=srate, nfft=self.config.nfft_points)
            Zxx = np.abs(Zxx)
            Zxx[Zxx == 0] = 1e-5
            Zxx = 20 * np.log10(Zxx)
            # print(Zxx.min())
            if (Zxx != Zxx).any():
                print(Zxx.min())
                print(np.isnan(Zxx).any())
    
            if mean and std:
                Zxx = einops.rearrange(Zxx, 'mod freq time -> time freq mod')
                stft_mean = np.array([mean[modality][ch] for ch in range(data[modality]["num_channels"])])
                stft_std = np.array([std[modality][ch] for ch in range(data[modality]["num_channels"])])
    
                if len(stft_mean.shape) > 1 and stft_mean.shape[0] != Zxx.shape[1]:
                    mod_mean = einops.rearrange(stft_mean, 'mod freq ->freq mod')
                    mod_std = einops.rearrange(stft_std, 'mod freq ->freq mod')
                else:
                    mod_mean = stft_mean
                    mod_std = stft_std
                Zxx = (Zxx - mod_mean) / (mod_std + 1e-20)
                Zxx = einops.rearrange(Zxx, 'time freq mod-> mod freq time')
    
            for i in range(len(Zxx)):
                ax[count_mod + i].pcolormesh(t, f, Zxx[i], vmin=0, vmax=amp, shading='gouraud')
                ax[count_mod + i].axis("off")
                ax[count_mod + i].set_ylabel("channel_{}".format(i))
            count_mod += i + 1
        fig.text(0.5, 0.04, 'Time [hours]', ha='center')
        fig.text(0.04, 0.5, 'Frequency [Hz]', va='center', rotation='vertical')
        m = []
        for l in data[0]["label_intervals"]:
            if l[1] == "no_label":
                l[1] = -1
            m.append(int(l[1]) * np.ones(len(range(int(l[0][0] / srate), int(l[0][1] / srate)))))
        m = np.concatenate(m).ravel()
        ax[count_mod].set_xlim([0, len(m)])
        ax[count_mod].set_yticks([-1, 0, 1, 2, 3, 4])
        hours = len(m) / 60
        ax[count_mod].set_xticks([i * 60 * 60 for i in range(int((hours // 60) + 1))])
        ax[count_mod].set_xticklabels(["{}".format(i) for i in range(int((hours // 60) + 1))])
        ax[count_mod].set_yticklabels(["No Label", "Wake", "N1", "N2", "N3", "REM"])
        ax[count_mod].plot(m)
        for ex_id in range(len(data[0]["exclude_map"])):
            if (ex_id == 1):
                ax[count_mod].axvspan(0, data[0]["exclude_map"][ex_id][1], alpha=.5, color='green')
            elif (ex_id == len(data[0]["exclude_map"]) - 1):
                ax[count_mod].axvspan(data[0]["exclude_map"][ex_id][0], self.file_duration, alpha=.5, color='green')
            else:
                ax[count_mod].axvspan(data[0]["exclude_map"][ex_id][0], data[0]["exclude_map"][ex_id][1], alpha=.5, color='green')
        # lights = self.readSignal(12)
        # ax[count_mod+1].plot(lights)
        # ax[count_mod+1].set_yticks(np.unique(lights))
        plt.show()
    
    
    def plot_signals(self, data, patient_num, file_num):
        fig = plt.figure()
        num_rows = 1
        for modality in range(data["modalities"]):
            num_rows += data[modality]["num_channels"]
        num_cols = 1
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(wspace=0.05, hspace=0.15)
        ax = [plt.subplot(gs[i]) for i in range(num_rows * num_cols)]
        count_mod = 0
        for modality in range(data["modalities"]):
            srate = data[modality]["sampling_rate"]
            signals = np.array([data[modality][ch] for ch in range(data[modality]["num_channels"])])
            t = np.arange(len(signals[0]))
            for i in range(len(signals)):
                ax[count_mod + i].plot(t, signals[i])
                # ax[count_mod+i].axis("off")
                # ax[count_mod+i].set_ylabel("channel_{}".format(i))
                ax[count_mod + i].set_xticks([])
                ax[count_mod + i].set_yticks([0])
                if data[modality]["name_"] == 'o2':
                    ax[count_mod + i].set_yticklabels(["eeg_{}".format(i)])
                else:
                    ax[count_mod + i].set_yticklabels(["ecg"])
            count_mod += i + 1
        fig.text(0.5, 0.04, 'Time [sec]', ha='center')
        # fig.text(0.04, 0.5, 'Frequency [Hz]', va='center', rotation='vertical')
        m = []
        for l in data[0]["label_intervals"]:
            if l[1] == "no_label":
                l[1] = -1
            m.append(int(l[1]) * np.ones(len(range(int(l[0][0] / srate), int(l[0][1] / srate)))))
        m = np.concatenate(m).ravel()
        ax[count_mod].set_xlim([0, len(m)])
        ax[count_mod].set_yticks([-1, 0, 1, 2, 3, 4])
        ax[count_mod].set_xticks([])
        ax[count_mod].set_yticklabels(["No Label", "Non-Seizure", "Seizure", "N2", "N3", "REM"])
        ax[count_mod].plot(m)
        for ex_id in range(len(data[0]["exclude_map"])):
            if (ex_id == 1):
                ax[count_mod].axvspan(0, data[0]["exclude_map"][ex_id][1], alpha=.5, color='green')
            elif (ex_id == len(data[0]["exclude_map"]) - 1):
                ax[count_mod].axvspan(data[0]["exclude_map"][ex_id][0], self.file_duration, alpha=.5, color='green')
            else:
                ax[count_mod].axvspan(data[0]["exclude_map"][ex_id][0], data[0]["exclude_map"][ex_id][1], alpha=.5, color='green')
        # lights = self.readSignal(12)
        #
        # ax[count_mod+1].plot(lights)
        # ax[count_mod+1].set_yticks(np.unique(lights))
        fig.text(0.5, 0.9, "Patient {} file {}".format(patient_num, file_num), ha='center')
        plt.show()