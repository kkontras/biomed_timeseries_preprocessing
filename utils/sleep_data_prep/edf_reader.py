from pyedflib import EdfReader
from scipy import signal as sg
import numpy as np
import json
import einops
from utils.helpers.Extract_Transform_Labeler import Get_Labeler
from utils.helpers.Save_Agent import Get_Savior
from utils.helpers.Window_Labeler import Get_Window_Labeler
from utils.helpers.Minute_Discarder import Get_Discarder
from utils.helpers.Additional_Operator import Get_Additional_Operator
from utils.helpers.Plotter import Plotter
import matlab.engine

# eng = matlab.engine.start_matlab()


class EDF_Reader(EdfReader):
    def __init__(self, config, edf_file, label_file, name, patient_num, file_num, mod_names):
        super().__init__(edf_file)
        self._file = edf_file
        self.config = config
        self.patient_num = patient_num
        self.file_num = file_num
        self.headers = self.getSignalHeaders()
        self.ch_labels = [self.headers[i]["label"] for i in range(len(self.headers))]
        self.ch_dict = {}
        self.name = name
        self.exclude_map = []
        self.mod_names = mod_names

        #
        self.extract_transform_labeler = Get_Labeler(type=self.config.dataset_type, label_file=label_file, label_mapping=self.config.label_dict, file_duration=self.file_duration)
        self.save_agent = Get_Savior(save_type=self.config.save_type, savedir=self.config.save_dir, dataset_name=self.config.dataset_type,
                                     test_patients=self.config.test_patients, patient_num=patient_num, file_num=file_num, patient_name=name)
        # self.augmentor = Get_Augmentor()
        self.window_labeler = Get_Window_Labeler(b_policy=self.config.b_policy, file_duration = self.file_duration, num_classes=self.config.num_classes)
        self.plotter = Plotter(file_duration=self.file_duration, config=self.config)
        self.discarder = Get_Discarder(type=self.config.dataset_type, file_duration=self.file_duration)
        self.additional_operator = Get_Additional_Operator(config=self.config,type=self.config.dataset_type)

    def _get_channels(self, group_label_json,  channel_labels):
        for channel in channel_labels:
            possible_labels = group_label_json["electrode_names"][channel]
            for p_label in possible_labels:
                for labels_index in range(len(self.ch_labels)):
                    if p_label == self.ch_labels[labels_index].lower():
                        self.ch_dict[channel] = labels_index
                        break

    def _set_mod_intervals(self, label_intervals, exclude_map):
        """
        Set label intervals and excluded intervals on every modality w.r.t. sampling rate of each one.

        :param label_intervals: List of label intervals, each one has the type [[start,end],label_int]
        :param exclude_map: List of excluded intervals, each one has the type [start,end]
        """
        for modality in self.signals["mod_name"]:
            self.signals[modality]["label_intervals"] = []
            self.signals[modality]["exclude_map"] = []
            for interval in label_intervals:
                self.signals[modality]["label_intervals"].append([[interval[0][0] * self.signals[modality]["sampling_rate"],interval[0][1] * self.signals[modality]["sampling_rate"]], interval[1]])
            for interval in exclude_map:
                self.signals[modality]["exclude_map"].append([[interval[0] * self.signals[modality]["sampling_rate"],interval[1] * self.signals[modality]["sampling_rate"]]])

    def _calculate_actual_length(self, exclude_map, rate):
        removed = 0
        for interval in exclude_map:
            removed += interval[1]-interval[0]
        total = self.file_duration - (removed/rate)
        if total ==0:
            print("Here we are at 0")
        return total

    def check_edf_labels(self, edf_labels, type):
        # try:
        with open(edf_labels) as json_file:

            group_label_json = json.load(json_file)
            labels = group_label_json["electrode_types"][type]["headers"]
            self.modalities = group_label_json["electrode_types"][type]["modalities"]
            self._get_channels(group_label_json, labels)
            if hasattr(self,"desire_channels"):
                self.desire_channels.append(labels)
            else:
                self.desire_channels = labels
        # except:
        #     raise Exception("Something is wrong with the file and the labels.")

    def verify_edf_channels(self, req):
        if len(self.ch_dict) < req:
            print(self.ch_labels)
            raise Warning("There might be some missing channels, we got {} while required is {}. \n file is {}".format(len(self.ch_dict), req ,self._file))
            return False
        return True

    def extract_transform_labels(self):
        """
        Extract the labels coming from file, transform them according to map_file and return them as part of each signal modality.

        :param file: Label file, supported types can be found in the extractor init
        :param map_file: Mapping of string labels to int numbers + definition of background
        :return: The self.signal[modality]["label_interval"] and "exclude_map" w.r.t. sampling rate
        """

        # Get the labels and any intervals that should be excluded
        label_intervals, exclude_map = self.extract_transform_labeler.get_labels()

        #End labelling is used in case we have not labelled the whole signal
        end_labelling = self.end_labelling if hasattr(self, "end_labelling") else None

        # Map labels from string to numbers (or any other form)
        label_intervals = self.extract_transform_labeler.transform_labels(label_intervals=label_intervals, end_labelling=end_labelling)

        # Set intervals with the correct sampling rate for every modality
        self._set_mod_intervals(label_intervals=label_intervals, exclude_map=exclude_map)

    def load_signals(self, ):
        self.signals = {"modalities":len(self.modalities), "mod_name":self.mod_names}
        count = 0
        for i, num_channels in enumerate(self.modalities):
            mod_name = self.signals["mod_name"][i]
            self.signals[mod_name] = {}

            for ch_idx in range(count, count + num_channels):
                self.signals[mod_name][ch_idx - count] = self.readSignal(self.ch_dict[self.desire_channels[ch_idx]])
                self.signals[mod_name]["name_".format(ch_idx-count)] = self.desire_channels[ch_idx]
                self.signals[mod_name]["sampling_rate"] = self.getSampleFrequency(self.ch_dict[self.desire_channels[ch_idx]])
                if self.signals[mod_name]["sampling_rate"]==3000 or self.signals[mod_name]["sampling_rate"]== 6000 :
                    #This is for Sleep-EDF, it seems it contains a mistake on the sampling frequency.
                    self.signals[mod_name]["sampling_rate"] = 100
                self.signals[mod_name]["num_channels"] = num_channels
            count += num_channels

    def resample(self, to_srate):

        for modality in self.signals["mod_name"]:
            for ch in range(self.signals[modality]["num_channels"]):
                self.signals[modality][ch] = sg.resample_poly( self.signals[modality][ch], down=self.signals[modality]["sampling_rate"], up=to_srate[modality])
            label_intervals, exclude_map = [], []
            mod_rate = to_srate[modality]/self.signals[modality]["sampling_rate"]
            for interval in self.signals[modality]["label_intervals"]:
                #SHHS SLEEPEDF
                label_intervals.append([[int(interval[0][0] * mod_rate),int(interval[0][1] * mod_rate)], interval[1]])
                #Neonatal
                # self.signals[modality]["label_intervals"].append([[interval[0] * to_srate,interval[1] * to_srate ], interval[2]])
            for interval in self.signals[modality]["exclude_map"]:
                exclude_map.append( [int(interval[0] * mod_rate), int(interval[1] * mod_rate)])
            self.signals[modality]["label_intervals"] = label_intervals
            self.signals[modality]["exclude_map"] = exclude_map
            self.signals[modality]["sampling_rate"] = to_srate[modality]

        self.srate = to_srate

    def create_windows(self, mean= None, std = None ,enhancement={"Gaussian":{"mean":0,"std":0.2,"times":5}}):


        # self.signal_length = len(self.signals[self.signals["mod_name"][0]])

        # self.enhance_signals(enhancement)
        metric= {"mean_time":{},"mean_sq_time":{},"sum_time":{},"count_labels":np.zeros(self.config.num_classes)}
        # metrics = {"mean":np.zeros([129,]),"mean_sq":np.zeros([129]),"sum":0,"count_labels":np.zeros([5])}

        count_signals = 0
        for modality in self.signals["mod_name"]:
            window_samples = self.config.window_time * self.signals[modality]["sampling_rate"]
            # These logs are used to ensure that we got the correct number of windows and labels.
            self.signals[modality]["windows"] = {i:[] for i in range(self.config.ch_per_mod[modality])}
            self.signals[modality]["w_labels"] =  {i:[] for i in range(self.config.ch_per_mod[modality])}
            self.signals[modality]["w_inits"] =  {i:[] for i in range(self.config.ch_per_mod[modality])}

            #This is used to ensured that it is aligned with the stft view.
            self.signals[modality]["starts"] =  {i:[] for i in range(self.config.ch_per_mod[modality])}

            start_index = 0
            flag = False #init flag
            len_signal = len(self.signals[modality][0])
            length_in_minutes = self._calculate_actual_length(self.signals[modality]["exclude_map"],self.signals[modality]["sampling_rate"])
            signals = np.array([self.signals[modality][ch] for ch in range(self.signals[modality]["num_channels"])])

            if mean and std:
                for ch in range(self.signals[modality]["num_channels"]):
                    assert isinstance(mean[modality][ch], (int, float)) and isinstance(std[modality][ch], (int, float))
                    signals[ch] = (signals[ch] - mean[modality][ch]) / (np.array(std[modality][ch]) + 1e-20)

            while( start_index + window_samples <= len_signal ):

                current_window_signal = signals[:,int(start_index): int(start_index) + window_samples]
                flag, init_label, end_of_discarded, w_label = self.window_labeler.check_find_window_label(window=[int(start_index), int(start_index) + window_samples],
                                                                                                          data= self.signals[modality],
                                                                                                          previous_flag=flag)
                if flag:
                    if not isinstance(w_label, np.ndarray) and w_label == "no_label":
                        #Set new start index and continue without saving this window
                        start_index = int(start_index + self.signals[modality]["sampling_rate"] * self.config.inner_stft_window * self.config.overlap_off)
                        continue
                    for ch in range(self.config.ch_per_mod[modality]):
                        self.signals[modality]["windows"][ch].append(current_window_signal[ch])
                        self.signals[modality]["w_labels"][ch].append(w_label)
                        self.signals[modality]["w_inits"][ch].append(init_label)
                        self.signals[modality]["starts"][ch].append([w_label, init_label, start_index])
                    #For binary problems mainly
                    overlap = self.config.overlap_off
                    #Progress start_index by a window length according to overlap
                    start_index =int(start_index + self.signals[modality]["sampling_rate"] * self.config.inner_stft_window * overlap)
                else:
                    # Set new start index to the end of the discarded period
                    if self.config.inner_stft_window:
                        #Make sure there is alignment to the quant that stft takes -> inner_stft_window
                        start_index = int((end_of_discarded/(self.config.inner_stft_window*self.signals[modality]["sampling_rate"])))*self.config.inner_stft_window*self.signals[modality]["sampling_rate"]
                    else:
                        start_index = int(end_of_discarded)


            count_signals += self.signals[modality]["num_channels"]
            for ch in range(self.config.ch_per_mod[modality]):
                self.signals[modality]["windows"][ch] = np.array(self.signals[modality]["windows"][ch])
                self.signals[modality]["w_labels"][ch] = np.array(self.signals[modality]["w_labels"][ch])
                self.signals[modality]["w_inits"][ch] = np.array(self.signals[modality]["w_inits"][ch])
                if len(self.signals[modality]["w_labels"][ch]) != len(self.signals[modality]["windows"][ch]):
                    raise Exception("labels and windows dont match")

            #Gather mean and std metrics while processing - One pass algorithm uses mean and mean_square. See the gathering function for the rest.
            metric["mean_time"][modality], metric["mean_sq_time"][modality], metric["sum_time"][modality] = {}, {}, {}
            for ch in range(self.config.ch_per_mod[modality]):
                if len(self.signals[modality]["windows"][ch].shape)<2:
                    print(self.signals[modality]["windows"][ch].shape)
                    metric["sum_time"][modality][ch] = self.signals[modality]["windows"][ch].shape[0]
                    if metric["sum_time"][modality][ch]==0:
                        metric["mean_time"][modality][ch] = np.zeros(1)
                        metric["mean_sq_time"][modality][ch] = np.zeros(1)
                    else:
                        metric["mean_time"][modality][ch] = self.signals[modality]["windows"][ch].mean()
                        metric["mean_sq_time"][modality][ch] = np.square(self.signals[modality]["windows"][ch]).mean()
                else:
                    metric["sum_time"][modality][ch] = self.signals[modality]["windows"][ch].shape[0]*self.signals[modality]["windows"][ch].shape[1]
                    metric["mean_time"][modality][ch] = self.signals[modality]["windows"][ch].mean()
                    metric["mean_sq_time"][modality][ch] = np.square(self.signals[modality]["windows"][ch]).mean()
        for l in self.signals[modality]["w_labels"][0]:
            metric["count_labels"][np.argmax(l)] += 1
            if not (1 in l):
                metric["count_labels"][-1] += 1

        if 1 in self.signals.keys() and len(self.signals[0]["windows"][0])>0 and len(self.signals[0]["windows"][0]) != len(self.signals[1]["windows"][0]):
            raise Exception("Windows between modalities do not match")

        return length_in_minutes, metric

    def create_sfft_windows(self,  mean, std, enhancement={"Gaussian":{"mean":0,"std":0.2, "times":5 }}, imp_label=1, hdf5_file = None):

        metric= {"mean_stft":{},"mean_sq_stft":{},"sum_stft":{}}
        window_samples = int(self.config.window_time / self.config.inner_stft_window)

        for modality in self.signals["mod_name"]:
            # These logs are used to ensure that we got the correct number of windows and labels.
            self.signals[modality]["windows_stft"] = {i:[] for i in range(self.config.ch_per_mod[modality])}
            self.signals[modality]["w_labels_stft"] = {i:[] for i in range(self.config.ch_per_mod[modality])}
            self.signals[modality]["w_inits_stft"] =  {i:[] for i in range(self.config.ch_per_mod[modality])}
            self.signals[modality]["starts_stft"] = {i:[] for i in range(self.config.ch_per_mod[modality])}

            w_stft, w_labels, start_index= [], [], 0
            srate = self.signals[modality]["sampling_rate"]
            signals = np.array([self.signals[modality][ch] for ch in range(self.signals[modality]["num_channels"])])

            len_signal =len(signals[0])
            flag = False #init flag

            # #V2
            # window_samples = window_samples*srate
            # while( start_index + window_samples <= len_signal ):
            #
            #     current_window_signal = signals[:,int(start_index): int(start_index) + window_samples]
            #     flag, init_label, end_of_discarded, w_label = self.window_labeler.check_find_window_label(window=[int(start_index), int(start_index) + window_samples],
            #                                                                                               data= self.signals[modality],
            #                                                                                               previous_flag=flag)
            #     if flag:
            #         if w_label == "no_label":
            #             #Set new start index and continue without saving this window
            #             start_index = int(start_index + self.signals[modality]["sampling_rate"] * self.config.inner_stft_window * self.config.overlap_off)
            #             continue
            #         for ch in range(self.config.ch_per_mod[modality]):
            #             Zxx = np.asarray(
            #                 eng.spectrogram(matlab.double(current_window_signal[ch].tolist()),
            #                                 matlab.double(eng.hamming(2 * srate)), srate, self.config.nfft_points))
            #
            #             Zxx = np.abs(Zxx)
            #             Zxx[Zxx == 0] = 1e-5
            #             Zxx = 20 * np.log10(Zxx)
            #             if mean and std:
            #                 Zxx = einops.rearrange(Zxx, 'freq time -> time freq')
            #                 stft_mean = np.array(mean[modality][ch])
            #                 stft_std = np.array(std[modality][ch])
            #
            #                 if len(stft_mean.shape) > 1 and stft_mean.shape[0] != Zxx.shape[1]:
            #                     raise ValueError("Mean and std have fault array shapes")
            #                 else:
            #                     mod_mean = stft_mean
            #                     mod_std = stft_std
            #                 Zxx = (Zxx - mod_mean) / (mod_std + 1e-20)
            #                 Zxx = einops.rearrange(Zxx, 'time freq->freq time')
            #
            #             self.signals[modality]["windows_stft"][ch].append(Zxx)
            #             self.signals[modality]["w_labels_stft"][ch].append([w_label, init_label])
            #             self.signals[modality]["starts_stft"][ch].append([w_label, init_label, start_index])
            #
            #         #For binary problems mainly
            #         overlap = self.config.overlap_on if w_label==1 else self.config.overlap_off
            #         #Progress start_index by a window length according to overlap
            #         start_index =int(start_index + self.signals[modality]["sampling_rate"] * self.config.inner_stft_window * overlap)
            #     else:
            #         print("yo")
            #         # Set new start index to the end of the discarded period
            #         if self.config.inner_stft_window:
            #             #Make sure there is alignment to the quant that stft takes -> inner_stft_window
            #             start_index = int((end_of_discarded/(self.config.inner_stft_window*self.signals[modality]["sampling_rate"]))+1)*self.config.inner_stft_window*self.signals[modality]["sampling_rate"]
            #         else:
            #             start_index = int(end_of_discarded+1)
            # if "starts" in self.signals[modality] and "starts_stft" in self.signals[modality] and self.signals[modality]["starts_stft"] != self.signals[modality]["starts"]:
            #     print("There is no exact alignment between time and stft views")


            #V2

            f, t, Zxx = sg.stft(signals, srate, nperseg=2*srate, noverlap=srate, nfft=self.config.nfft_points, window=sg.windows.hamming(2*srate), boundary=None)
            # Zxx_mat = np.concatenate([np.expand_dims(np.asarray(eng.spectrogram(matlab.double(signals[i, :].tolist()), matlab.double(eng.hamming(2*self.config.inner_stft_window*srate)), self.config.inner_stft_window*srate, self.config.nfft_points)),axis=0) for i in range(len(signals))],axis=0)

            Zxx = np.abs(Zxx)
            Zxx[Zxx == 0] = 1e-5
            Zxx = 20 * np.log10(Zxx)
            if (Zxx != Zxx).any():
                print(Zxx.min())
                print(np.isnan(Zxx).any())
            #
            if mean and std:
                for ch in range(self.signals[modality]["num_channels"]):
                    assert isinstance(mean[modality][ch], (int, float)) and isinstance(std[modality][ch], (int, float))
                    signals[ch] = (signals[ch] - mean[modality][ch]) / (np.array(std[modality][ch]) + 1e-20)

            while ((start_index + window_samples)*self.config.inner_stft_window*srate <= len_signal):

                flag, init_label, end_of_discarded, w_label = self.window_labeler.check_find_window_label(window=[int(start_index)*self.config.inner_stft_window*srate, int(start_index + window_samples)*self.config.inner_stft_window*srate],
                                                                                                          data=self.signals[modality],
                                                                                                          previous_flag=flag)
                flag_label = True
                if not isinstance(w_label, np.ndarray) and w_label == "no_label":
                    flag_label = False
                if flag:
                    if not flag_label:
                        start_index = start_index + self.config.overlap_off
                        continue
                    overlap = self.config.overlap_off
                    for ch in range(self.config.ch_per_mod[modality]):
                        self.signals[modality]["windows_stft"][ch].append(Zxx[ch, :, int(start_index): int(start_index + window_samples -1)])
                        self.signals[modality]["w_labels_stft"][ch].append(w_label)
                        self.signals[modality]["w_inits_stft"][ch].append(init_label)
                        self.signals[modality]["starts_stft"][ch].append([w_label, init_label, int(start_index)*self.config.inner_stft_window*srate])
                    start_index = start_index + overlap

                else:
                    start_index = int((end_of_discarded/(srate*self.config.inner_stft_window)))


            # if "starts" in self.signals[modality] and "starts_stft" in self.signals[modality] and self.signals[modality]["starts_stft"] != self.signals[modality]["starts"]:
            #     print("There is no exact alignment between time and stft views")

            for ch in range(self.config.ch_per_mod[modality]):
                if len(np.array(self.signals[modality]["windows_stft"][ch]).shape)<2:
                    self.signals[modality]["windows_stft"][ch] = np.array(np.zeros(0))
                    self.signals[modality]["w_labels_stft"][ch] = np.array(np.zeros(0))
                    self.signals[modality]["w_inits_stft"][ch] = np.array(np.zeros(0))

                    # self.signals[modality]["windows_stft"][ch] = np.array(self.signals[modality]["windows_stft"][ch])
                    # self.signals[modality]["w_labels_stft"][ch] = np.array(self.signals[modality]["w_labels_stft"][ch])
                    # self.signals[modality]["w_inits_stft"][ch] = np.array(self.signals[modality]["w_inits_stft"][ch])
                else:
                    self.signals[modality]["windows_stft"][ch] = np.array(self.signals[modality]["windows_stft"][ch])
                    self.signals[modality]["w_labels_stft"][ch] = np.array(self.signals[modality]["w_labels_stft"][ch])
                    self.signals[modality]["w_inits_stft"][ch] = np.array(self.signals[modality]["w_inits_stft"][ch])

            metric["mean_stft"][modality], metric["sum_stft"][modality], metric["mean_sq_stft"][modality] = {}, {}, {}
            for ch in range(self.config.ch_per_mod[modality]):
                if ch not in self.signals[modality]["windows_stft"] or (ch in self.signals[modality]["windows_stft"] and len(self.signals[modality]["windows_stft"][ch].shape)<2):
                    print(self.patient_num)
                    print(self.signals[modality]["windows_stft"][ch].shape)
                    break
                mean_stft = self.signals[modality]["windows_stft"][ch].mean(axis=(0,2))
                mean_stft_sq = np.square(self.signals[modality]["windows_stft"][ch]).mean(axis=(0,2))
                file_lenth = self.signals[modality]["windows_stft"][ch].shape[0]*self.signals[modality]["windows_stft"][ch].shape[-1]
                metric["sum_stft"][modality][ch] = file_lenth
                metric["mean_stft"][modality][ch] = mean_stft
                metric["mean_sq_stft"][modality][ch] = mean_stft_sq

        if 1 in self.signals.keys() and len(self.signals[0]["windows_stft"][0])>0 and len(self.signals[0]["windows_stft"][0]) != len(self.signals[1]["windows_stft"][0]):
            print("This is strange")

        return metric

    def plot_data(self, view, mean, std):
        if self.config.show_plot == False:
            return
        if view == "time":
            self.plotter.plot_signals(data=self.signals, mean=mean, std=std)
        elif view == "stft":
            self.plotter.plot_stft(data=self.signals, mean=mean, std=std)
        else:
            raise ValueError("There is no plot operation for this view {}".format(view))

    def discard_parts(self):
        #TODO: Try to get the lights if they exist
        lights = None

        # if self.config.skip_30_mins:
        #     if self.config.types[0] == "shhs":
        #         file.reader.skip_minutes_shhs_signals(self.config.keep_mins)
        #         # pass
        #     else:
        #         try:
        #             lights_off_time = self.lights_off.loc[
        #                 (self.lights_off['subject'] == int(real_p_name[-2:])) & (
        #                         self.lights_off['night'] == file_num + 1), 'LightsOff'].values[0]
        #         except:
        #             lights_off_time = self.lights_off.loc[
        #                 (self.lights_off['subject'] == int(real_p_name[-2:])) & (
        #                         self.lights_off['night'] == file_num + 2), 'LightsOff'].values[0]
        #         file.reader.skip_minutes_signals(lights_off_time, self.config.keep_mins)
        # if len(file.reader.signals[0][0]) < 1600:
        #     print("This one was skipped")
        #     continue

        return self.discarder.discard(self.signals, self.patient_num, lights)

    def additional_dataset_operations(self):
        self.signals = self.additional_operator.operations(self.signals, self.ch_dict)

    def save_windows(self, view, metrics):
        if view == "stft":
            metrics = self.save_agent.save_windows_stft(data=self.signals, metrics=metrics)
        elif view == "time":
            metrics = self.save_agent.save_windows(data=self.signals, metrics=metrics)

        return metrics

    def sign_mem_free(self):
        try:
            del self.signals, self.f_labels, self.w_labels, self.s_windows, self.srate, self.r_labels
        except:
            pass

    def filter_signals(self, filt_type, order, cutoff):

        for modality in self.signals["mod_name"]:

            if filt_type[modality]["impulse"] == "butter":

                nyq = 0.5 * self.signals[modality]["sampling_rate"]
                sos = sg.butter(order, [cutoff[modality][0]/ nyq, cutoff[modality][1]/ nyq], btype='band', output="sos")
                for ch in range(self.signals[modality]["num_channels"]):
                    self.signals[modality][ch]= sg.sosfiltfilt(sos, self.signals[modality][ch])

            elif filt_type[modality]["impulse"] == "fir":
            # b, a = sg.butter(order, [cutoff[0]/ nyq, cutoff[1]/ nyq], btype='band')
            # self.signals= np.array([sg.filtfilt(b, a, signal) for signal in self.signals])
                if len(cutoff[modality])<2:
                    coef = sg.firwin(fs=self.signals[modality]["sampling_rate"], numtaps=order[modality], cutoff=cutoff[modality], window="hamming", pass_zero=filt_type[modality]["type"])
                    #check these coef for emg
                else:
                    if self.signals[modality]["sampling_rate"]/2 <= cutoff[modality][1]:
                        # print(self.signals[modality]["sampling_rate"])
                        cutoff[modality][1] = (self.signals[modality]["sampling_rate"]/2) - 1
                    coef = sg.firwin(fs=self.signals[modality]["sampling_rate"], numtaps=order[modality], cutoff=cutoff[modality], window="hamming", pass_zero=filt_type[modality]["type"])
                for ch in range(self.signals[modality]["num_channels"]):
                    if len(self.signals[modality][ch])<1600:
                        return
                    self.signals[modality][ch] = sg.filtfilt(coef, [1], self.signals[modality][ch], padtype = 'odd', padlen=3*(max(len([1]),len(coef))-1))

    def _get_thse_lights_off(self, filename):
        self.lights_off = pd.read_excel(filename)

