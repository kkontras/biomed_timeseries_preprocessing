from abc import ABC, abstractmethod
import numpy as np

def Get_Discarder(type, file_duration):
    if type == "sleep_edf":
        return Discarder_Sleep_EDF(file_duration=file_duration)
    elif type == "shhs":
        return Discarder_SHHS()
    elif type == "seizit1":
        return Discarder_Seizit1(file_duration=file_duration)
    elif type == "neonatal":
        return Discarder_Neonatal(file_duration=file_duration)
    elif type == "nch" or type=="nch_total":
        return Discarder_Sleep_NCH(file_duration=file_duration)
    else:
        raise ValueError("Discarder for this dataset do not exists")

class Discarder(ABC):

    @abstractmethod
    def discard(self, data, patient_num, lights):
        raise NotImplementedError()

    def _contain_all_labels(self, data, num_classes):
        present_class = {}
        for i in range(num_classes):
            present_class[i] = {}
            present_class[i]["times"] = 0
            present_class[i]["p"] = 0
        total_misses = 0

        for interval in data[data["mod_name"][0]]["label_intervals"]:
            if interval[1]!='no_label':
                present_class[int(interval[1])]["times"] += 1
                present_class[int(interval[1])]["p"] += int(interval[0][1]-interval[0][0])
            else:
                total_misses +=1
                print("We no labels here for the part {}-{}".format(int(interval[0][1]),int(interval[0][0])))
        distribution = []
        for i in range(num_classes):
            if present_class[i]["times"] == 0:
                return None
            distribution.append(present_class[i]["p"])
        return np.array(distribution)/(data[data["mod_name"][0]]['sampling_rate']*30)


class Discarder_Sleep_NCH(Discarder):
    def __init__(self, file_duration):
        super().__init__()
        self.file_duration = file_duration

    def discard(self, data, patient_num, l):
        return True

class Discarder_Sleep_EDF(Discarder):
    def __init__(self, file_duration):
        super().__init__()
        self.file_duration = file_duration

    def discard(self, data, patient_num, lights):

        first_sleep, last_sleep = self.find_sleep_period(data)
        # self.exclude_map.append([0,30*60])
        # self.exclude_map.append([self.file_duration-30*60, self.file_duration])
        for mod in data["mod_name"]:
            data[mod]["exclude_map"].append([0,max(0, first_sleep-30*60*data[mod]["sampling_rate"])])
            data[mod]["exclude_map"].append([ min((last_sleep + 30*60*data[mod]["sampling_rate"]), self.file_duration*data[mod]["sampling_rate"]) , self.file_duration*data[mod]["sampling_rate"]])

        return True

    def find_sleep_period(self, data):

        for interval in data[data["mod_name"][0]]["label_intervals"]:
            if interval[1]!=0 and interval[1]!="no_label":
                first_sleep = interval[0][0]
                break
        for i in range(len(data[data["mod_name"][0]]["label_intervals"])-1, -1, -1):
            interval = data[data["mod_name"][0]]["label_intervals"][i]
            if interval[1]!=0 and interval[1]!="no_label":
                last_sleep = interval[0][1]
                break
        return first_sleep, last_sleep

class Discarder_SHHS(Discarder):
    def __init__(self):
        super()

    def discard(self, data, patient_num, lights):

        class_wind_distr = self._contain_all_labels(data=data, num_classes=5)
        if class_wind_distr is not None:
            if np.argmax(class_wind_distr) == 0:
                second_largest = np.argmax(class_wind_distr[1:]) + 1
                last_evening_W_length, first_morning_W_length = 0, 0
                if data[data["mod_name"][0]]["label_intervals"][0][1] == 0:
                    last_evening_W_length = data[data["mod_name"][0]]["label_intervals"][0][0][1] - \
                                            data[data["mod_name"][0]]["label_intervals"][0][0][0]
                if data[data["mod_name"][0]]["label_intervals"][-1][1] == 0:
                    first_morning_W_length = data[data["mod_name"][0]]["label_intervals"][0][0][1] - \
                                             data[data["mod_name"][0]]["label_intervals"][0][0][0]
                total_W_to_remove = (last_evening_W_length + first_morning_W_length) / (
                        30 * data[data["mod_name"][0]]["sampling_rate"]) - class_wind_distr[second_largest]
                # TODO: We could remove all last_venting and first_morning W even if they are not greater than the difference
                if total_W_to_remove > 0:
                    if (last_evening_W_length > total_W_to_remove):
                        for mod in data["mod_name"]:
                            data[mod]["exclude_map"].append(
                                [0, total_W_to_remove * (30 * data[mod]["sampling_rate"])])
                    else:
                        morning_W_to_remove = total_W_to_remove - last_evening_W_length
                        if first_morning_W_length - morning_W_to_remove < 0:
                            morning_W_to_remove = first_morning_W_length
                        for mod in range(data["modalities"]):
                            data[mod]["exclude_map"].append(
                                [0, last_evening_W_length * (30 * data[mod]["sampling_rate"])])
                            end_of_modality = self.file_duration * data[mod]["sampling_rate"]
                            data[mod]["exclude_map"].append(
                                [morning_W_to_remove * (30 * data[mod]["sampling_rate"]) - end_of_modality,
                                 end_of_modality])
                    print("W is greater than the other classes in patient {} ! Cutted {}".format(patient_num,
                                                                                                 total_W_to_remove))

            return True
        else:
            return False

class Discarder_Neonatal(Discarder):
    def __init__(self, file_duration):
        super().__init__()
        self.file_duration = file_duration

    def discard(self, data, patient_num, lights):
        return True

class Discarder_Seizit1(Discarder):
    def __init__(self, file_duration):
        super()

    def discard(self, data, patient_num, lights):
        for mod in range(data["modalities"]):
           #If it is less than 20 minutes
           if len(data[mod][0]) < data[mod]["sampling_rate"]*20:
               return False
        distribution = self._contain_all_labels(data, 2)
        if distribution == None: return False
        return True


def skip_minutes(self, lights_off, keep_mins):
    # self.exclude_map.append([0,30*60])
    # self.exclude_map.append([self.file_duration-30*60, self.file_duration])
    #
    # for modality in range(data["modalities"]):
    print("{}:{}:{}".format(self.starttime_hour, self.starttime_minute, self.starttime_second))
    print(lights_off)
    sum = - self.starttime_hour * 3600 - 60 * self.starttime_minute - self.starttime_second
    sum += lights_off.hour * 3600 + 60 * lights_off.minute + lights_off.second

    if (self.starttime_hour > lights_off.hour):
        sum += 3600 * 24

    # for i, interval in enumerate(self.label_intervals):
    #     if interval[1] != 'Sleep stage W' and interval[1] != 'Sleep stage ?' and interval[1] != 'Movement time':
    #         max_i = i
    #
    # keep_mins = 15
    # for i in range(max_i,len(self.label_intervals)):
    #     interval = self.label_intervals[i]
    #     if interval[1] == 'Sleep stage W':
    #         max = {"i": i, "start": interval[0][0], "end": interval[0][1]}
    #         break

    if sum > keep_mins * 60:
        self.exclude_map.append([0, sum - keep_mins * 60])

    # if self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_48/SC4481F0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[60930+(keep_mins+15)*60, 80220 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_76/SC4762E0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[70080+(keep_mins+15)*60, 82860 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_75/SC4751E0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[52650+(keep_mins+15)*60, 73590 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_48/SC4481F0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[60930+(keep_mins+15)*60, 80220 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_48/SC4482F0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[58080+(keep_mins+15)*60, 82350 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_65/SC4651E0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[61560+(keep_mins+15)*60, 76770 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_62/SC4622E0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[65940+(keep_mins+15)*60, 71760 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_74/SC4741E0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[55440+(keep_mins+15)*60, 74580 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_59/SC4591G0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[59190+(keep_mins+15)*60, 78030 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_47/SC4472F0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[54600+(keep_mins+15)*60, 79710 - (keep_mins+15)*60]]
    # elif self.file_name == '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_66/SC4661E0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[55500+(keep_mins+15)*60, 75390 - (keep_mins+15)*60]]
    # elif self.file_name ==  '/esat/smcdata/users/kkontras/sleep_edf78_data/patient_66/SC4662E0-PSG.edf':
    #     self.exclude_map = [[0, sum - keep_mins*60],[61140+(keep_mins+15)*60, 80130 - (keep_mins+15)*60]]

    # if max["start"]+ (keep_mins+15)*60 < self.file_duration:
    #     self.exclude_map.append([max["start"]+(keep_mins+15)*60, self.file_duration])

    # if 7070000<max["start"]+ (keep_mins+15)*60:
    #     print("Something here is wrong")
    # if len(self.exclude_map)<2:
    #     print("Something here is wrong")

    flag_end = False
    for i, interval in enumerate(self.label_intervals):
        if interval[1] == 'Sleep stage W' and interval[0][0] != 0:
            if interval[0][1] - interval[0][0] > 3600:
                if interval[0][1] + (keep_mins + 15) * 60 > self.file_duration:
                    begin = interval[0][0] + (keep_mins + 15) * 60 if interval[0][0] + (
                                keep_mins + 15) * 60 < self.file_duration else interval[0][0]
                    self.exclude_map.append([begin, self.file_duration])
                    flag_end = True
                    break
                else:
                    # self.exclude_map.append([interval[0][0] + (keep_mins + 15) * 60, interval[0][1] - (keep_mins + 15) * 60])
                    max = {"i": i, "start": interval[0][0] + (keep_mins + 15) * 60, "end": interval[0][1]}
            elif interval[0][1] + (keep_mins + 15) * 60 > self.file_duration:
                begin = interval[0][0] + (keep_mins + 15) * 60 if interval[0][0] + (
                        keep_mins + 15) * 60 < self.file_duration else interval[0][0]
                flag_end = True
                self.exclude_map.append([begin, self.file_duration])

                break
    if not flag_end:
        self.exclude_map.append([max["start"], self.file_duration])

    #
    # length_in_minutes = self.calculate_actual_length(self.exclude_map, 1)
    #
    # if (length_in_minutes/3600)>10:
    #     print(self.exclude_map)
    #     print("Something here is wrong")

    # from datetime import timedelta
    # true_time = timedelta(hours=self.starttime_hour,minutes=self.starttime_minute,seconds=self.starttime_second)
    # true_time_2 = timedelta(hours=(max["start"]+ (keep_mins+15)*60)//3600,minutes=((max["start"]+ (keep_mins+15)*60)%3600)//60,seconds=self.starttime_second)
    # print(true_time+true_time_2)

    # return max["start"]


def skip_minutes_shhs_signals(self, keep_mins):
    ch_num = False
    for i, ch in enumerate(self.headers):
        if 'LIGHT' == ch['label']:
            ch_num = i
    if not ch_num:
        return
    lights = self.readSignal(ch_num)
    min_max = {"min": 0, "max": self.file_duration}
    first_time_lights_off = True
    sum = 0
    for i, l in enumerate(lights):
        # keep only the first and last time lights were off
        if l == 1:
            if first_time_lights_off:
                min_max["min"] = i
                first_time_lights_off = False
            min_max["max"] = i
            sum += 1
    # This threshold 200 is a random choice.
    if sum < 200 or first_time_lights_off:
        min_max["min"] = self.file_duration

    if min_max["min"] > keep_mins * 60:
        min_max["min"] = min_max["min"] - keep_mins * 60
    else:
        min_max["min"] = 0

    if min_max["max"] < self.file_duration - keep_mins * 60:
        min_max["max"] = min_max["max"] + keep_mins * 60
    else:
        min_max["max"] = self.file_duration
    # print(min_max)
    for modality in range(data["modalities"]):
        for ch in range(data[modality]["num_channels"]):
            data[modality][ch] = data[modality][ch][
                                         min_max["min"] * data[modality]["sampling_rate"]:min_max[
                                                                                                      "max"] *
                                                                                                  data[
                                                                                                      modality][
                                                                                                      "sampling_rate"]]

def skip_minutes_signals(self, lights_off, keep_mins):
            """
            #TODO: Instead of using this one, get a function that finds the final exclude map and reduce the signal based on that.
            This function is used to reduce the signals only for the get_mean and std. Ultimately we should use the exclude map to crop the signal based on the final exclude map.
            :return:
            """

            sum = - self.starttime_hour * 3600 - 60 * self.starttime_minute - self.starttime_second
            sum += lights_off.hour * 3600 + 60 * lights_off.minute - lights_off.second

            if (self.starttime_hour > lights_off.hour):
                sum += 3600 * 24

            if sum > keep_mins * 60:
                sum = sum - keep_mins * 60
            else:
                sum = 0

            exclude_signal = []
            for i, interval in enumerate(self.label_intervals):
                if interval[1] == 'Sleep stage W' and interval[0][0] != 0:
                    if interval[0][1] - interval[0][0] > 3600:
                        if interval[0][1] + (keep_mins + 15) * 60 > self.file_duration:
                            begin = interval[0][0] + (keep_mins + 15) * 60 if interval[0][0] + (
                                        keep_mins + 15) * 60 < self.file_duration else interval[0][0]
                            exclude_signal.append([begin, self.file_duration])
                            break
                        else:
                            exclude_signal.append(
                                [interval[0][0] + (keep_mins + 15) * 60, interval[0][1] - (keep_mins + 15) * 60])
                    elif interval[0][1] + (keep_mins + 15) * 60 > self.file_duration:
                        begin = interval[0][0] + (keep_mins + 15) * 60 if interval[0][0] + (
                                keep_mins + 15) * 60 < self.file_duration else interval[0][0]

                        exclude_signal.append([begin, self.file_duration])
                        break

            for modality in range(data["modalities"]):
                for ch in range(data[modality]["num_channels"]):
                    total_signal = [data[modality][ch][
                                    (int(sum)) * data[modality]["sampling_rate"]:(int(exclude_signal[0][0])) *
                                                                                         data[modality][
                                                                                             "sampling_rate"]]]
                    for i in range(len(exclude_signal) - 1):
                        total_signal.append(data[modality][ch][
                                            (int(exclude_signal[i][1])) * data[modality]["sampling_rate"]:(
                                                                                                                      int(
                                                                                                                          exclude_signal[
                                                                                                                              i + 1][
                                                                                                                              0])) *
                                                                                                                  data[
                                                                                                                      modality][
                                                                                                                      "sampling_rate"]])
                    data[modality][ch] = np.concatenate(total_signal)
