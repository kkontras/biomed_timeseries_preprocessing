from abc import ABC, abstractmethod
import numpy as np
import warnings
def Get_Additional_Operator(config, type):
    if type == "sleep_edf":
        return Add_Operator_Sleep_EDF(config)
    elif type == "shhs":
        return Add_Operator_SHHS(config)
    elif type == "seizit1":
        return Add_Operator_Seizit1(config)
    elif type == "neonatal":
        return Add_Operator_Neonatal(config)
    elif type == "nch" or type=="nch_total":
        return Add_Operator_NCH(config)
    else:
        warnings.warn("There is no additional Operator for this dataset, so we assign an empty one")
        return Empty_Add_Operator(config)

class Additional_Operator(ABC):
    @abstractmethod
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def operations(self, data, ch_dict):
        raise NotImplementedError()

class Empty_Add_Operator(Additional_Operator):
    def __init__(self, config):
        super().__init__(config)

    def operations(self, data, ch_dict):
        return data

class Add_Operator_Sleep_EDF(Additional_Operator):
    def __init__(self, config):
        super().__init__(config)

    def operations(self, data, ch_dict):
        return data

class Add_Operator_SHHS(Additional_Operator):
    def __init__(self, config):
        super().__init__(config)

    def operations(self, data, ch_dict):
        return data

class Add_Operator_NCH(Additional_Operator):
    def __init__(self, config):
        super().__init__(config)

    def operations(self, data, ch_dict):
        # ΤΟDO: We need to handle somehow patients that do not have any modality at all (do not save them)
        # Also in the case that we have some modalities or channels we need to save the ground truth correctly and somehow notice that these windows are missing entirely on this modality.
        return data

class Add_Operator_Neonatal(Additional_Operator):
    def __init__(self, config):
        super().__init__(config)
    def _max_consec_zeros(self, data):
        outs = []
        for ch_num in data[0].keys():
            if not isinstance(ch_num, (int, float)): continue
            sig = data[0][ch_num]
            if (np.count_nonzero(sig == 0) > 0):
                m1 = np.r_[False, sig == 0, False]
                idx = np.flatnonzero(m1[:-1] != m1[1:])
                diff = (idx[1::2] - idx[::2])
                for ind, d  in enumerate(diff):
                    if (d > 15):
                        outs.append([idx[ind * 2], idx[ind * 2+1]])
        return outs
        # outs = outs + self.exclude_map
        # outs.sort()
        # outs = list(k for k, _ in itertools.groupby(outs))
        # outs = np.array(outs)
        # self.exclude_map = self._merge_time_frames(outs)

    def _reference_out(self, ch_dict, data):
        # Under normal circustances we should subtract ref from signal. Here it has already been done.
        ref = list(ch_dict.keys()).index("cz")
        copied_data = data
        reference_signal = copied_data[0][ref]
        for ch_num in copied_data[0].keys():
            if ch_num != ref and isinstance(ch_num, (int, float)):
                copied_data[0][ch_num] = copied_data[0][ch_num] - reference_signal
        return copied_data

    def _merge_time_frames(self, exclude_intervals):
        gathered_intervals = []
        for possible_interval in exclude_intervals:
            if gathered_intervals == []:
                gathered_intervals.append(possible_interval)
            else:
                p_start, p_end = possible_interval[0], possible_interval[1]
                for i, interval in enumerate(gathered_intervals):
                    start, end = interval[0], interval[1]
                    if p_start <= start and p_end <= end and p_end >= start:
                        gathered_intervals[i][0] = p_start
                    elif  p_start >= start and p_end >= end and p_start <= end:
                        gathered_intervals[i][1] = p_end
                    elif  p_start <= start and p_end >= end:
                        gathered_intervals[i][0] = p_start
                        gathered_intervals[i][1] = p_end
                    elif  p_start >= start and p_end <= end:
                        pass
        return gathered_intervals

    def operations(self, data, ch_dict):
        s = data
        ref_out_data = self._reference_out(data=data, ch_dict=ch_dict)
        extra_exclude_map = self._max_consec_zeros(data = ref_out_data)
        extra_exclude_map = self._merge_time_frames(extra_exclude_map)
        data[0]["exclude_map"] += extra_exclude_map
        for ch_num in data[0].keys():
            if isinstance(ch_num, (int,float)):
                data[0][ch_num] = s[0][ch_num]
        return data


class Add_Operator_Seizit1(Additional_Operator):
    def __init__(self, config):
        super().__init__(config)
        
    def longtitudinal_montage(self, data, ch_dict):
        temp_dict = {}
        for i, montage_pair in enumerate(self.config.montage):
            # In data[0] the 0 refers to first modality -> EEG.
            #TODO: Build this functionality (montage) for any type of modalities, dynamically.
            temp_dict[i] = data[0][ch_dict[montage_pair[0].lower()]] - data[0][ch_dict[montage_pair[1].lower()]]
        keys = list(data[0].keys())
        for key in keys:
            if isinstance(key, (int, float)):
                del data[0][key]
        for i in range(len(temp_dict)):
            data[0][i] = temp_dict[i]
        data[0]['num_channels'] = len(temp_dict)
        return data

    def operations(self, data, ch_dict):
        data =  self.longtitudinal_montage( data, ch_dict)

        return data

