from abc import ABC, abstractmethod
import numpy as np

def Get_Window_Labeler(b_policy, file_duration, num_classes):
    if b_policy == "seizure":
        return Window_Labeler_Seizure(file_duration, num_classes)
    elif b_policy == "majority":
        return Window_Labeler_Majority(file_duration, num_classes)
    elif b_policy == "softlabels":
        return Window_Labeler_Softlabel(file_duration, num_classes)
    else:
        raise ValueError("This type of label policy do not exist")

class Window_Labeler(ABC):
    @abstractmethod
    def __init__(self, file_duration, num_classes):
        self.file_duration = file_duration
        self.num_classes = num_classes
        self.class_identity = np.identity(self.num_classes)

    @abstractmethod
    def get_label(self, start, end, label_intervals, id):
        raise NotImplementedError()

    def _find_window_label(self, data, start, end):
        flag = True
        for i, interval in enumerate(data["label_intervals"]):
            if start >= interval[0][0] and start < interval[0][1]:
                flag = False
                if end <= interval[0][1]:
                    if type(interval[1]) == str:
                        return interval[1]
                    else:
                        return self.class_identity[int(interval[1]),:]
                else:
                    return self.get_label(start=start, end=end, label_intervals=data["label_intervals"], id=i)
        if flag:
            if end < self.file_duration * data["sampling_rate"]:
                pass
                # print("there is no labelling for some of it in start:{} end: {} and file_duration: {}".format(start, end, self.file_duration * data["sampling_rate"]))
            return "no_label"

    def check_find_window_label(self, data, window, previous_flag):
        """
        :param window: A list of two numbers, the beginning and the end of the window
        :return:
            :flag: Bool revealing if we should discard the next window
            :init_label: 0 or 1 showing whether the current window has time coherence with the neighboring windows
            :end_excluded: int the point we should start sampling the signal from, indicates where problems stop
        """
        flag_exclude_current, flag_exclude_next, init_label, end_excluded = True, True, 0, -1
        window_length = window[1] - window[0]
        next_window = [i + window_length for i in window]
        for interval in data["exclude_map"]:
            if (window[0] > interval[0] - window_length and window[0] < interval[1]) or (
                    window[1] > interval[0] and window[1] < interval[1] + window_length):
                flag_exclude_current = False
                end_excluded = interval[1]
            if (next_window[0] > interval[0] - window_length and next_window[0] < interval[1]) or (
                    next_window[1] > interval[0] and next_window[1] < interval[1] + window_length):
                init_label = 1
                flag_exclude_next = False

        if flag_exclude_current:
            w_label = self._find_window_label(data=data, start=window[0], end=window[1])
            # w_label = self._find_wlabel(modality, b_policy, window[0], window[1])
        else:
            w_label = -1

        # If previous one was skipped
        if not previous_flag:
            init_label = 1

        if flag_exclude_next:
            w_label_next = self._find_window_label(data=data, start=next_window[0], end=next_window[1])
            # If next one is about to be skipped
            if type(w_label_next)==str and w_label_next == 'no_label':
                init_label = 1
        else:
            init_label = 1

        return flag_exclude_current, init_label, end_excluded, w_label

class Window_Labeler_Seizure(Window_Labeler):
    def __init__(self, file_duration):
        super().__init__(file_duration)

    def get_label(self, start, end, label_intervals, id):

        labels = []
        for li in range(id, len(label_intervals)):
            if end < label_intervals[li][0][0]:
                break
        s = max(start, label_intervals[li][0][0])
        e = min(end, label_intervals[li][0][1])
        labels.append([label_intervals[li][1], e - s])

        l_counts = np.zeros(10)
        for l in labels:
            if l[0] == "no_label":
                return "no_label"
            l_counts[l[0]] += l[1]
        if l_counts[1] != 0:
            return 1
        else:
            return 0

class Window_Labeler_Majority(Window_Labeler):
    def __init__(self, file_duration, num_classes):
        super().__init__(file_duration, num_classes)

    def get_label(self,start, end, label_intervals, id):
        labels = []
        for li in range(id, len(label_intervals)):
            if end < label_intervals[li][0][0]:
                break
            s = max(start, label_intervals[li][0][0])
            e = min(end, label_intervals[li][0][1])
            labels.append([label_intervals[li][1],e-s])
        l_counts = np.zeros(10)
        for l in labels:
            if l[0] == "no_label":
                return "no_label"
            l_counts[l[0]]+=l[1]
        return np.argmax(l_counts)

class Window_Labeler_Softlabel(Window_Labeler):
    def __init__(self, file_duration, num_classes):
        super().__init__(file_duration, num_classes)

    def get_label(self,start, end, label_intervals, id):
        labels = []
        for li in range(id, len(label_intervals)):
            if end < label_intervals[li][0][0]:
                break
            s = max(start, label_intervals[li][0][0])
            e = min(end, label_intervals[li][0][1])
            labels.append([label_intervals[li][1],e-s])
        l_counts = np.zeros(self.num_classes)
        for l in labels:
            if l[0] == "no_label":
                return "no_label"

            try:
                l_counts[l[0]]+=l[1]
            except:
                print("Here is the mistake")
                print(l[0])

        return l_counts/l_counts.sum()
