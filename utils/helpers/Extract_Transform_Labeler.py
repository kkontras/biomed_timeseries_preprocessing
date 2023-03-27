from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
import pandas as pd
import mne
import json
import warnings

def Get_Labeler(type, label_file, label_mapping, file_duration):
    file_type = label_file.split(".")[-1]
    if type == "nch" or type=="nch_total":
        return Extract_Labels_TSV_NCH(file= label_file, label_mapping=label_mapping, file_duration=file_duration)

    if file_type == "edf":
        return Extract_Labels_EDF(file= label_file, label_mapping=label_mapping)
    elif file_type == "xml":
        return Extract_Labels_XML(file= label_file, label_mapping=label_mapping)
    elif file_type == "tsv":
        return Extract_Labels_TSV(file= label_file, label_mapping=label_mapping, file_duration=file_duration)
    else:
        assert "This type of label extractor do not exist"

class Extract_Transform_Labeler(ABC):
    @abstractmethod
    def __init__(self, file, label_mapping):
        self.file = file
        self.file_type = file.split(".")[-1]
        #Read mapping
        with open(label_mapping) as json_file:
            self.label_mapping = json.load(json_file)
        self.exclude_map = []
        self.label_intervals = []

    def transform_labels(self, label_intervals, end_labelling = None):

        previous_label = 0
        tf_labels = []
        background_label = self.label_mapping["Background"]
        for ind, interval in enumerate(label_intervals):
            if interval[1] not in self.label_mapping:
                label = "no_label"
                warnings.warn("You have not include label {} in the mapping".format(interval[1]))
            else:
                label = self.label_mapping[interval[1]]

            #If its a concurrent label skip it.
            if label == "no_label_cuncurrent":
                continue

            if (len(tf_labels) == 0):
                if (interval[0][0] != 0):
                    if (background_label != label):
                        tf_labels.append([[0, interval[0][0]], background_label])
                        tf_labels.append([[interval[0][0], interval[0][1]], label])
                    else:
                        tf_labels.append([[0, interval[0][1]], background_label])
                else:
                    tf_labels.append([[interval[0][0], interval[0][1]], label])
                previous_label = label
            else:
                if tf_labels[-1][0][1] < interval[0][0]:
                    if (previous_label == background_label):
                        tf_labels[-1][0][1] = interval[0][1]
                    else:
                        tf_labels.append([[tf_labels[-1][0][1], interval[0][0]], background_label])
                        previous_label = background_label
                if (label == previous_label):
                    tf_labels[-1][0][1] = interval[0][1]
                else:
                    tf_labels.append([[interval[0][0], interval[0][1]], label])
                    previous_label = label

        if end_labelling:
            if tf_labels[-1][0][1] < end_labelling:
                tf_labels.append([[tf_labels[-1][0][1], end_labelling],
                                  background_label])

        return tf_labels

    @abstractmethod
    def get_labels(self, file):
        raise NotImplementedError()

    @abstractmethod
    def _get_events(self, file):
        raise NotImplementedError()

class Extract_Labels_XML(Extract_Transform_Labeler):
    def __init__(self, file, label_mapping):
        super().__init__(file, label_mapping)

    def _get_events(self):
        """
        :param file: The xml file that includes the data labels
        :return: the events of the xml.
        """
        tree = ET.parse(self.file)
        root = tree.getroot()
        events = root.getchildren()[2].getchildren()
        return events

    def get_labels(self):

        events = self._get_events()
        previous_end = 0
        for event in events:
            event_descr = event.getchildren()
            if event_descr[0].text == "Stages|Stages":
                start_ann = int(float(event_descr[2].text))
                end_ann = int(float(event_descr[2].text)) + int(float(event_descr[3].text))
                if (previous_end != start_ann):
                    self.exclude_map.append([previous_end, start_ann])
                self.label_intervals.append([[start_ann, end_ann], event_descr[1].text])
                previous_end = end_ann
        return self.label_intervals, self.exclude_map

class Extract_Labels_TSV(Extract_Transform_Labeler):
    def __init__(self, file, label_mapping, file_duration):
        super().__init__(file, label_mapping)
        self.file_duration = file_duration
    def _get_events(self):
        """
        :param file: The xml file that includes the data labels
        :return: the events of the xml.
        """
        events = pd.read_csv(self.file, sep='\t', header=None, names=[0, 1, 2, 3])
        return events

    def get_labels(self):

        #ONLY FOR SeizIT1
        events = self._get_events()
        self.label_intervals, start_ann, end_ann = [], -1, -1
        for i in range(events.shape[0] - 9):
            if events[0][9 + i] != 'None' and events[1][9 + i] != 'None' and '#' not in events[0][9 + i]:
                if len(events[0][9 + i])>10:
                    label_list = [x for x in events[0][9 + i].split(' ') if x]
                    start_sec, stop_sec, label = label_list[0], label_list[1], label_list[2]
                else:
                    start_sec, stop_sec, label = int(events[0][9 + i]), int(events[1][9 + i]), events[2][9 + i]

                if len(self.label_intervals) == 0 and start_sec !=0:
                    self.label_intervals.append([[0, start_sec], "Background"])
                self.label_intervals.append([[start_sec, stop_sec],label])

        if self.label_intervals == []:
            self.label_intervals.append([[0, self.file_duration], "Background"])
        # if self.label_intervals[-1][0][1] != self.file_duration:
        #     self.label_intervals.append([[self.label_intervals[-1][0][1], self.file_duration],"Background"])

        return self.label_intervals, self.exclude_map

class Extract_Labels_TSV_NCH(Extract_Transform_Labeler):
    def __init__(self, file, label_mapping, file_duration):
        super().__init__(file, label_mapping)
        self.file_duration = file_duration
    def _get_events(self):
        """
        :param file: The xml file that includes the data labels
        :return: the events of the xml.
        """
        events = pd.read_csv(self.file, sep='\t', header=None, names=[0, 1, 2, 3])

        return events

    def get_labels(self):

        #ONLY FOR SeizIT1
        events = self._get_events()
        self.label_intervals, start_ann, end_ann = [], -1, -1

        start_sec = 0
        for i, dur in enumerate(events[1]):
            if isfloat(dur) and float(dur) > 0:
                self.label_intervals.append([[float(events[0][i]), float(events[0][i]) + float(dur)], events[2][i]])

        # for i in range(events.shape[0] - 9):
        #     if events[0][9 + i] != 'None' and events[1][9 + i] != 'None' and '#' not in events[0][9 + i]:
        #         if len(events[0][9 + i])>10:
        #             print(events[0][9 + i])
        #             label_list = [x for x in events[0][9 + i].split(' ') if x]
        #             print(label_list)
        #             start_sec, stop_sec, label = label_list[0], label_list[1], label_list[2]
        #         else:
        #             start_sec, stop_sec, label = int(events[0][9 + i]), int(events[1][9 + i]), events[2][9 + i]

        # if len(self.label_intervals) == 0 and start_sec !=0:
        #     self.label_intervals.append([[0, start_sec], "Background"])
        # self.label_intervals.append([[start_sec, stop_sec],label])

        if self.label_intervals == []:
            self.label_intervals.append([[0, self.file_duration], "Background"])
        # if self.label_intervals[-1][0][1] != self.file_duration:
        #     self.label_intervals.append([[self.label_intervals[-1][0][1], self.file_duration],"Background"])

        return self.label_intervals, self.exclude_map

class Extract_Labels_EDF(Extract_Transform_Labeler):
    def __init__(self, file, label_mapping):
        super().__init__(file, label_mapping)

    def _get_events(self):
        """
        :param file: The edf file that includes the data labels
        :return: the events of the edf.
        """
        events = mne.read_annotations(self.file)
        return events

    def get_labels(self):

        events = self._get_events()

        previous_end = 0
        for ann in events:
            start_ann = ann["onset"]
            end_ann = ann["onset"] + ann["duration"]
            if (previous_end != start_ann):
                self.exclude_map.append([previous_end, start_ann])
            self.label_intervals.append([[start_ann, end_ann], ann["description"]])
            previous_end = end_ann

        return self.label_intervals, self.exclude_map

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
