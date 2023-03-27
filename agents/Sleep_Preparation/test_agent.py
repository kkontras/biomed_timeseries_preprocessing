"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np
from agents.base import BaseAgent
from utils.sleep_data_prep.edf_reader import EDF_Reader
from tqdm import tqdm
from utils.sleep_data_prep.sleep_data_utils import load_pyedf, rearrange_headers, resample, create_windows
from pathlib import Path
import os
import csv
import re
import sys
import json
import shutil


from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from utils.sleep_data_prep.File_Struct import *
# cudnn.benchmark = True

class Sleep_Agent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)


    def run(self):
        """
        The main operator
        :return:
        """
        try:
            #SLEEP EDF78
            # file_object = File_Struct(self.config, ["E0-PSG.edf","F0-PSG.edf","G0-PSG.edf"], ["-Hypnogram.edf"])
            # file_object.create_windows(types = ["time","stft"], signals=["eeg-eog"])

            #SHHS
            # file_object = File_Struct(self.config, [".edf"], ["-nsrr.xml"])
            # file_object.create_windows(types = ["time","stft"], signals=["shhs"])

            #NEONATAL
            file_object = File_Struct(self.config)
            file_object.process_patients(types = ["time", "stft"])
            # file_object.split_val_test_train_files()
            # file_object.record_mat_files()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        # self.save_checkpoint("./data/{}".format(self.config.checkpoint_file),0)
        print("We are in the final state.")
        pass




