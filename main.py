"""
Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from utils.config import *

from agents.Sleep_Preparation.test_agent import Sleep_Agent

import matplotlib.pyplot as plt
import numpy as np


def main():
    print("We are running main")
    config_list = [
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/shhs/eeg_eog_nooverlap_npz.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/shhs/eeg_eog_nooverlap_zarr.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/shhs/eeg_eog_nooverlap_mat.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/shhs/eeg_eog_nooverlap.json",
        "/users/sista/kkontras/Documents/Sleep_Prepare/configs/nch/eeg_eog_nooverlap_mat.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/nch/eog_hdf5.json",
        # "./configs/shhs/eeg_eog_nooverlap_mat.json",
        # "./configs/shhs/eeg_eog_nooverlap_mat_shhs2.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/sleep_edf78/eeg_eog_nooverlap_mat.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/neonatal/eeg_eog_nooverlap.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/sleep_edf78/eeg_eog_nooverlap_mat.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/neonatal/eeg_eog_nooverlap_mat.json",
        # "./configs/sleep_edf78/eeg_eog_nooverlap.json",
    ]
    for i in config_list:
        config = process_config(i)
        agent = Sleep_Agent(config)
        agent.run()
        agent.finalize()

main()