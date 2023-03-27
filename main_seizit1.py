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
        "/users/sista/kkontras/Documents/Sleep_Prepare/configs/SeizIT1/eeg_ecg.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/sleep_edf78/eeg_eog_nooverlap.json",
        # "/users/sista/kkontras/Documents/Sleep_Prepare/configs/neonatal/eeg_eog_nooverlap.json",
        # "./configs/sleep_edf78/eeg_eog_nooverlap.json",
    ]
    for i in config_list:
        config = process_config(i)
        agent = Sleep_Agent(config)
        agent.run()
        agent.finalize()

main()