def enhance_signals(self, enhancement):
    initial_num_labels = len(self.f_labels)
    initial_nun_excl = len(self.exclude_map)
    initial_signal_shape = self.signals.shape
    for enh in enhancement:
        print(enh)
        if enhancement[enh]["method"] == "Gaussian":
            self.augmentation_gaussian(enh=enhancement[enh], initial_num_labels=initial_num_labels,
                                       initial_nun_excl=initial_nun_excl, initial_signal_shape=initial_signal_shape)
        elif enhancement[enh]["method"] == "Extract_Freq":
            self.augmentation_extract_freq(enh=enhancement[enh], initial_num_labels=initial_num_labels,
                                           initial_nun_excl=initial_nun_excl, initial_signal_shape=initial_signal_shape)
        else:
            raise Exception("That type of data augmentation does not exist")


def augmentation_gaussian(self, enh, initial_num_labels, initial_nun_excl, initial_signal_shape):
    for t in range(1, enh["times"] + 1):
        # Augment the signal
        aug = np.random.normal(enh["mean"], enh["std"], size=(initial_signal_shape[0], initial_signal_shape[1]))
        aug = self.signals[:, :self.signal_length] + aug
        self.signals = np.concatenate([self.signals, aug], axis=1)

        current_num_labels = len(self.f_labels)
        current_num_excl = len(self.exclude_map)

        # Add (Duplicate) labels
        for i in range(current_num_labels - initial_num_labels, current_num_labels):
            self.f_labels.append(
                [self.f_labels[i][0] + t * self.signal_length, self.f_labels[i][1] + t * self.signal_length,
                 self.f_labels[i][2]])
        # Add (duplicate) the exclude map
        for i in range(current_num_excl - initial_nun_excl, current_num_excl):
            self.exclude_map.append(
                [self.exclude_map[i][0] + t * self.signal_length, self.exclude_map[i][1] + t * self.signal_length])


def augmentation_extract_freq(self, enh, initial_num_labels, initial_nun_excl, initial_signal_shape):
    if enh["filt_type"] == "butter":
        nyq = 0.5 * self.srate
        sos = sg.butter(enh["order"], [enh["filt_cutoffs"][0] / nyq, enh["filt_cutoffs"][1] / nyq], btype='band',
                        output="sos")
    elif enh["filt_type"] == "fir":
        coef = sg.firwin(fs=self.srate, numtaps=enh["order"], cutoff=[enh["filt_cutoffs"][0], enh["filt_cutoffs"][1]],
                         window="hamming", pass_zero=False)

    current_num_labels = len(self.f_labels)
    current_num_excl = len(self.exclude_map)

    for t in range(1, enh["times"] + 1):
        if enh["filt_type"] == "butter":
            aug = np.array([sg.sosfiltfilt(sos, signal) for signal in self.signals[:, :self.signal_length]])
        elif enh["filt_type"] == "fir":
            aug = np.array([sg.filtfilt(coef, [1], signal) for signal in self.signals[:, :self.signal_length]])

        # Augment the signal
        self.signals = np.concatenate([self.signals, aug], axis=1)

        # Add (Duplicate) labels
        for i in range(current_num_labels - initial_num_labels, current_num_labels):
            self.f_labels.append(
                [self.f_labels[i][0] + t * self.signal_length, self.f_labels[i][1] + t * self.signal_length,
                 self.f_labels[i][2]])
        # Add (duplicate) the exclude map
        for i in range(current_num_excl - initial_nun_excl, current_num_excl):
            self.exclude_map.append(
                [self.exclude_map[i][0] + t * self.signal_length, self.exclude_map[i][1] + t * self.signal_length])
