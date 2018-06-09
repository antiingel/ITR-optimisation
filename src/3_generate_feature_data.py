import scipy.signal
import scipy.interpolate
import numpy as np
import pandas as pd
import sklearn.cross_decomposition
import os

frequencies = ["14", "28", "8"]
subjects = ["1", "2", "3", "4"]
recordings = ["1", "2", "3", "4", "5"]
sampling_freq = 256
window_length = 256
time_step = 32

def sliding_window(data, size, stepsize=1, axis=0):
    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)
    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])
    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )
    return strided

def get_breakpoints(breakpoints_count):
    breakpoints_count = breakpoints_count + 1
    breakpoints_list = []
    for i in range(breakpoints_count):
        breakpoints_list.append(window_length / breakpoints_count * i)
    return breakpoints_list

def interpolation(x, y):
    return scipy.interpolate.interp1d(x, y, kind="linear")

def get_corr(model, signal, reference):
    model.fit(signal, reference)
    res_x, res_y = model.transform(signal, reference)
    corr = np.corrcoef(res_x.T, res_y.T)[0][1]
    return corr

def getReferenceSignals(length, frequencies, all_harmonics):
    reference_signals = []
    t = np.arange(0, length, step=1.0)/sampling_freq
    for freq, harmonics in zip(frequencies, all_harmonics):
        reference_signals.append([])
        for harmonic in harmonics:
            reference_signals[-1].append(np.sin(np.pi*2*harmonic*freq*t))
            reference_signals[-1].append(np.cos(np.pi*2*harmonic*freq*t))
    return np.array(reference_signals)

for subject in subjects:
    for recording in recordings:
        input_file_path = os.path.join(os.pardir, "data", "eeg_data", "sub" + subject + "rec" + recording + ".csv")
        output_file_path = os.path.join(os.pardir, "data", "feature_data", "sub" + subject + "rec" + recording + ".csv")
        all_data = pd.read_csv(input_file_path)
        eeg_data = all_data.as_matrix(["A15 (O1)", "A28 (O2)"])
        label_data = all_data.as_matrix(["label"])

        split_data = sliding_window(eeg_data, window_length, time_step)
        split_labels = sliding_window(label_data, window_length, time_step)
        breakpoints = get_breakpoints(4)
        window_function = scipy.signal.get_window(("kaiser", 14), window_length)
        fft_bins = np.fft.rfftfreq(window_length)[1:]*sampling_freq
        model = sklearn.cross_decomposition.CCA(n_components=1)
        reference_signals = getReferenceSignals(window_length, [14,28,8], [[1,2,3],[1,2,3],[1,2,3]])

        all_features = []
        for window, labels in zip(split_data, split_labels):
            detrended_signals = []
            for channel in window:
                # print(channel)
                detrended_signal = scipy.signal.detrend(channel, type="linear", bp=breakpoints)
                windowed_signal = detrended_signal*window_function
                amplitude_spectrum = (np.abs(np.fft.rfft(windowed_signal)) ** 2)[1:]
                # interpolation_func = interpolation(fft_bins, amplitude_spectrum)
                # psd_features = [interpolation_func(frequency) for frequency in frequencies]
                psd_featuress = [amplitude_spectrum[int(frequency)-1] for frequency in frequencies]
                detrended_signals.append(detrended_signal)
                # print(psd_featuress)

            cca_features = [get_corr(model, np.transpose(detrended_signals), np.transpose(reference_signal)) for reference_signal in reference_signals]
            # print(cca_features)

            signal_sum = np.average(window, axis=0)
            detrended_signal = scipy.signal.detrend(signal_sum, type="linear", bp=breakpoints)
            windowed_signal = detrended_signal * window_function
            amplitude_spectrum = (np.abs(np.fft.rfft(windowed_signal)) ** 2)[1:]
            # interpolation_func = interpolation(fft_bins, amplitude_spectrum)
            # psd_featuress = [interpolation_func(frequency) for frequency in frequencies]
            psd_features = [amplitude_spectrum[int(frequency)*harmonic - 1] for harmonic in [1,2,3] for frequency in frequencies]
            sum_psd_features = [sum(amplitude_spectrum[int(frequency)*harmonic - 1] for harmonic in [1,2,3]) for frequency in frequencies]
            all_features.append(psd_features + sum_psd_features + cca_features + [{8:1, 14:2, 28:3}[labels[0][-1]]])
            # print(psd_features)
            # print(np.log10(psd_features))
            # print(np.array(psd_features)/sum(amplitude_spectrum))
        feature_names=["1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_1_14.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_1_28.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_1_8.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_2_14.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_2_28.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_2_8.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_3_14.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_3_28.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_3_8.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_Sum_14.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_Sum_28.0","1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_Sum_8.0","2_('O1', 'O2')_CCA_('O1', 'O2')_Sum_14.0","2_('O1', 'O2')_CCA_('O1', 'O2')_Sum_28.0","2_('O1', 'O2')_CCA_('O1', 'O2')_Sum_8.0"]
        df = pd.DataFrame(all_features, columns=feature_names + ["label"])
        df.to_csv(os.path.join(output_file_path))
