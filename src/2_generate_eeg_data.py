
import pandas as pd
import os


frequencies = ["8", "14", "28"]
subjects = ["1", "2", "3", "4"]
recordings = ["1", "2", "3", "4", "5"]
sampling_freq = 256
stimulation_start_second = 5
stimulation_stop_second = 20
stimulation_start_packet = sampling_freq*stimulation_start_second
stimulation_stop_packet = sampling_freq*stimulation_stop_second
O1_index = 15
O2_index = 28
output_folder_path = os.path.join(os.pardir, "data", "eeg_data")

for subject in subjects:
    for recording in recordings:
        output_file_name = "sub" + subject + "rec" + recording + ".csv"
        data_frames = []
        for frequency in frequencies:
            input_file_path = os.path.join(os.pardir, "data", "original_data_as_csv", frequency + "sub" + subject + "trial" + recording + ".csv")
            all_data = pd.read_csv(input_file_path)
            data = all_data.iloc[stimulation_start_packet:stimulation_stop_packet, [O1_index, O2_index]]
            data["label"] = [frequency]*(stimulation_stop_packet-stimulation_start_packet)
            data_frames.append(data)
        result = pd.concat(data_frames)
        result.to_csv(os.path.join(output_folder_path, output_file_name))
