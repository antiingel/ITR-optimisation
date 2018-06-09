import scipy.io
import numpy as np
import pandas as pd
import os

frequencies = ["8", "14", "28"]
subjects = ["1", "2", "3", "4"]
trials = ["1", "2", "3", "4", "5"]

electrodes = ['A1 (Cz)', 'A2', 'A3 (CPz)', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10 (PO7)', 'A11', 'A12', 'A13', 'A14', 'A15 (O1)', 'A16', 'A17', 'A18', 'A19 (Pz)', 'A20', 'A21 (POz)', 'A22', 'A23 (Oz)', 'A24', 'A25 (Iz, Inion)', 'A26', 'A27', 'A28 (O2)', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7 (PO8)', 'B8', 'B9', 'B10', 'B11 (P8)', 'B12', 'B13', 'B14 (TP8)', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20 (C2)', 'B21', 'B22 (C4)', 'B23', 'B24 (C6)', 'B25', 'B26 (T8)', 'B27 (FT8)', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7 (F8)', 'C8 (AF8)', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16 (Fp2)', 'C17 (Fpz)', 'C18', 'C19 (AFz)', 'C20', 'C21 (Fz)', 'C22', 'C23 (FCz)', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29 (Fp1)', 'C30 (AF7)', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7 (F7)', 'D8 (FT7)', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14 (C1)', 'D15', 'D16', 'D17', 'D18', 'D19 (C3)', 'D20', 'D21 (C5)', 'D22', 'D23 (T7)', 'D24 (TP7)', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31 (P7)', 'D32']

for frequency in frequencies:
    for subject in subjects:
        output_folder_path = os.path.join(os.pardir, "data", "original_data_as_csv")
        for trial in trials:
            input_file_path = os.path.join(os.pardir, "data", "original_data", "SSVEP_" + frequency + "Hz_Trial" + trial + "_SUBJ" + subject +".MAT")
            data = scipy.io.loadmat(input_file_path)
            transposed_eeg_data = pd.DataFrame(np.transpose(data["EEGdata"]), columns=electrodes)
            file_name = frequency + "sub" + subject + "trial" + trial + ".csv"
            transposed_eeg_data.to_csv(os.path.join(output_folder_path, file_name))
