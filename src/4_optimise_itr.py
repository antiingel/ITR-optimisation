from itr_optimisation import ModelTrainer
import os
import pandas as pd
import numpy as np


subjects = ["1", "2", "3", "4"]
recordings = ["1", "2", "3", "4", "5"]
feature_names = ["1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_1_14.0", "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_1_28.0",
                       "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_1_8.0", "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_2_14.0",
                       "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_2_28.0", "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_2_8.0",
                       "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_3_14.0", "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_3_28.0",
                       "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_3_8.0", "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_Sum_14.0",
                       "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_Sum_28.0", "1_('O1', 'O2')_Sum PSDA_('O1', 'O2')_Sum_8.0",
                       "2_('O1', 'O2')_CCA_('O1', 'O2')_Sum_14.0", "2_('O1', 'O2')_CCA_('O1', 'O2')_Sum_28.0",
                       "2_('O1', 'O2')_CCA_('O1', 'O2')_Sum_8.0"]

for subject in subjects:
    print "\nOptimising for subject " + str(subject)
    all_data_for_subject = []
    labels_for_subject = []
    for recording in recordings:
        input_file_name = os.path.join(os.pardir, "data", "feature_data", "sub" + subject + "rec" + recording + ".csv")
        data = pd.read_csv(input_file_name)
        features = data.iloc[:,1:16].as_matrix()
        labels = data["label"].as_matrix()
        labels = np.delete(labels, list(range(113, 120)) + list(range(233, 240)), axis=0)
        features = np.delete(features, list(range(113, 120)) + list(range(233, 240)), axis=0)
        features_dictionary = [{key: value for key, value in zip(feature_names, feature_vector)} for feature_vector in features]

        all_data_for_subject.append(features_dictionary)
        labels_for_subject.append(labels)

    trainer = ModelTrainer.ModelTrainer()
    trainer.setup(
        1,
        feature_names,
        all_data_for_subject,
        labels_for_subject
    )
    trainer.start(subject, make_plots=True)
    raw_input("\nPress enter to start optimising for next subject")
