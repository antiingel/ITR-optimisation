from itr_optimisation import *

from sklearn.feature_selection import SelectFpr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.metrics


subjects = ["1", "2", "3", "4"]
recordings = ["1", "2", "3", "4", "5"]


n_classes = 3
window_length = 1
step_length = 0.125

do_feature_selection = True
add_ratios = True
do_lda = True
do_lda_separately = True
do_skew_norm_plots = False

itr_function, gradient_function = construct_objective_function(window_length, step_length, n_classes)


for subject in subjects:

    all_data_for_subject, labels_for_subject = read_data(subject, recordings, "data")

    n_trials, n_samples, n_features = all_data_for_subject.shape

    assert n_classes == len(np.unique(labels_for_subject))

    lda_model = LinearDiscriminantAnalysis()

    feature_selector = SelectFpr(alpha=5e-2 if do_feature_selection else 1)

    accuracies = []
    mdts = []
    prediction_probabilities = []
    standard_itrs = []
    mi_itrs = []

    for cv_index in range(n_trials):

        training_data = np.delete(all_data_for_subject, cv_index, 0).copy()
        test_data = all_data_for_subject[cv_index,:,:].copy()

        training_labels = np.delete(labels_for_subject, cv_index, 0)
        test_labels = labels_for_subject[cv_index,:]

        training_data, test_data = scale_data(all_data_for_subject, training_data, test_data, n_features, n_classes)

        n_new_features, training_data, test_data = add_ratios_as_features(training_data, test_data, n_trials, n_samples, n_features, n_classes, add_ratios)

        new_train_features, new_test_features = apply_lda(training_data, training_labels, test_data, feature_selector, lda_model, n_new_features, n_trials, do_lda, do_lda_separately)

        train_features_given_class = np.array([new_train_features.reshape(-1, n_classes)[np.where(training_labels.reshape(-1)==i+1)] for i in range(n_classes)])

        skew_norm_params = calculate_skew_norm_params(train_features_given_class, n_classes)

        plot_histogram_and_skew_norm(train_features_given_class, skew_norm_params, n_classes, do_skew_norm_plots)

        itrs, thresholds = train_classifier(new_train_features, itr_function, gradient_function, skew_norm_params, subject, n_classes)

        if not np.all(np.isnan(itrs)):
            best_thresholds = thresholds[np.nanargmax(itrs)]

            test_predict = predict(new_test_features, best_thresholds, n_classes)
            test_confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, test_predict, labels=[i+1 for i in range(n_classes+1)])

            accuracies.append(accuracy_from_confusion_matrix(test_confusion_matrix))
            prediction_probabilities.append(prediction_probability_from_confusion_matrix(test_confusion_matrix))
            mi_itrs.append(itr_mi_from_confusion_matrix(test_confusion_matrix, window_length, step_length, n_classes))
            standard_itrs.append(standard_itr_from_confusion_matrix(test_confusion_matrix, window_length, step_length, n_classes))
            mdts.append(mdt_from_prediction_prob(prediction_probabilities[-1], window_length, step_length))

    print("Results for subject " + str(subject) + ":")
    print(np.mean(accuracies), accuracies)
    print(np.mean(mdts), mdts)
    print(np.mean(mi_itrs), mi_itrs)
    print(np.mean(standard_itrs), standard_itrs)
    print(np.mean(prediction_probabilities), prediction_probabilities)
