
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skewnorm
from scipy.optimize import basinhopping
import sklearn.metrics

import os


def construct_objective_function(window_length, step_length, n_classes):
    p = [sympy.Symbol("P_" + str(i)) for i in range(n_classes)]
    c = [sympy.Symbol("C_" + str(i)) for i in range(n_classes)]

    m = sympy.Symbol("P(M)")

    p_given_m = [sympy.Symbol("P(P_" + str(i) + "|M)") for i in range(n_classes)]
    p_given_c = [[sympy.Symbol("P(P_" + str(i) + "|C_" + str(j) + ")") for j in range(n_classes)] for i in range(n_classes)]
    c_given_m = [sympy.Symbol("P(C_" + str(j) + "|M)") for j in range(n_classes)]
    m_given_c = [sympy.Symbol("P(M|C_" + str(j) + ")") for j in range(n_classes)]
    p_given_cm = [[sympy.Symbol("P(P_" + str(i) + "|C_" + str(j) + ",M)") for j in range(n_classes)] for i in range(n_classes)]

    f_given_c = [[sympy.Function("F_" + str(i) + "C_" + str(j) + "") for j in range(n_classes)] for i in range(n_classes)]

    t = [sympy.Symbol("t_" + str(i)) for i in range(n_classes)]

    itr = sympy.Add(*[p_given_cm[i][j]*c_given_m[j]*sympy.Piecewise(
        (0, sympy.Le(p_given_m[i], 0)), (0, sympy.Le(p_given_cm[i][j], 0)), (sympy.log(p_given_cm[i][j]/p_given_m[i], 2), True)
    ) for i in range(n_classes) for j in range(n_classes)])
    # itr = sympy.Add(*[p_given_cm[i][j]*c_given_m[j]*sympy.log(p_given_cm[i][j]/p_given_m[i], 2) for i in range(n_classes) for j in range(n_classes)])

    mdt = window_length + (1/m - 1) * step_length

    unfolded_itr = itr*60/mdt

    unfolded_itr = unfolded_itr.subs({p_given_cm[i][j]: p_given_c[i][j]/m_given_c[j] for i in range(n_classes) for j in range(n_classes)})
    unfolded_itr = unfolded_itr.subs({m_given_c[j]: sympy.Add(*[p_given_c[i][j] for i in range(n_classes)]) for j in range(n_classes)})
    unfolded_itr = unfolded_itr.subs({c_given_m[j]: sympy.Add(*[p_given_c[i][j]*c[j]/m for i in range(n_classes)]) for j in range(n_classes)})
    unfolded_itr = unfolded_itr.subs({p_given_m[i]: p[i]/m for i in range(n_classes)})
    unfolded_itr = unfolded_itr.subs({m: sympy.Add(*[p[i] for i in range(n_classes)])})
    unfolded_itr = unfolded_itr.subs({p[i]: sympy.Add(*[p_given_c[i][j]*c[j] for j in range(n_classes)]) for i in range(n_classes)})
    unfolded_itr = unfolded_itr.subs({c[j]: sympy.Integer(1)/n_classes for j in range(n_classes)})
    unfolded_itr = unfolded_itr.subs({p_given_c[i][k]: (1-f_given_c[i][k](t[i]))*sympy.Mul(*[f_given_c[j][k](t[j]) for j in range(n_classes) if j != i]) for i in range(n_classes) for k in range(n_classes)})

    itr_gradient = [unfolded_itr.diff(t[i]) for i in range(n_classes)]

    itr_function = sympy.lambdify([e(t[i]) for i,l in enumerate(f_given_c) for e in l], unfolded_itr, "numpy")

    gradient_arguments = [sympy.Derivative(f_given_c[i][j](t[i]), t[i]) for i in range(n_classes) for j in range(n_classes)]
    gradient_arguments += [f_given_c[i][j](t[i]) for i in range(n_classes) for j in range(n_classes)]
    gradient_function = sympy.lambdify(gradient_arguments, itr_gradient, "numpy")
    return itr_function, gradient_function


def calculate_itr(t_values, itr_function, skew_norm_params, n_classes):
    return itr_function(*[skewnorm.cdf(t_values[i], *skew_norm_params[i][j]) for i in range(n_classes) for j in range(n_classes)])


def calculate_gradient(t_values, gradient_function, skew_norm_params, n_classes):
    gradient_arguments = [skewnorm.pdf(t_values[i], *skew_norm_params[i][j]) for i in range(n_classes) for j in range(n_classes)]
    gradient_arguments += [skewnorm.cdf(t_values[i], *skew_norm_params[i][j]) for i in range(n_classes) for j in range(n_classes)]
    return np.array(gradient_function(*gradient_arguments))


def calculate_itr_and_gradient(t_values, itr_function, gradient_function, skew_norm_params, n_classes):
    return (
        -calculate_itr(t_values, itr_function, skew_norm_params, n_classes),
        -calculate_gradient(t_values, gradient_function, skew_norm_params, n_classes)
    )


def predict(features, thresholds, n_classes, margin=0):
    predictions = []
    for sample_features in features:
        predicted = n_classes+1
        for i in range(len(sample_features)):
            if all(map(lambda arg: arg[1][0] >= arg[1][1]*(1+margin) if i == arg[0] else arg[1][0] < arg[1][1]*(1-margin), enumerate(zip(sample_features, thresholds)))):
                predicted = i+1
                break
        predictions.append(predicted)
    return predictions


def standard_itr_per_trial(a, n_classes):
    if a == 1:
        return np.log2(n_classes)
    elif a == 0:
        return np.log2(n_classes)+np.log2(1.0/(n_classes-1))
    else:
        return np.log2(n_classes)+a*np.log2(a)+(1-a)*np.log2((1.0-a)/(n_classes-1))


def mdt_from_prediction_prob(p, window_length, step_length):
    return window_length + (1.0/p - 1)*step_length


def standard_itr_from_confusion_matrix(confusion_matrix, window_length, step_length, n_classes):
    a = accuracy_from_confusion_matrix(confusion_matrix)
    itr = standard_itr_per_trial(a, n_classes)
    p = prediction_probability_from_confusion_matrix(confusion_matrix)
    mdt = mdt_from_prediction_prob(p, window_length, step_length)
    if p == 0:
        return 0
    else:
        return itr*60.0/mdt


def mi_from_confusion_matrix(confusion_matrix, n_classes):
    c_and_p = confusion_matrix/confusion_matrix.sum()
    m = prediction_probability_from_confusion_matrix(confusion_matrix)
    c_and_p = c_and_p/m
    c_and_p = np.delete(c_and_p, n_classes, 1)
    c_and_p = np.delete(c_and_p, n_classes, 0)
    p = c_and_p.sum(axis=0)
    c = c_and_p.sum(axis=1)
    itr = 0
    for i in range(n_classes):
        for j in range(n_classes):
            if c_and_p[i][j] != 0:
                itr += c_and_p[i][j]*np.log2(c_and_p[i][j]/(c[i]*p[j]))
    return itr


def itr_mi_from_confusion_matrix(confusion_matrix, window_length, step_length, n_classes):
    p = prediction_probability_from_confusion_matrix(confusion_matrix)
    mdt = mdt_from_prediction_prob(p, window_length, step_length)
    mi = mi_from_confusion_matrix(confusion_matrix, n_classes)
    return mi*60/mdt


def accuracy_from_confusion_matrix(confusion_matrix):
    return np.trace(confusion_matrix)/(confusion_matrix.sum()-confusion_matrix.sum(axis=0)[-1])


def prediction_probability_from_confusion_matrix(confusion_matrix):
    return (confusion_matrix.sum()-confusion_matrix.sum(axis=0)[-1])/confusion_matrix.sum()


def read_data(subject, recordings, data_folder):
    all_data_for_subject = []
    labels_for_subject = []
    for recording in recordings:
        input_file_name = os.path.join(os.pardir, data_folder, "feature_data", "sub" + subject + "rec" + recording + ".csv")
        data = pd.read_csv(input_file_name)
        features = data.iloc[:, 1:16].to_numpy()
        labels = data["label"].to_numpy()
        labels = np.delete(labels, list(range(113, 120)) + list(range(233, 240)), axis=0)
        features = np.delete(features, list(range(113, 120)) + list(range(233, 240)), axis=0)

        all_data_for_subject.append(features)
        labels_for_subject.append(labels)

    all_data_for_subject = np.array(all_data_for_subject)
    labels_for_subject = np.array(labels_for_subject)
    return all_data_for_subject, labels_for_subject


def calculate_ratios(data_row, n_classes):
    return np.array(
        [data_row[i:i + n_classes] / np.sum(data_row[i:i + n_classes]) for i in range(0, len(data_row), n_classes)]
    ).flatten()


def scale_data(all_data_for_subject, training_data, test_data, n_features, n_classes):
    for i in range(0, n_features, n_classes):
        minf = np.min(all_data_for_subject[:, :, i:i + n_classes])
        maxf = np.max(all_data_for_subject[:, :, i:i + n_classes])
        training_data[:, :, i:i + n_classes] = (training_data[:, :, i:i + n_classes] - minf) / (maxf - minf) + 1
        test_data[:, i:i + n_classes] = (test_data[:, i:i + n_classes] - minf) / (maxf - minf) + 1
    return training_data, test_data


def add_ratios_as_features(training_data, test_data, n_trials, n_samples, n_features, n_classes, add_ratios):
    if add_ratios:
        training_ratios = np.array(list(map(lambda x: calculate_ratios(x, n_classes), training_data.reshape(-1, n_features))))
        test_ratios = np.array(list(map(lambda x: calculate_ratios(x, n_classes), test_data.reshape(-1, n_features))))

        n_new_features = n_features * 2

        training_data = np.concatenate([training_data, training_ratios.reshape((n_trials - 1, n_samples, n_features))], axis=2)
        test_data = np.concatenate([test_data, test_ratios], axis=1)
    else:
        n_new_features = n_features
    return n_new_features, training_data, test_data


def apply_lda(training_data, training_labels, test_data, feature_selector, lda_model, n_new_features, n_trials, do_lda, do_lda_separately):
    if do_lda:
        if do_lda_separately:
            lda_features = []
            for i in range(n_trials - 1):
                lda_training_data = np.delete(training_data, i, 0).reshape(-1, n_new_features)
                lda_training_labels = np.delete(training_labels, i, 0).reshape(-1)
                lda_prediction_data = training_data[i, :, :].reshape(-1, n_new_features)
                feature_selector.fit(lda_training_data, lda_training_labels)
                lda_model.fit(feature_selector.transform(lda_training_data), lda_training_labels)
                lda_features.append(lda_model.decision_function(feature_selector.transform(lda_prediction_data)))
            lda_features = np.array(lda_features)

            new_train_features = lda_features
            feature_selector.fit(training_data.reshape(-1, n_new_features), training_labels.reshape(-1))
            lda_model.fit(feature_selector.transform(training_data.reshape(-1, n_new_features)), training_labels.reshape(-1))
            new_test_features = lda_model.decision_function(feature_selector.transform(test_data.reshape(-1, n_new_features)))
        else:
            lda_training_data = training_data.reshape(-1, n_new_features)
            lda_training_labels = training_labels.reshape(-1)
            feature_selector.fit(lda_training_data, lda_training_labels)
            lda_model.fit(feature_selector.transform(lda_training_data), lda_training_labels)
            new_train_features = lda_model.decision_function(feature_selector.transform(lda_training_data))
            new_test_features = lda_model.decision_function(feature_selector.transform(test_data.reshape(-1, n_new_features)))
        return new_train_features, new_test_features
    else:
        return training_data, test_data


def calculate_skew_norm_params(train_features_given_class, n_classes):
    return np.array([
            [
                skewnorm.fit(train_features_given_class[class_i,:,feature_i])
                for feature_i in range(n_classes)
            ] for class_i in range(n_classes)
        ])


def plot_histogram_and_skew_norm(train_features_given_class, skew_norm_params, n_classes, do_skew_norm_plots):
    if do_skew_norm_plots:
        all_min = np.min(train_features_given_class)
        all_max = np.max(train_features_given_class)
        for class_i in range(n_classes):
            for feature_i in range(n_classes):
                plt.subplot(3, 3, 1 + feature_i + n_classes * class_i)
                plt.hist(train_features_given_class[class_i, :, feature_i], density=True)
                minf = np.min(train_features_given_class[class_i, :, feature_i])
                maxf = np.max(train_features_given_class[class_i, :, feature_i])
                x = np.linspace(minf, maxf)
                plt.plot(x, skewnorm.pdf(x, *skew_norm_params[class_i][feature_i]))
                plt.xlim((all_min, all_max))
                plt.ylim((0, 0.5))
                plt.title("C=" + str(class_i + 1) + " F=" + str(feature_i + 1))
        plt.show()


def train_classifier(new_train_features, itr_function, gradient_function, skew_norm_params, subject, n_classes):
    itrs = []
    thresholds = []
    max_feature = np.max(new_train_features, axis=0)
    min_feature = np.min(new_train_features, axis=0)
    for i in range(3 if subject == "4" else 1):
        # initial_guess = np.mean(skew_norm_params[:, :, 1], axis=0)
        initial_guess = skew_norm_params[i, :, 1]
        result = basinhopping(
            calculate_itr_and_gradient,
            initial_guess,
            minimizer_kwargs={"method": "L-BFGS-B", "jac": True,
                              "args": (itr_function, gradient_function, skew_norm_params, n_classes)},
            niter=200,
            stepsize=np.max(max_feature - min_feature) / 20
        )
        thresholds.append(result.x)
        itrs.append(-result.fun)
    return itrs, thresholds


def evaluate_performance(new_test_features, best_thresholds, test_labels, n_classes, window_length, step_length, n_samples, treat_as_online):
    if treat_as_online:

        test_predictions = []
        test_correct_labels = []
        i = 0
        while i < len(test_labels):
            current_features = new_test_features[i]
            current_label = test_labels[i]
            test_prediction = predict([current_features], best_thresholds, n_classes)[0]
            if test_prediction != n_classes+1:
                test_predictions.append(test_prediction)
                test_correct_labels.append(current_label)
                i += int(window_length/step_length)
                if i < len(test_labels) and test_labels[i] != current_label:
                    i = np.where(test_labels == test_labels[i])[0][0]
            else:
                test_predictions.append(test_prediction)
                test_correct_labels.append(current_label)
                i += 1
        test_confusion_matrix = sklearn.metrics.confusion_matrix(test_correct_labels, test_predictions, labels=[i+1 for i in range(n_classes+1)])

        prediction_count = np.sum(np.array(test_predictions) != n_classes+1)
        accuracy = accuracy_from_confusion_matrix(test_confusion_matrix)
        mdt = (n_samples*step_length+(window_length-step_length)*n_classes)/prediction_count
        mi_itr = mi_from_confusion_matrix(test_confusion_matrix, n_classes)*60/mdt
        standard_itr = (standard_itr_per_trial(accuracy, n_classes)*60/mdt)
    else:
        test_predict = predict(new_test_features, best_thresholds, n_classes)
        test_confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, test_predict, labels=[i + 1 for i in range(n_classes + 1)])

        accuracy = accuracy_from_confusion_matrix(test_confusion_matrix)
        prediction_probability = prediction_probability_from_confusion_matrix(test_confusion_matrix)
        mi_itr = itr_mi_from_confusion_matrix(test_confusion_matrix, window_length, step_length, n_classes)
        standard_itr = standard_itr_from_confusion_matrix(test_confusion_matrix, window_length, step_length, n_classes)
        mdt = mdt_from_prediction_prob(prediction_probability, window_length, step_length)
        prediction_count = test_confusion_matrix.sum() - test_confusion_matrix.sum(axis=0)[-1]

    return mi_itr, standard_itr, accuracy, mdt, prediction_count
