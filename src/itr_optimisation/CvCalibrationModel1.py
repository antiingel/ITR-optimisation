from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFpr
import DataCollectors, FeaturesHandler, MatrixBuilder, ScalingFunction

import numpy as np


class Model(object):
    def __init__(self):
        self.model = None
        self.labels = None
        self.matrix_builders = None
        self.extraction_method_names = None
        self.scaling_functions = None

    def getMethodFromFeature(self, feature):
        return "_".join(feature.split("_")[:-1])

    def iterateColumns(self, columns, extraction_method_names):
        for key in sorted(columns):
            method = self.getMethodFromFeature(key)
            if method in extraction_method_names:
                yield method, columns[key]

    def fit(self, data, labels):
        self.model.fit(data, labels)

    def getOrderedLabels(self):
        return self.model.classes_

    def predict(self, data):
        return self.model.predict(data)

    def moving_average(self, a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def predictProba(self, data, n, normalise):
        if normalise:
            probas = map(lambda probas: list(probas[i]/sum(probas) for i in range(len(probas))), self.model.decision_function(data))
        else:
            probas = self.model.decision_function(data)
        return np.transpose(map(lambda x: self.moving_average(x, n), np.transpose(probas)))

    def splitThresholdPredict(self, scores, thresholds, margin):
        predictions = []
        thresholds = np.transpose(thresholds)
        for sample_scores in scores:
            predicted = None
            for i, class_thresholds in enumerate(thresholds):
                if all(map(lambda (j, (s, t)): s >= t*(1+margin) if j == 0 else s < t*(1-margin), enumerate(zip(sample_scores, class_thresholds)))):
                    predicted = i+1
                    break
            predictions.append(str(predicted))
        return predictions

    def thresholdPredict(self, scores, thresholds, margin):
        predictions = []
        for sample_scores in scores:
            predicted = None
            for i in range(len(sample_scores)):
                if all(map(lambda (j, (s, t)): s >= t*(1+margin) if i == j else s < t*(1-margin), enumerate(zip(sample_scores, thresholds)))):
                    predicted = i+1
                    break
            predictions.append(str(predicted))
        return predictions

    def thresholdPredict1(self, scores, thresholds, margin):
        predictions = []
        for sample_scores in scores:
            predicted = None
            for i in range(len(sample_scores)):
                if all(map(lambda (j, (s, t)): s >= t*(1+margin) if i == j else s < t*(1-margin), enumerate(zip(sample_scores, thresholds)))):
                    predicted = i+1
                    break
            else:
                predicted = np.argmax(sample_scores-thresholds)+1
            predictions.append(predicted)
        return predictions

    def buildRatioMatrix(self, data):
        matrices = [builder.buildRatioMatrix(self.iterateColumns(data, self.extraction_method_names)) for builder in self.matrix_builders]
        return np.concatenate(matrices, axis=1)

    def getMinMax(self):
        return self.scaling_functions.minima, self.scaling_functions.maxima


class TrainingModel(Model):
    def __init__(self):
        Model.__init__(self)
        self.features_to_use = None
        self.collector = None
        self.features_handler = None
        self.feature_selector = None
        self.do_feature_selection = None

    def setup(self, features_to_use, sample_count, recordings, matrix_builder_types):
        self.extraction_method_names = self.setupFeaturesHandler(features_to_use, recordings)
        self.setupScalingFunctions(self.extraction_method_names, recordings)
        self.feature_selector = SelectFpr(alpha=5e-2)
        self.model = LinearDiscriminantAnalysis()
        self.collector = DataCollectors.TrainingCollector(sample_count)
        self.setupCollectorAndBuilder(sample_count, self.scaling_functions, self.extraction_method_names, matrix_builder_types)

    def fit(self, data, labels):
        self.feature_selector.fit(data, labels)
        print len(self.feature_selector.get_support(True))
        if len(self.feature_selector.get_support(True)) == 0:
            self.do_feature_selection = False
        else:
            self.do_feature_selection = True
        Model.fit(self, self.selectFeatures(data), labels)

    def selectFeatures(self, data):
        if self.do_feature_selection:
            return self.feature_selector.transform(data)
        else:
            return data

    def predict(self, data):
        return Model.predict(self, self.selectFeatures(data))

    def softmax(sellf, x):
        a = 0.25
        return np.exp(x*a) / np.sum(np.exp(x*a), axis=0)

    def predictProba(self, data, n, normalise):
        return (Model.predictProba(self, self.selectFeatures(data), n, normalise))

    def setupScalingFunctions(self, extraction_method_names, recordings):
        self.scaling_functions = ScalingFunction.TrainingScalingFunctions()
        self.scaling_functions.setup(extraction_method_names, recordings)

    def setupCollectorAndBuilder(self, sample_count, scaling_functions, extraction_method_names, matrix_builder_types):
        self.collector = DataCollectors.TrainingCollector(sample_count)
        self.matrix_builders = []
        for type in matrix_builder_types:
            builder = MatrixBuilder.TrainingMatrixBuilder()
            builder.setup(scaling_functions, extraction_method_names, type)
            self.matrix_builders.append(builder)

    def setupFeaturesHandler(self, features_to_use, recordings):
        self.features_handler = FeaturesHandler.TrainingFeaturesHandler(recordings)
        self.features_handler.setup(features_to_use)
        self.features_to_use = self.features_handler.getUsedFeatures()
        return self.features_handler.getExtractionMethodNames()

    def collectSamples(self, features, labels):
        self.collector.reset()
        return self.collector.combineSamples(features, labels)

    def getColumns(self, list_of_dicts):
        columns = {}
        for dict in list_of_dicts:
            for key in dict:
                if key in columns:
                    columns[key].append(float(dict[key]))
                else:
                    columns[key] = [float(dict[key])]
        return columns

    def getAllLookBackRatioMatrices(self, recordings, labels):
        self.collector.reset()
        all_matrices = []
        all_labels = []
        for recording in recordings:
            ratio_matrix = self.buildRatioMatrix(self.getColumns(recording))
            look_back_ratio_matrix, labels = self.collectSamples(ratio_matrix, labels)
            all_matrices.append(look_back_ratio_matrix)
            all_labels.append(labels)
        return all_matrices, all_labels

    def getConcatenatedMatrix(self, recordings, labels):
        matrices, labels = self.getAllLookBackRatioMatrices(recordings, labels)
        data_matrix = np.concatenate(matrices, axis=0)
        data_labels = np.concatenate(labels, axis=0)
        return data_matrix, data_labels

    def getUsedFeatures(self):
        return self.features_to_use
