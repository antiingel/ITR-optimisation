

class ScalingFunctions(object):
    def __init__(self):
        self.minima = None
        self.maxima = None
        self.scaling_functions = None

    def setup(self, *args):
        raise NotImplementedError("setup not implemented!")

    def __getitem__(self, item):
        return self.scaling_functions[item]

    def getScalingFunction(self, minimum, maximum):
        return lambda x: (x-minimum)/(maximum-minimum)+1

    def getScalingFunctions(self, minima, maxima, extraction_method_names):
        self.minima = minima
        self.maxima = maxima
        return {method: self.getScalingFunction(minima[method], maxima[method]) for method in extraction_method_names}


class TrainingScalingFunctions(ScalingFunctions):
    def __init__(self):
        ScalingFunctions.__init__(self)

    def setup(self, extraction_method_names, recordings):
        min_max_finder = MinMaxFinder(extraction_method_names)
        minima = min_max_finder.findMin(recordings)
        maxima = min_max_finder.findMax(recordings)
        self.scaling_functions = self.getScalingFunctions(minima, maxima, extraction_method_names)


class OnlineScalingFunctions(ScalingFunctions):
    def __init__(self):
        ScalingFunctions.__init__(self)

    def setup(self, minima, maxima, extraction_method_names):
        self.scaling_functions = self.getScalingFunctions(minima, maxima, extraction_method_names)


class MinMaxFinder(object):
    def __init__(self, extraction_method_names):
        self.extraction_method_names = extraction_method_names

    def getMethodFromFeature(self, feature):
        return "_".join(feature.split("_")[:-1])

    def iterateColumns(self, columns, extraction_method_names):
        for key in sorted(columns):
            method = self.getMethodFromFeature(key)
            if method in extraction_method_names:
                yield method, columns[key]

    def findMin(self, recordings):
        return self.findExtremum(min, recordings)

    def findMax(self, recordings):
        return self.findExtremum(max, recordings)

    def getColumns(self, list_of_dicts):
        columns = {}
        for dict in list_of_dicts:
            for key in dict:
                if key in columns:
                    columns[key].append(float(dict[key]))
                else:
                    columns[key] = [float(dict[key])]
        return columns

    def findExtremum(self, function, recordings):
        extrema = {method: [] for method in self.extraction_method_names}
        for recording in recordings:
            for method, column in self.iterateColumns(self.getColumns(recording), self.extraction_method_names):
                extrema[method].append(function(column))
        return {method: function(extrema[method]) for method in self.extraction_method_names}
