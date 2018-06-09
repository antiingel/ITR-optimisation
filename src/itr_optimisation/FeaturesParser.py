


class ResultsParser(object):
    def __init__(self, add_sum=True, data_has_method=False):
        self.data = None
        self.parse_result = {}
        self.add_sum = add_sum
        self.data_has_method = data_has_method

    def setup(self, data):
        self.data = data

    def parseSensorResults(self, parse_result, results, data):
        for sensor in results:
            # if sensor in data:  # Currently data does not contain sensors??
                self.parseHarmonicResults(self.parseResultValue(parse_result, sensor), results[sensor], data)

    def parseHarmonicResults(self, parse_result, results, data):
        for harmonic in results:
            if harmonic in data:
                self.parseFrequencyResults(self.parseResultValue(parse_result, harmonic), results[harmonic], data[harmonic])

    def parseFrequencyResults(self, parse_result, result, data):
        raise NotImplementedError("parseFrequencyResult not implemented!")

    def parseResultValue(self, parse_result, key):
        if key not in parse_result:
            parse_result[key] = {}
        return parse_result[key]

    def parseResults(self, results):
        for tab in results:
            for method in results[tab]:
                if tab in self.data:
                    parse_result = self.parseResultValue(self.parse_result, tab)
                    parse_result = self.parseResultValue(parse_result, method)
                    if self.data_has_method:
                        data = self.data[tab][method]
                    else:
                        data = self.data[tab]
                    if method[0] == "CCA" or method[0] == "LRT":
                        if self.add_sum:
                            self.parseHarmonicResults(parse_result, {"SUM": results[tab][method]}, data)
                        else:
                            self.parseHarmonicResults(parse_result, results[tab][method], data)
                    elif method[0] == "SUM PSDA":
                        self.parseHarmonicResults(parse_result, results[tab][method], data)
                    elif method[0] == "PSDA":
                        self.parseSensorResults(parse_result, results[tab][method], data)
                    elif method[0] == "MEC":
                        self.parseHarmonicResults(parse_result, results[tab][method], data)
        return self.parse_result

    def parseResultsNewDict(self, results):
        self.parse_result = {}
        return self.parseResults(results)


def getMethodFromFeature(feature):
    return "_".join(feature.split("_")[:-1])

