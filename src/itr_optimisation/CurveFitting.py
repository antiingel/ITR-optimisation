import matplotlib.pyplot as plt


class AbstractCurveFitting(object):
    def __init__(self):
        self.curves = []

    def fitCurves(self, *args):
        raise NotImplementedError("fitCurves not implemented!")

    def extract(self, curves, extractor):
        return [extractor(curve) for curve in curves]

    def functionExtractor(self, curve):
        return curve.fit_function

    def derivativeExtractor(self, curve):
        return curve.fit_function_derivative

    def extractFunctions(self, curves=None):
        return self.extract(self.curves if curves is None else curves, self.functionExtractor)

    def extractDerivatives(self, curves=None):
        return self.extract(self.curves if curves is None else curves, self.derivativeExtractor)


class CurveFitting(AbstractCurveFitting):
    def __init__(self):
        AbstractCurveFitting.__init__(self)

    def plotCurves(self, classes):
        for i, curve in enumerate(self.curves):
            if i in classes:
                plt.figure()
                plt.subplot(121)
                curve.plotFunction()
                plt.subplot(122)
                curve.plotDerivative()

    def fitCurves(self, *args):
        self.calculateCurves(*args)
        # self.plotCurves([0,1,2])
        return (
            self.extractFunctions(),
            self.extractDerivatives(),
        )

    def getCurve(self, *args):
        raise NotImplementedError("getCurve not implemented!")

    def calculateCurves(self, *args):
        self.curves = []
        for arg in zip(*args):
            self.curves.append(self.getCurve(*arg))


class Curve(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.fit_function = None
        self.fit_function_derivative = None

    def plotFunction(self):
        raise NotImplementedError("plotFunction not implemented!")

    def plotDerivative(self):
        raise NotImplementedError("plotDerivative not implemented!")
