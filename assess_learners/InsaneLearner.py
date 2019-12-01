import BagLearner as bl
import LinRegLearner as lrl
import numpy as np

class InsaneLearner(object):

    def __init__(self, verbose = False):
        self.learners = []
        for ind in range(20): self.learners.append(bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False))

    def author(self):
        return 'akarthik3'

    def addEvidence(self, dataX, dataY):
        for learner in self.learners: learner.addEvidence(dataX, dataY)

    def query(self, points):
        result = []
        for learner in self.learners: result.append(learner.query(points))
        return np.mean(np.asarray(result))