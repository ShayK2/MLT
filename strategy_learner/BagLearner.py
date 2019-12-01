import RTLearner as rt
import numpy as np
from scipy import stats

# Akshay Karthik
# akarthik3

class BagLearner(object):

    def __init__(self, learner = rt.RTLearner, kwargs = {"leaf_size" : 1}, bags = 20, boost = False, verbose = False):
        self.kwargs, self.bags, self.boost, self.verbose, self.learners = kwargs, bags, boost, verbose, []
        for ind in range(0, bags): self.learners.append(learner(**self.kwargs))

    def author(self):
        return 'akarthik3'

    def addEvidence(self, dataX, dataY):
        for ind in range(0, self.bags):
            # Choose random set of indices in dataX of size len(dataX), replacements allowed --> train this learner on that data
            sample = np.random.choice(dataX.shape[0], dataX.shape[0])
            self.learners[ind].addEvidence(dataX[sample], dataY[sample])

    def query(self, points):
        results = []
        # Query all learners and average results
        for ind in range(0, self.bags): results.append(self.learners[ind].query(points))
        return stats.mode(np.asarray(results), axis = 0)[0]