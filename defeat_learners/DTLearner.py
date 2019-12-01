import numpy as np

class DTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size, self.verbose, self.dTree = leaf_size, verbose, None

    def author(self):
        return 'akarthik3'

    def addEvidence(self, dataX, dataY):
        tree = self.buildTree(dataX, dataY)

        if self.dTree: self.dTree = np.vstack((self.dTree, tree))
        else: self.dTree = tree

    def query(self, points):
        result = []
        for ind in points: result.append(self.find(ind, 0))
        return np.asarray(result)

    def buildTree(self, dataX, dataY):
        # Return a leaf if the set is small enough to be a leaf, or if there's only one unique y value
        if dataX.shape[0] <= self.leaf_size: return np.array([-1, dataY.mean(), np.nan, np.nan])
        if len(np.unique(dataY)) == 1: return np.array([-1, dataY.mean(), np.nan, np.nan])

        # Choose feature with highest absolute value correlation
        correlations = []
        for i in range(dataX.shape[1]):
            corr = np.corrcoef(dataX[:, i], dataY)
            correlations.append((i, abs(corr[0, 1])))

        # Choose the index of the highest x correlation
        secondElems = [corr[1] for corr in correlations]
        maxInd = secondElems.index(max(secondElems))
        best = correlations[maxInd][0]
        split = np.median(dataX[:, best])

        # Create a leaf if all values in best feature are below split value, otherwise make left/right trees and root
        if (np.all(dataX[:, best] <= split)): return np.array([-1, dataY.mean(), np.nan, np.nan])

        left = self.buildTree(dataX[dataX[:, best] <= split], dataY[dataX[:, best] <= split])
        right = self.buildTree(dataX[dataX[:, best] > split], dataY[dataX[:, best] > split])
        # Special case for if leftTree has one dimension, make the right index right after it
        root = np.array([best, split, 1, 2 if left.ndim == 1 else left.shape[0] + 1])

        return np.vstack((root, left, right))

    # Find the point in the tree where the input is found
    def find(self, test, row):
        feature, split = self.dTree[row, 0], self.dTree[row, 1]
        if feature < 0: return split
        if test[int(feature)] <= split: return self.find(test, row + int(self.dTree[row, 2])) # Left tree
        return self.find(test, row + int(self.dTree[row, 3])) # Right tree