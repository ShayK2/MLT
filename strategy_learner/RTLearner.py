import numpy as np
import random
from scipy import stats

# Akshay Karthik
# akarthik3

class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size, self.verbose, self.rTree = leaf_size, verbose, None

    def author(self):
        return 'akarthik3'

    def addEvidence(self, dataX, dataY):
        tree = self.buildTree(dataX, dataY)

        if self.rTree: self.rTree = np.vstack((self.rTree, tree))
        else: self.rTree = tree

    def query(self, points):
        result = []
        for ind in points: result.append(self.find(ind, 0))
        return np.asarray(result)

    def buildTree(self, dataX, dataY):
        # Return a leaf if the set is small enough to be a leaf, or if there's only one unique y value
        if dataX.shape[0] <= self.leaf_size: return np.array([-1, stats.mode(dataY, axis = None)[0].mean(), np.nan, np.nan])
        if len(np.unique(dataY)) == 1: return np.array([-1, stats.mode(dataY, axis = None)[0].mean(), np.nan, np.nan])

        # Choose best feature at random
        best = random.randint(0, dataX.shape[1] - 1)
        split = np.median(dataX[:, best])

        # Create a leaf if all values in best feature are below split value, otherwise make left/right trees and root
        if (np.all(dataX[:, best] <= split)): return np.array([-1, stats.mode(dataY, axis = None)[0].mean(), np.nan, np.nan])

        left = self.buildTree(dataX[dataX[:, best] <= split], dataY[dataX[:, best] <= split])
        right = self.buildTree(dataX[dataX[:, best] > split], dataY[dataX[:, best] > split])
        # Special case for if leftTree has one dimension, make the right index right after it
        root = np.array([best, split, 1, 2 if left.ndim == 1 else left.shape[0] + 1])

        return np.vstack((root, left, right))

    # Find the point in the tree where the input is found
    def find(self, test, row):
        feature, split = self.rTree[row, [0, 1]]
        if feature < 0: return split
        if test[int(feature)] <= split: return self.find(test, row + int(self.rTree[row, 2])) # Left tree
        return self.find(test, row + int(self.rTree[row, 3])) # Right tree