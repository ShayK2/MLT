"""  		   	  			  	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
"""

import numpy as np
import math
import time
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import util
import matplotlib.pyplot as plt
import sys


def author():
    return 'akarthik3'


if __name__ == "__main__":

    # Get istanbul.csv data and remove unwanted parts
    data = np.genfromtxt(util.get_learner_data_file('Istanbul.csv'), delimiter = ',')
    data = data[1:, 1:]

    # Separate data into training and testing (want every data point, split into 60-40 ratio between train/test)
    dataSize = int(0.6 * data.shape[0])
    xTraining = data[:dataSize, 0:-1]
    yTraining = data[:dataSize, -1]
    xTesting = data[dataSize:, 0:-1]
    yTesting = data[dataSize:, -1]

    # Test 1
    trainingRMSEs = np.zeros((100, 1))
    testingRMSEs = np.zeros((100, 1))
    for size in range(1, 101):
        learner = dt.DTLearner(size)
        learner.addEvidence(xTraining, yTraining)

        # training data
        yPreds = learner.query(xTraining)
        rmse = math.sqrt(((yTraining - yPreds) ** 2).sum() / yTraining.shape[0])
        trainingRMSEs[size - 1, 0] = rmse

        # testing data
        yPreds = learner.query(xTesting)
        rmse = math.sqrt(((yTesting - yPreds) ** 2).sum() / yTesting.shape[0])
        testingRMSEs[size - 1, 0] = rmse

    plt.plot(np.arange(1, 101), trainingRMSEs, label = "Training Data")
    plt.plot(np.arange(1, 101), testingRMSEs, label = "Testing Data")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Experiment 1 - Overfitting")
    plt.savefig("Exp1.png")
    plt.clf()

    # Test 2
    trainingRMSEs = np.zeros((100, 1))
    testingRMSEs = np.zeros((100, 1))
    for size in range(1, 101):
        learner = bl.BagLearner(dt.DTLearner, {"leaf_size" : size})
        learner.addEvidence(xTraining, yTraining)

        # training data
        yPreds = learner.query(xTraining)
        rmse = math.sqrt(((yTraining - yPreds) ** 2).sum() / yTraining.shape[0])
        trainingRMSEs[size - 1, 0] = rmse

        # testing data
        yPreds = learner.query(xTesting)
        rmse = math.sqrt(((yTesting - yPreds) ** 2).sum() / yTesting.shape[0])
        testingRMSEs[size - 1, 0] = rmse

    plt.plot(np.arange(1, 101), trainingRMSEs, label = "Training Data")
    plt.plot(np.arange(1, 101), testingRMSEs, label = "Testing Data")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Experiment 2 - Overfitting w/Bagging")
    plt.savefig("Exp2.png")
    plt.clf()

    # Test 3
    dtTimes = np.zeros((100, 1))
    for size in range(1, 101):
        start = time.time()
        learner = dt.DTLearner(size)
        learner.addEvidence(xTraining, yTraining)

        # training data
        yPreds = learner.query(xTraining)

        # testing data
        yPreds = learner.query(xTesting)
        end = time.time()
        dtTimes[size - 1, 0] = end - start


    rtTimes = np.zeros((100, 1))
    for size in range(1, 101):
        start = time.time()
        learner = rt.RTLearner(leaf_size = size, verbose = False)
        learner.addEvidence(xTraining, yTraining)

        # training data
        yPreds = learner.query(xTraining)

        # testing data
        yPreds = learner.query(xTesting)
        end = time.time()
        rtTimes[size - 1, 0] = end - start

    plt.plot(np.arange(1, 101), dtTimes, label = "DT Times")
    plt.plot(np.arange(1, 101), rtTimes, label = "RT Times")
    plt.xlabel("Leaf Size")
    plt.ylabel("Execution Time")
    plt.legend()
    plt.title("Experiment 3-1 - DT vs. RT on Training/Querying Time")
    plt.savefig("Exp3.png")
    plt.clf()

    # Test 4
    dtMAEs = np.zeros((100, 1))
    for size in range(1, 101):
        learner = dt.DTLearner(size)
        learner.addEvidence(xTraining, yTraining)

        # MAE using testing data
        yPreds = learner.query(xTesting)
        mae = ((np.absolute(yTesting - yPreds)).sum() / yTesting.shape[0])
        dtMAEs[size - 1, 0] = mae

    rtMAEs = np.zeros((100, 1))
    for size in range(1, 101):
        learner = rt.RTLearner(size)
        learner.addEvidence(xTraining, yTraining)

        # MAE using testing data
        yPreds = learner.query(xTesting)
        mae = ((np.absolute(yTesting - yPreds)).sum() / yTesting.shape[0])
        rtMAEs[size - 1, 0] = mae

    plt.plot(np.arange(1, 101), dtMAEs, label = "DT MAEs")
    plt.plot(np.arange(1, 101), rtMAEs, label = "RT MAEs")
    plt.xlabel("Leaf Size")
    plt.ylabel("MAE")
    plt.legend()
    plt.title("Experiment 3-2 - DT vs. RT on Mean Absolute Error")
    plt.savefig("Exp4.png")
    plt.clf()