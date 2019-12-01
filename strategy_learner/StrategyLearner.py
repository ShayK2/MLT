"""  		   	  			    		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
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
  		   	  			    		  		  		    	 		 		   		 		  
Student Name: Akshay Karthik
GT User ID: akarthik3
GT ID: 903212846
"""

from util import get_data
import datetime as dt
import pandas as pd
import numpy as np
from indicators import sma, bb
import BagLearner as bl
import RTLearner as rt

class StrategyLearner(object):

    def __init__(self, verbose = False, impact = 0.0):
        self.verbose, self.impact = verbose, impact

    def author(self):
        return 'akarthik3'

    def addEvidence(self, symbol = "AAPL", sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), sv = 100000):
        leaf_size, bags, days, yBuy, ySell = 5, 20, 10, 0.04, -0.04

        prices = get_data([symbol], pd.date_range(sd, ed))[[symbol]]
        nDayReturns = (prices.shift(-days) / prices) - 1.0

        # Create x data for training by combining price/SMA and Bollinger calculations
        # Decided not to use momentum since it is not very reliable in predicting price trends
        dataX = pd.concat([prices / sma(prices), bb(prices)], axis = 1)[:-days].values

        # Every entry in y data is -1, 0, or 1 based on how associated nDayReturns values compares to buy/sell thresholds
        dataY = []
        for index, row in nDayReturns.iterrows():
            entry = row[nDayReturns.columns.values[0]]
            dataY.append(1.0 if entry > (self.impact + yBuy) else (-1.0 if entry < (ySell - self.impact) else 0.0))

        self.learner = bl.BagLearner(rt.RTLearner, {'leaf_size' : leaf_size}, bags, False, False)

        self.learner.addEvidence(dataX, np.asarray(dataY))

    def testPolicy(self, symbol = "AAPL", sd = dt.datetime(2010, 1, 1), ed = dt.datetime(2011, 12, 31), sv = 100000):
        prices = get_data([symbol], pd.date_range(sd, ed))[[symbol]]

        trades = pd.DataFrame(0.0, prices.index, [symbol])

        dataX = pd.concat([prices / sma(prices), bb(prices)], axis = 1).values

        queryResults = self.learner.query(dataX)

        # Update values in trades based on holdings at each step and output from query to learner
        holdings = 0.0
        for i in range(trades.shape[0]):
            if abs(queryResults[0][i]) < 0.5: trades[symbol].iloc[i] = 0.0
            else: trades[symbol].iloc[i] = 1000.0 - holdings - (0 if queryResults[0][i] >= 0.5 else 2000)
            holdings += trades[symbol].iloc[i]

        return trades

if __name__=="__main__":
    print ("One does not simply think up a strategy")