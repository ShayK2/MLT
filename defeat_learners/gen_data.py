"""  		   	  			  	 		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
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

import numpy as np

# Returns data set (X & Y) that works better for linear regression than for decision trees
def best4LinReg(seed = 1489683273):
    np.random.seed(seed)
    numRows = np.random.randint(10, 1001)
    X = np.random.rand(numRows, np.random.randint(2, 11))
    Y = np.zeros(numRows)
    # Make Y values sum of each row --> LinReg is trivial but DT is hard b/c hard to find good splits
    for row in range(numRows): Y[row] = X[row, :].sum()
    return X, Y

# Returns data set (X & Y) that works better for decision trees than for linear regression
def best4DT(seed = 1489683273):
    np.random.seed(seed)
    numRows = np.random.randint(10, 1001)
    X = np.random.rand(numRows, np.random.randint(2, 11))
    Y = np.zeros(numRows)
    # Y values are 1 of 2 choices --> DT more likely to find good splits, but LinReg will have hard time with coeffs
    for row in range(numRows): Y[row] = -1 if X[row, 0] < 0.5 else 1
    return X, Y

def author():
    return 'akarthik3'

if __name__=="__main__":
    print("they call me Tim.")