"""MC1-P2: Optimize a portfolio.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import datetime as dt
from util import get_data, plot_data

def optimize_portfolio(sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 1, 1), \
    syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot = False):

    startValue = 1000000
    tradeFrequency = 252.0

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    # Fill in missing data
    prices_all.fillna(method = "ffill")
    prices_all.fillna(method = "bfill")

    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Find the allocations for the optimal portfolio

    # Initialize allocations array with same allocation for each stock (initial guess) & normalize prices to first entries
    allocs = np.asarray([(1.0 / len(syms)) for i in range(len(syms))])
    normalized = prices / prices.values[0]

    # Optimize allocation to minimize negative of Sharpe ratio equation (maximize Sharpe ratio)
    bounds = [(0.0, 1.0) for i in range(len(syms))]
    constraints = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    minimized = opt.minimize(negSharpe, allocs, args = (normalized, startValue), method = 'SLSQP', constraints = constraints,
                             bounds = bounds, options = {'disp': True})

    # Calculate portfolio value to use in calculation of return values
    new_allocs = minimized.x
    new_sddr = minimized.fun
    allocations = normalized.multiply(new_allocs)
    position_vals = allocations.multiply(startValue)
    portfolio_val = position_vals.sum(axis = 1)

    # Get daily portfolio values, cumulative value, and Sharpe ratio
    daily = (portfolio_val / portfolio_val.shift(1)) - 1
    cumulative = (portfolio_val[-1] / portfolio_val[0]) - 1
    sharpe = np.sqrt(tradeFrequency) * daily.mean() / new_sddr

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        portfolio_val /= portfolio_val[0]
        prices_SPY /= prices_SPY[0]
        df_temp = pd.concat([portfolio_val, prices_SPY], keys = ['Portfolio', 'SPY'], axis = 1)
        plot_data(df_temp, title = "Optimal Portfolio vs. SPY Daily Comparison", ylabel = "Normalized Price")
        plt.savefig('plot.png')
        pass

    return new_allocs, cumulative, daily.mean(), new_sddr, sharpe

def negSharpe(allocs, normalized, sv):
    allocations = normalized.multiply(allocs)
    position_vals = allocations.multiply(sv)
    portfolio_val = position_vals.sum(axis = 1)
    daily = (portfolio_val / portfolio_val.shift(1)) - 1
    return -1 * (np.mean(daily) / daily.std())

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print(f"Start Date: {start_date}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		   	  			  	 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		   	  			  	 		  		  		    	 		 		   		 		  
    test_code()  		   	  			  	 		  		  		    	 		 		   		 		  
