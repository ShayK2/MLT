"""Assess a betting strategy.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
import matplotlib.pyplot as plt

def author():
        return 'akarthik3' # replace tb34 with your Georgia Tech username.

def gtid():
    return 903212846 # replace with your GT ID number

def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result

def test_code():
    plotlist = []
    win_prob = 9.0 / 19.0  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once

    for ind in range(1000):
        winnings = np.zeros(1000, dtype = int)

        episode_winnings = 0
        count = 0

        while episode_winnings < 80 and count < 999:
            won = False
            bet_amount = 1
            while not won and count < 999:
                count += 1
                won = get_spin_result(win_prob)
                if won: episode_winnings += bet_amount
                else:
                    episode_winnings -= bet_amount
                    bet_amount *= 2
                winnings[count] = episode_winnings
        if count < 999: winnings[count:1000] = episode_winnings
        plotlist.append(winnings)

    plt.axis([0, 300, -256, 100])
    for ind in range(10): plt.plot(plotlist[ind])
    plt.savefig('fig1.png')

    means = []
    medians = []
    meanUpOne = []
    meanDownOne = []
    medianUpOne = []
    medianDownOne = []
    for ind in range(1000):
        vals = [plot[ind] for plot in plotlist]
        mean = np.mean(vals)
        median = np.median(vals)
        std = np.std(vals)
        means.append(mean)
        meanUpOne.append(mean + std)
        meanDownOne.append(mean - std)
        medians.append(median)
        medianUpOne.append(median + std)
        medianDownOne.append(median - std)

    plt.clf()
    plt.axis([0, 300, -256, 100])
    plt.plot(means)
    plt.plot(meanUpOne)
    plt.plot(meanDownOne)
    plt.savefig('fig2.png')

    plt.clf()
    plt.axis([0, 300, -256, 100])
    plt.plot(medians)
    plt.plot(medianUpOne)
    plt.plot(medianDownOne)
    plt.savefig('fig3.png')

    plotlist = []
    for ind in range(1000):
        winnings = np.zeros(1000, dtype = int)

        episode_winnings = 0
        count = 0

        done = False
        while episode_winnings < 80 and not done:
            won = False
            bet_amount = 1
            while not won and not done:
                count += 1
                won = get_spin_result(win_prob)
                if won: episode_winnings += bet_amount
                else:
                    episode_winnings -= bet_amount
                    bet_amount *= 2
                    if (episode_winnings == -256): done = True
                    if (bet_amount > (episode_winnings + 256)): bet_amount = episode_winnings + 256
                if (count == 999): done = True
                winnings[count] = episode_winnings
        if count < 999: winnings[count:1000] = episode_winnings
        plotlist.append(winnings)

    means = []
    medians = []
    meanUpOne = []
    meanDownOne = []
    medianUpOne = []
    medianDownOne = []
    for ind in range(1000):
        vals = [plot[ind] for plot in plotlist]
        mean = np.mean(vals)
        median = np.median(vals)
        std = np.std(vals)
        means.append(mean)
        meanUpOne.append(mean + std)
        meanDownOne.append(mean - std)
        medians.append(median)
        medianUpOne.append(median + std)
        medianDownOne.append(median - std)

    plt.clf()
    plt.axis([0, 300, -256, 100])
    plt.plot(means)
    plt.plot(meanUpOne)
    plt.plot(meanDownOne)
    plt.savefig('fig4.png')

    plt.clf()
    plt.axis([0, 300, -256, 100])
    plt.plot(medians)
    plt.plot(medianUpOne)
    plt.plot(medianDownOne)
    plt.savefig('fig5.png')
    return

if __name__ == "__main__":
    test_code()