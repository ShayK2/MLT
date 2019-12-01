import datetime as dt
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
import StrategyLearner as sl
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Akshay Karthik
# akarthik3

def author():
    return 'akarthik3'

if __name__ == '__main__':
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    tradeNumbers = []

    # Create learners + add data to plots for impact from 0 to 0.1 in steps of 0.02
    for index in range(6):
        learner = sl.StrategyLearner(impact = 0.02 * index)
        learner.addEvidence('JPM', sd, ed)
        trades = learner.testPolicy('JPM', sd, ed)

        # Calculate final normalized portfolio value produced by this trained learner
        learnerPortvals = compute_portvals(trades, commission = 0.0, impact = 0.02 * index)[0]
        plt.plot(learnerPortvals / learnerPortvals.iloc[0], label = 0.02 * index)

        # Count number of trades made by this trained learner
        numTrades = 0
        for index in range(1, trades.shape[0]): numTrades += trades.iloc[index]['JPM'] != 0
        tradeNumbers.append(numTrades)

    plt.title("Effect of Impact on Portfolio Value")
    plt.xlabel("Dates")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.savefig("Exp2-1.png")
    plt.clf()

    plt.bar(['0', '0.02', '0.04', '0.06', '0.08', '0.1'], tradeNumbers)
    plt.title("Effect of Impact on Number of Trades")
    plt.xlabel("Impact Value")
    plt.ylabel("Number of Trades")
    plt.savefig("Exp2-2.png")
    plt.clf()