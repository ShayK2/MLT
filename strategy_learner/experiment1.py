import datetime as dt
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
import StrategyLearner as sl
import ManualStrategy as ms
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

    # Calculate normalized portfolio value of manual strategy's policy and strategy learner's learned policy, then plot them
    manualPortvals = compute_portvals(ms.testPolicy(sd = sd, ed = ed), commission = 0.0, impact = 0.0)[0]

    learner = sl.StrategyLearner()
    learner.addEvidence('JPM', sd, ed)
    learnerPortvals = compute_portvals(learner.testPolicy('JPM', sd, ed), commission = 0.0, impact = 0.0)[0]

    plt.title("Manual Strategy vs. Strategy Learner")
    plt.xlabel("Dates")
    plt.ylabel("Normalized Portfolio Value")

    plt.plot(manualPortvals / manualPortvals.iloc[0], label = "Manual")
    plt.plot(learnerPortvals / learnerPortvals.iloc[0], label = "Learner")
    plt.legend()
    plt.savefig("Exp1.png")
    plt.clf()