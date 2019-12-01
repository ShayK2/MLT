from util import get_data
import datetime as dt
import pandas as pd
import numpy as np
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def author():
    return 'akarthik3'

def testPolicy(symbol = 'JPM', sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), sv = 100000):
    prices = get_data([symbol], pd.date_range(sd, ed))[symbol]

    # Turn the price changes into signs for increase/decrease, and convert those to order types (while maintaining DF)
    priceRelations = pd.Series(np.nan, index = prices.index)
    priceRelations[:-1] = prices[:-1] / prices.values[1:] - 1
    signs = priceRelations.apply(np.sign)
    orders = -1 * signs.diff() / 2
    orders[0] = signs[0]

    trades = []
    for date in orders.index: trades.append((date, orders.loc[date] * 1000))

    tradesDataframe = pd.DataFrame(trades, columns = ["Date", "Shares"])
    tradesDataframe.set_index("Date", inplace = True)

    return tradesDataframe

if __name__ == "__main__":
    benchmarkPrices = get_data(['JPM'], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))['JPM']

    benchmarkTrades = np.zeros(len(benchmarkPrices.index))
    benchmarkTrades[0] = 1000
    benchmarkTrades = pd.DataFrame(data = benchmarkTrades, index = benchmarkPrices.index, columns = ['Shares'])

    benchmarkPortvals, benchmarkCR, benchmarkMean, benchmarkSTD = compute_portvals(benchmarkTrades, 100000, 0.0, 0.0)
    normalizedBenchmark = benchmarkPortvals / benchmarkPortvals.iloc[0]

    trades = testPolicy()

    optimalPortvals, optimalCR, optimalMean, optimalSTD = compute_portvals(trades, 100000, 0.0, 0.0)
    normalizedOptimal = optimalPortvals / optimalPortvals.iloc[0]

    print("Benchmark CR: ", benchmarkCR)
    print("Benchmark ADR: ", benchmarkMean)
    print("Benchmark SDDR: ", benchmarkSTD)
    print("Optimal CR: ", optimalCR)
    print("Optimal ADR: ", optimalMean)
    print("Optimal SDDR: ", optimalSTD)

    plt.title("Benchmark vs. Theoretically Optimal Strategy")
    plt.xlabel("Dates")
    plt.ylabel("Normalized Portfolio Value")
    plt.plot(normalizedBenchmark, 'g', label = "Benchmark")
    plt.plot(normalizedOptimal, 'r', label = "Optimal")
    plt.legend()
    plt.savefig("TheoreticallyOptimal.png")
    plt.clf()