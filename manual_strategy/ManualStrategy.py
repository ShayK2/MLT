from util import get_data
import datetime as dt
import pandas as pd
import numpy as np
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from indicators import sma, bb
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def author():
    return 'akarthik3'

def testPolicy(symbol = 'JPM', sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), sv = 100000):
    prices = get_data([symbol], pd.date_range(sd, ed))[symbol]

    priceSMA = prices / sma(prices)
    BB = bb(prices)

    # Decide order types based on price-to-SMA ratio and calculated Bollinger value
    orders = prices.copy()
    orders[:] = 0
    orders[(priceSMA > 1.02) & (BB > 1)] = -1
    orders[(priceSMA < 0.98) & (BB < 0)] = 1

    trades = []
    for date in orders.index: trades.append((date, orders.loc[date] * 1000))

    tradesDataframe = pd.DataFrame(trades, columns = ["Date", "Shares"])
    tradesDataframe.set_index("Date", inplace = True)

    return tradesDataframe

if __name__ == "__main__":
    benchmarkPrices = get_data(['JPM'], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))['JPM']
    # benchmarkPrices = get_data(['JPM'], pd.date_range(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)))['JPM']

    benchmarkTrades = np.zeros(len(benchmarkPrices.index))
    benchmarkTrades[0] = 1000
    benchmarkTrades = pd.DataFrame(data = benchmarkTrades, index = benchmarkPrices.index, columns = ['Shares'])

    benchmarkPortvals, benchmarkCR, benchmarkMean, benchmarkSTD = compute_portvals(benchmarkTrades, 100000, 0.0, 0.0)
    normalizedBenchmark = benchmarkPortvals / benchmarkPortvals.iloc[0]

    trades = testPolicy()
    # trades = testPolicy(sd = dt.datetime(2010, 1, 1), ed = dt.datetime(2011, 12, 31))

    optimalPortvals, optimalCR, optimalMean, optimalSTD = compute_portvals(trades, 100000, 0.0, 0.0)
    normalizedOptimal = optimalPortvals / optimalPortvals.iloc[0]

    print("Benchmark CR: ", benchmarkCR)
    print("Benchmark ADR: ", benchmarkMean)
    print("Benchmark SDDR: ", benchmarkSTD)
    print("Optimal CR: ", optimalCR)
    print("Optimal ADR: ", optimalMean)
    print("Optimal SDDR: ", optimalSTD)

    plt.title("Benchmark vs. Manual Strategy")
    plt.xlabel("Dates")
    plt.ylabel("Normalized Value of Portfolio")
    plt.plot(normalizedBenchmark, 'g', label="Benchmark")
    plt.plot(normalizedOptimal, 'r', label = "Manual")
    plt.legend()

    prev = 0
    for day, order in trades.iterrows():
        if order['Shares'] - prev < 0 and order['Shares'] < 0:  # SHORT
            plt.axvline(day, color = "k", alpha = 0.5)
        elif order['Shares'] - prev > 0 and order['Shares'] > 0:  # LONG
            plt.axvline(day, color = "b", alpha = 0.5)
        prev = order['Shares']

    plt.savefig("Manual.png")
    # plt.savefig("ManualOutOfSample.png")
    plt.clf()