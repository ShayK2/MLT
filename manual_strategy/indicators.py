import util
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def author():
    return 'akarthik3'

def sma(prices, windowSize = 10):
    SMA = prices.rolling(window = windowSize, min_periods = 1).mean()
    normalizedSMA = SMA / SMA.iloc[0]
    normalizedPrices = prices / prices.iloc[0]
    ratio = normalizedPrices / normalizedSMA

    figure, axes = plt.subplots()
    axes.set(xlabel = 'Time', ylabel = "Price", title = "Price to SMA Ratio")
    axes.plot(normalizedPrices, label = "Normalized Prices")
    axes.plot(normalizedSMA, label = "Normalized SMA")
    axes.plot(ratio, label = "Price/SMA")
    axes.legend()
    figure.savefig('SMA.png')
    plt.clf()
    return SMA

def bb(prices, windowSize = 10):
    normalizedPrices = prices / prices.iloc[0]
    sma = normalizedPrices.rolling(window = windowSize, min_periods = 1).mean()
    stDev = normalizedPrices.rolling(window = windowSize, min_periods = 1).std()
    topBand, bottomBand = sma + 2 * stDev, sma - 2 * stDev
    bb = (normalizedPrices - sma) / (2 * stDev)

    figure, axes = plt.subplots()
    axes.set(xlabel = 'Time', ylabel = "Price", title = "Bollinger Bands")
    axes.plot(normalizedPrices, label = "Normalized Prices")
    axes.plot(topBand, label = "Top Band")
    axes.plot(bottomBand, label = "Bottom Band")
    axes.plot(sma, label = "SMA")
    axes.plot(bb / 5, label = "BB Value (Scaled Down)")
    axes.legend()
    figure.savefig('Bollinger.png')
    plt.clf()
    return bb

def momentum(prices, windowSize = 10):
    normalizedPrices = prices / prices.iloc[0]
    momentum = pd.DataFrame(data = 0, index = prices.index, columns = ['Momentum'])
    for index in range(windowSize, momentum.size): momentum.iloc[index] = (normalizedPrices.iloc[index]['JPM'] / normalizedPrices.iloc[index - windowSize]['JPM']) - 1

    figure, axes = plt.subplots()
    axes.set(xlabel = 'Time', ylabel = "Price", title = "Momentum")
    axes.plot(normalizedPrices, label = "Normalized Prices")
    axes.plot(momentum, label = "Momentum")
    axes.legend()
    figure.savefig('Momentum.png')
    plt.clf()
    return momentum

if __name__ == "__main__":
    symbols = ['JPM']
    prices = util.get_data(symbols, pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))[symbols]

    sma(prices)
    bb(prices)
    momentum(prices)