import util
import pandas as pd
import datetime as dt

# Akshay Karthik
# akarthik3

def author():
    return 'akarthik3'

def sma(prices, windowSize = 10):
    return prices.rolling(window = windowSize, min_periods = 1).mean()

def bb(prices, windowSize = 10):
    normalizedPrices = prices / prices.iloc[0]
    stDev = normalizedPrices.rolling(window = windowSize, min_periods = 1).std()
    return (normalizedPrices - sma(normalizedPrices)) / (2 * stDev)

# Momentum = ratio of current price to price some number of days ago (default 10)
def momentum(prices, windowSize = 10):
    normalizedPrices = prices / prices.iloc[0]
    momentum = pd.DataFrame(data = 0, index = prices.index, columns = ['Momentum'])
    for index in range(windowSize, momentum.size): momentum.iloc[index] = (normalizedPrices.iloc[index][prices.columns.values[0]] / normalizedPrices.iloc[index - windowSize][prices.columns.values[0]]) - 1.0
    return momentum

if __name__ == "__main__":
    symbols = ['JPM']
    prices = util.get_data(symbols, pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))[symbols]

    sma(prices)
    bb(prices)
    momentum(prices)