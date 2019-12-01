import pandas as pd
from util import get_data

def author():
    return 'akarthik3'

def compute_portvals(orders, start_val = 1000000, commission = 9.95, impact = 0.005):
    orders.sort_index(ascending = True, inplace = True)
    dates = pd.date_range(orders.index.min(), orders.index.max())

    symbols = ['JPM']
    prices = get_data(symbols, dates)[symbols]
    prices['CASH'] = 1.0
    trades = prices.copy()
    trades[:] = 0
    trades['CASH'][orders.index.min()] += start_val

    for index, row in orders.iterrows():
        date, symbol, shares = index, 'JPM', row['Shares']
        holdings = trades[symbol].sum()
        if shares == 1000:
            trades[symbol][date] += (1000 - holdings)
            trades['CASH'][date] -= (prices[symbol][date] * (1000 - holdings) * (1 + impact)) + commission
        elif shares == -1000:
            trades[symbol][date] -= (holdings + 1000)
            trades['CASH'][date] += (prices[symbol][date] * (holdings + 1000) * (1 + impact)) - commission

    portfolioValue = (prices * trades.cumsum()).sum(axis = 1)
    dailyReturns = (portfolioValue[1:] / portfolioValue[:-1].values) - 1

    return portfolioValue, (portfolioValue[-1] / portfolioValue[0]) - 1, dailyReturns.mean(), dailyReturns.std()