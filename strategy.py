import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta
import math


def get_prices(ticker):
    df = yf.download(tickers=ticker, period="255d", interval="1h")
    df.reset_index(inplace=True)
    df.columns = ['time','open', 'high', 'low', 'close', 'adj close', 'volume']
    df = df.set_index(pd.DatetimeIndex(df.time.values))
    dtypes = {"open": np.float64, "high": np.float64, "low": np.float64, "close": np.float64, "volume": np.float64}
    df = df.astype(dtypes)

    return df


def sma(df, period=21, column='close'):

    return df[column].rolling(window=period).mean()


def rsi(df, period=14, column='close'):
    delta = df[column].diff(1)
    delta = delta[1:]
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    df['up'] = up
    df['down'] = down

    avg_gain = sma(df, period, column='up')
    avg_loss = abs(sma(df, period, column='down'))
    RS = avg_gain/avg_loss
    RSI = 100 - (100 / (1+RS))

    df['rsi'] = RSI

    return df


def supertrend(df, atr_period, atr_multiplier):
    high_array = df['high']
    low_array = df['low']
    close_array = df['close']
    try:
        atr = ta.ATR(high_array, low_array, close_array, atr_period)
    except:
        return False, False

    previous_final_upperband = 0
    previous_final_lowerband = 0
    final_upperband = 0
    final_lowerband = 0
    previous_close = 0
    previous_supertrend = 0
    supertrend = []
    supertrendc = 0

    for i in range(0, len(close_array)):
        if np.isnan(close_array[i]):
            pass
        else:
            highc = high_array[i]
            lowc = low_array[i]
            atrc = atr[i]
            closec = close_array[i]

            if math.isnan(atrc):
                atrc = 0

            basic_upperband = (highc + lowc) / 2 + atr_multiplier * atrc
            basic_lowerband = (highc + lowc) / 2 - atr_multiplier * atrc

            if basic_upperband < previous_final_upperband or previous_close > previous_final_upperband:
                final_upperband = basic_upperband
            else:
                final_upperband = previous_final_upperband

            if basic_lowerband > previous_final_lowerband or previous_close < previous_final_lowerband:
                final_lowerband = basic_lowerband
            else:
                final_lowerband = previous_final_lowerband

            if previous_supertrend == previous_final_upperband and closec <= final_upperband:
                supertrendc = final_upperband
            else:
                if previous_supertrend == previous_final_upperband and closec >= final_upperband:
                    supertrendc = final_lowerband
                else:
                    if previous_supertrend == previous_final_lowerband and closec >= final_lowerband:
                        supertrendc = final_lowerband
                    elif previous_supertrend == previous_final_lowerband and closec <= final_lowerband:
                        supertrendc = final_upperband

            supertrend.append(supertrendc)

            previous_close = closec

            previous_final_upperband = final_upperband

            previous_final_lowerband = final_lowerband

            previous_supertrend = supertrendc

    return supertrend


def stragey_supertrend(df, interval, atr_period, atr_multiplier):
    logic = {'open': 'first',
             'high': 'max',
             'low': 'min',
             'close': 'last',
             'volume': 'sum'}

    df = df.resample(f'{interval}min').apply(logic)
    df.dropna(subset=['close'], inplace=True)
    df = rsi(df, 8, 'close')
    df['supertrend'] = supertrend(df, atr_period, atr_multiplier)
    df['position'] = np.where((df.close > df.supertrend) & (df.rsi <= 94), 1, 0)
    df['return'] = np.log(df['close'].pct_change() + 1)
    df['strategy_return'] = df.position.shift(1) * df['return']

    return df


if __name__ == '__main__':

    ticker = "AAPL"
    df = get_prices(ticker)
    return_value = df['close'].iloc[-1] / df['close'].iloc[0] - 1

    print(f"{ticker}'s return profit in 255 business days: {return_value}")
    intervals = []
    atr_periods = []
    multipliers = []
    strategy_returns = []
    total_trades = []
    print("Running optimization...")
    for interval in [60, 120, 240, 480, 1440]:
        for atr_period in range(6, 18):
            for multiplier in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.2, 3.4]:
                super_df = stragey_supertrend(df, interval, atr_period, multiplier)
                trades = super_df.position.diff().value_counts().iloc[1:].sum()
                costs = trades * 0.0006
                strategy_return = np.exp(super_df['strategy_return'].sum()) - 1 - costs
                intervals.append(interval)
                atr_periods.append(atr_period)
                multipliers.append(multiplier)
                strategy_returns.append(strategy_return)
                total_trades.append(trades)

    strategy_df = pd.DataFrame({"interval": intervals, "atr_period": atr_periods,
                                "multiplier": multipliers, "strategy_return": strategy_returns, "trades": total_trades})
    strategy_df.sort_values(by='strategy_return', ascending=False, inplace=True)

    print(strategy_df.head())



