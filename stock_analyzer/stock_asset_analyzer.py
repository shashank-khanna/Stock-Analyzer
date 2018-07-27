import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from mpl_finance import candlestick_ohlc

from stock_analyzer.analyzer_base import AnalyzerBase

logging.basicConfig(format='%(level_name)s: %(message)s', level=logging.DEBUG)

style.use('ggplot')


class StockAssetAnalyzer(AnalyzerBase):

    def __init__(self, ticker, hist_start_date=None, refresh=False):
        super(StockAssetAnalyzer, self).__init__(ticker, hist_start_date)
        # Get underlying data and setup required parameters
        self.setup_underlying_data(refresh=refresh)

    @property
    def mean(self):
        return self.stock_data['Close'].mean()

    @property
    def std(self):
        return self.stock_data['Close'].std()

    @property
    def asset_returns(self):
        if self.stock_data.empty:
            raise ValueError("Historical stock prices unavailable")
        print(self.stock_data.head())
        self.stock_data['returns'] = self.stock_data['Close'].pct_change()
        return self.stock_data.returns[1:]

    @property
    def index_returns(self):
        if self.sp500_data.empty:
            raise ValueError("Historical stock prices unavailable")
        self.sp500_data['returns'] = self.sp500_data['Close'].pct_change()
        return self.sp500_data.returns[1:]

    def plot_returns(self):
        plt.figure(figsize=(10, 5))
        self.asset_returns.plot()
        plt.ylabel("Daily Returns of %s " % self.ticker)
        plt.show()

    def plot_returns_against_snp500(self):
        plt.figure(figsize=(10, 5))
        self.asset_returns.plot()
        self.index_returns.plot()
        plt.ylabel("Daily Returns of %s against SNP500" % self.ticker)
        plt.show()

    def plot_candlestick(self):
        df_ohlc = self.stock_data['Close'].resample('4D').ohlc()
        df_volume = self.stock_data['Volume'].resample('4D').sum()
        df_ohlc = df_ohlc.reset_index()
        df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

        fig = plt.figure(figsize=(20, 10))
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(self.ticker)
        plt.legend()
        plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0.8)

        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
        ax1.xaxis_date()

        candlestick_ohlc(ax1, df_ohlc.values, width=3, colorup='#77d879', colordown='#db3f3f')
        ax2.bar(df_volume.index.map(mdates.date2num), df_volume.values)
        # ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
        plt.show()

    def plot_moving_averages(self, window1=14, window2=42):
        self.stock_data['%dDay' % window1] = self.stock_data['Mean'].rolling(window=window1).mean()
        self.stock_data['%dDay' % window2] = self.stock_data['Mean'].rolling(window=window2).mean()
        self.stock_data[['Mean', '%dDay' % window1, '%dDay' % window2]].plot()
        plt.show()

    def plot_ols(self):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.index_returns.values, self.asset_returns.values, 'r.')
        ax = plt.axis()
        x = np.linspace(ax[0], ax[1] + 0.01)
        plt.plot(x, self.alpha + self.beta * x, 'b', lw=2)

        plt.grid(True)
        plt.axis('tight')
        plt.xlabel('SNP 500 Returns')
        plt.ylabel('{} returns'.format(self.ticker))
        plt.show()

    @property
    def alpha(self):
        return self.ols_model.params[0]

    @property
    def beta(self):
        return self.ols_model.params[1]

    @property
    def ols_model(self):
        return AnalyzerBase.ordinary_least_square_model(self.asset_returns, self.index_returns)


if __name__ == '__main__':
    analyzer = StockAssetAnalyzer('TSLA')
    print(analyzer.asset_returns.head())
    print(analyzer.index_returns.head())
    analyzer.plot_returns()
    analyzer.plot_returns_against_snp500()
    analyzer.plot_candlestick()
    analyzer.plot_moving_averages()
    analyzer.plot_ols()
    print(analyzer.ols_model.summary())
    print("Alpha ", analyzer.alpha)
    print("Beta ", analyzer.beta)
    print("Mean ", analyzer.mean)
