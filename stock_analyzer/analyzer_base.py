import datetime
import logging
import os

import pandas as pd
import statsmodels.api as stats
from pandas.tseries.offsets import BDay

from stock_analyzer.data_fetcher import get_ranged_data, get_spx_prices

logging.basicConfig(format='%(level_name)s: %(message)s', level=logging.DEBUG)


class AnalyzerBase(object):
    DATA_FOLDER = 'asset_data'

    SP500_FILE = 'sp500'

    def __init__(self, ticker, hist_start_date=None):
        self.ticker = ticker
        self.stock_data = pd.DataFrame()
        self.sp500_data = pd.DataFrame()
        self.hist_start_date = hist_start_date or datetime.datetime.today() - BDay(252)

    @property
    def mean(self):
        raise NotImplementedError("Need to be implemented by sub-classes")

    @property
    def asset_returns(self):
        raise NotImplementedError("Need to be implemented by sub-classes")

    @property
    def asset_returns(self):
        raise NotImplementedError("Need to be implemented by sub-classes")

    @property
    def index_returns(self):
        raise NotImplementedError("Need to be implemented by sub-classes")

    def plot_returns(self):
        raise NotImplementedError("Need to be implemented by sub-classes")

    def plot_returns_against_snp500(self):
        raise NotImplementedError("Need to be implemented by sub-classes")

    @property
    def beta(self):
        raise NotImplementedError("Need to be implemented by sub-classes")

    @property
    def ols_model(self):
        raise NotImplementedError("Need to be implemented by sub-classes")

    def setup_underlying_data(self, refresh=False):
        if not os.path.exists(self.DATA_FOLDER):
            os.makedirs(self.DATA_FOLDER)

        self.get_sp500_data(refresh=refresh)
        self.get_stock_data(refresh=refresh)

    def get_stock_data(self, refresh=False):
        df = pd.DataFrame()
        if refresh or not os.path.exists('{}/{}.csv'.format(self.DATA_FOLDER, self.ticker)):
            try:
                df = get_ranged_data(self.ticker, self.hist_start_date, useQuandl=False)
                self.save_data(df, self.ticker)
            except Exception as exception:
                logging.error("#### Error for Ticker %s" % self.ticker)
                logging.error(str(exception))
        else:
            logging.debug('Already have {}'.format(self.ticker))
            df = pd.read_csv('{}/{}.csv'.format(self.DATA_FOLDER, self.ticker), parse_dates=True, index_col=1)
        if df.empty:
            logging.error("Unable to get Stock-Data from the Web. Please check connection")
            raise IOError("Unable to get Stock-Data from the Web")

        df.reset_index(inplace=True)
        # df = df.drop("Symbol", axis=1)
        df.set_index("Date", inplace=True)
        self.stock_data = df

    def save_data(self, data_frame, file_name):
        data_frame.to_csv('{}/{}.csv'.format(self.DATA_FOLDER, file_name))
        logging.info("Save completed for {}".format(file_name))

    def get_sp500_data(self, refresh=False):
        df = pd.DataFrame()
        if refresh or not os.path.exists('{}/{}.csv'.format(self.DATA_FOLDER, self.SP500_FILE)):
            try:
                df = get_spx_prices(self.hist_start_date)
                self.save_data(df, self.SP500_FILE)
            except Exception as exception:
                logging.error("#### Error in getting SNP-500 data")
                logging.error(str(exception))
        else:
            logging.debug('Already have {}'.format(self.SP500_FILE))
            df = pd.read_csv('{}/{}.csv'.format(self.DATA_FOLDER, self.SP500_FILE), parse_dates=True, index_col=1)
        if df.empty:
            logging.error("Unable to get Stock-Data from the Web. Please check connection")
            raise IOError("Unable to get SNP 500 from the Web")

        df.reset_index(inplace=True)
        df.set_index("Date", inplace=True)
        self.sp500_data = df

    @staticmethod
    def ordinary_least_square_model(asset_returns, index_returns):
        def lin_reg(x, y):
            x = stats.add_constant(x)
            model = stats.OLS(y, x).fit()
            x = x[:, 1]
            return model

        return lin_reg(index_returns.values, asset_returns.values)
