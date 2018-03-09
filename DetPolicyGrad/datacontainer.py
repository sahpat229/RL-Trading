import matplotlib
matplotlib.use('Agg')

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import talib

from sklearn import metrics, preprocessing
from talib.abstract import *

class ContainerException(Exception):
    pass

class Container():
    """
    Container class for loading and providing data to the TradingStateModel.
    The container class assumes that data is of the form [num_assets, num_periods, num_asset_features].
    It assumes that close data is of the form [num_assets, num_periods]

    An instance should have members self.train_data, self.test_data, self.train_close, and self.test_close
    """
    def __init__(self):
        pass

    @property
    def num_assets(self):
        return self.train_data.shape[0]

    @property
    def train_length(self):
        return self.train_data.shape[1]

    @property
    def test_length(self):
        return self.test_data.shape[1]

    @property
    def num_asset_features(self):
        return self.train_data.shape[2]

    @property
    def num_flattened_features(self):
        return self.num_assets * self.num_asset_features + self.num_assets

    @staticmethod
    def split(closes, data, split_level):
        train_close = closes[:, 0:split_level]
        test_close = closes[:, split_level:]
        train_data = data[:, 0:split_level, :]
        test_data = data[:, split_level:, :]
        return train_data, test_data, train_close, test_close

    def get_data(self, train=True):
        if train:
            return self.train_data
        else:
            return self.test_data

    def get_all_prices(self, train=True):
        if train:
            return self.train_close 
        else:
            return self.test_close 

    def initial_time(self, train=True, episode_length=None, history_length=0):
        if train:
            if history_length > self.train_length:
                raise ValueError('History length should be less than or equal to length of training set')
            init_time = np.random.randint(low=history_length,
                                          high=self.train_length - episode_length)
        else:
            if history_length > self.test_length:
                raise ValueError('History length should be less than or equal to length of test set')
            init_time = history_length
        end_time = init_time + episode_length
        return init_time, end_time 

    def get_asset_features(self, train, time, history_length=None):
        data = self.get_data(train=train)
        if history_length is None:
            return data[:, time, :] # [num_assets, num_asset_features]
        else:
            return data[:, time-history_length+1:time+1, :] # [num_assets, history_length, num_asset_features]

    def get_prices(self, train, time, history_length=None):
        prices = self.get_all_prices(train=train)
        if history_length is None:
            return prices[:, time] # [num_assets]
        else:
            return prices[:, time-history_length+1:time+1] # [num_assets, history_length]

    def get_price_returns(self, train, time):
        curr_prices = self.get_prices(train=train, time=time)
        old_prices = self.get_prices(train=train, time=time-1)
        returns = (curr_prices - old_prices) / old_prices
        return returns

    def plot_prices(self, train):
        prices = self.get_all_prices(train=train)
        for ind in range(prices.shape[0]):
            plt.plot(prices[ind, :])
        plt.show()

    def plot_returns(self, train):
        returns = self.get_data(train=train)[:, :, 0]
        for ind in range(returns.shape[0]):
            plt.plot(returns[ind, :])
        plt.show()

    def featurize(self, closes, conf):
        """
        param closes is of the form [num_assets, num_periods]

        returns array of form [num_assets, num_periods, num_features]
        The first feature should always be the returns X_t^i = change(price_asset_(t-1 to t))/price_asset_(t-1)
        """
        num_assets = closes.shape[0]
        num_periods = closes.shape[1]
        features = []

        if conf['returns'] is True:
            diff = np.diff(closes)
            returns = diff / closes[:, 0:num_periods-1]
            returns = np.concatenate((np.zeros((num_assets, 1)), returns),
                                     axis=1) # [num_assets, num_periods]
            #returns = np.log(1  + returns)
            features.append(returns)

        if len(features) == 0:
            raise ValueError('No features')
        elif len(features) == 1:
            feature = features[0]
            return np.expand_dims(feature, axis=2)
        else:
            return np.stack(features, axis=2)

class TestContainer(Container):
    def __init__(self, shape='sine', num_assets=3, num_samples=200, train_split=0.7):
        super().__init__()

        if shape is 'sine':
            closes = [np.sin(2*np.pi*np.linspace(start=0, # [num_assets, num_samples]
                                                 stop=8,
                                                 num=num_samples)+(5*np.pi/8)*asset) for asset in range(num_assets)]
            closes = np.array(closes)
            closes = closes+5
            # closes = np.concatenate((np.ones((1, num_samples)), closes),
            #                         axis=0)

        data = self.featurize(closes,
                              conf={'returns': True}) # [num_assets, num_samples, num_asset_features]

        split_level = int(num_samples * train_split)
        self.train_data = data[:, 0:split_level, :]
        self.train_close = closes[:, 0:split_level]                                                                                                                        
        self.test_data = data[:, split_level:, :]
        self.test_close = closes[:, split_level:]

class EasyContainer(Container):
    def __init__(self, num_samples=200, train_split=0.7):
        super().__init__()

        closes = [10+np.arange(1, num_samples+1)*10, np.linspace(10000, 0, num_samples)+0.01, np.linspace(1000, 0, num_samples)+0.01]
        closes = np.array(closes)
        print("Closes:", closes)        

        data = self.featurize(closes,
                              conf={'returns': True})
        split_level = int(num_samples * train_split)
        self.train_data, self.test_data, self.train_close, self.test_close = \
            Container.split(closes=closes, data=data, split_level=split_level)

class BitcoinTestContainer(Container):
    def __init__(self, csv_file_name=None, train_split=0.7):
        assert csv_file_name is not None
        file = open(csv_file_name)
        reader = csv.DictReader(file)

        self.data = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        times = []

        for line in reader:
            for key in self.data:
                self.data[key].append(float(line[key]))
            times.append(int(line['time']))

        self.df = pd.DataFrame(data=self.data,
                               index=times)
        split_level = int(len(times) * train_split)
        self.train_df = self.df.iloc[:split_level, :]
        self.test_df = self.df.iloc[split_level:, :]
        self.process(train_df=self.train_df,
                     test_df=self.test_df)

    def featurize(self, df):
        close = df['close'].values
        diff = np.diff(close)
        diff = np.insert(diff, 0, 0)
        sma15 = SMA(df, timeperiod=15)
        sma60 = SMA(df, timeperiod=60)
        rsi = RSI(df, timeperiod=14)
        atr = ATR(df, timeperiod=14)

        data = np.column_stack((diff, sma15, close-sma15, sma15-sma60, rsi, atr))
        data = np.nan_to_num(data) 
        return np.array(data), np.expand_dims(close, 1) # [num_periods, num_features], [num_periods, 1]

    def process(self, train_df, test_df):
        self.pre_train_data, self.pre_train_close = self.featurize(train_df)
        self.pre_test_data, self.pre_test_close = self.featurize(test_df)

        self.feature_scaler = preprocessing.MinMaxScaler()
        self.train_data = self.feature_scaler.fit_transform(self.pre_train_data) # [num_periods, features]
        self.test_data = self.feature_scaler.transform(self.pre_test_data)

        self.pre_train_data, self.pre_test_data, self.train_data, self.test_data = \
            [np.array([arr]) for arr in [self.pre_train_data, self.pre_test_data, self.train_data, self.test_data]]
        # [1, num_periods, num_features]

        self.price_scaler = preprocessing.MinMaxScaler()
        self.train_close = self.feature_scaler.fit_transform(self.pre_train_close)
        self.test_close = self.feature_scaler.transform(self.pre_test_close)

        self.pre_train_close, self.pre_test_close, self.train_close, self.test_close = \
            [np.array([arr]) for arr in [self.pre_train_close, self.pre_test_close, self.train_close, self.test_close]]

        self.train_close = self.pre_train_close
        self.test_close = self.pre_test_close
        # [1, num_periods, 1]

class DataContainer(Container):
    def __init__(self, csv_file_name=None, hdf_file_name=None):
        if hdf_file_name is not None:
            key = 'train'
            pd_data = pd.read_hdf(hdf_file_name, key=key)
            asset_names = list(pd_data.columns.levels[0])
            train_closing_prices = [pd_data[asset_name, 'close'].values for asset_name in asset_names]

            key = 'test'
            pd_data = pd.read_hdf(hdf_file_name, key=key)
            asset_names = list(pd_data.columns.levels[0])
            test_closing_prices = [pd_data[asset_name, 'close'].values for asset_name in asset_names]

        self.train_close = np.array(train_closing_prices)
        self.test_close = np.array(test_closing_prices)

        self.sma15_train, self.sma15_test = [talib.SMA(arr, timeperiod=15) for arr in 
            [self.train_close, self.test_close]]

        self.train_data, self.test_data = [self.featurize(closes, {'returns': True}) for closes in
            [self.sma15_train, self.sma15_test]]
        self.train_data, self.test_data = [np.nan_to_num(arr) for arr in
            [self.train_data, self.test_data]]