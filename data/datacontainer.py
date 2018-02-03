import csv
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from talib.abstract import *

class DataContainer():
    def __init__(self, csv_file_name=None, hdf_file_name=None, key=None):
        if hdf_file_name is not None:
            self.pd_data = pd.read_hdf(hdf_file_name, key=key)
            asset_names = list(self.pd_data.columns.levels[0])
            closing_prices = [self.pd_data[asset_name, 'close'].values for asset_name in asset_names]
            high_prices = [self.pd_data[asset_name, 'high'].values for asset_name in asset_names]
            low_prices = [self.pd_data[asset_name, 'high'].values for asset_name in asset_names]

            all_closing = np.stack(closing_prices) # [num_assets, num_periods]
            all_high = np.stack(high_prices) # [num_assets, num_periods]
            all_low = np.stack(low_prices) # [num_assets, num_periods]

        else:
            # make a Pandas DataFrame using the csv data
            file = open(csv_file_name, 'r')
            reader = csv.DictReader(file)
            closing_prices = []
            high_prices = []
            low_prices = []
            for line in reader:
                closing_prices.append(line['close'])
                high_prices.append(line['high'])
                low_prices.append(line['low'])
            closing_prices, high_prices, low_prices = [list(map(float, arr)) for arr in [closing_prices,
                high_prices, low_prices]]
            all_closing = np.array([closing_prices]) # [1, num_periods]
            all_high = np.array([high_prices])
            all_low = np.array([low_prices])

        self.all_prices = np.stack([all_closing, all_high, all_low],
                                   axis=2) # [num_assets, num_periods, 3]
        print("Shape", self.all_prices.shape)

    @property
    def num_periods(self):
        return self.all_prices.shape[1]

    @property
    def num_assets(self):
        return self.all_prices.shape[0]

    def yield_data(self, history_length=10, batch_size=10):
        time = 0 # end time = history_length - 1
        while time <= self.all_prices.shape[1] - history_length - batch_size:
            yield self.make_batch(time=time,
                                  history_length=history_length,
                                  batch_size=batch_size), time
            time += 1

        print("Here")

    def make_batch(self, time, history_length=10, batch_size=10):
        batch_current_prices = [] # size: [batch_size, num_assets, history_length, 3]
        batch_future_prices = [] # size: [batch_size, num_assets]
        for time in range(time, time+batch_size):
            d = self.create_window(time, history_length=history_length)
            batch_current_prices.append(d['current_prices'])
            batch_future_prices.append(d['future_prices'])
        return {'batch_current_prices': np.array(batch_current_prices), 
                'batch_future_prices': np.array(batch_future_prices)}

    def create_window(self, time, history_length=10):
        if time <= self.all_prices.shape[1] - history_length - 1:
            window = self.all_prices[:, time:time+history_length, :] # [num_assets, history_length, 3]
            future = self.all_prices[:, time+history_length, 0] # v_(t+1), [num_assets]
            y = future / window[:, -1, 0] # y_(t+1)
            window = window / window[-1, :] # normalized, X_t
            return {'current_prices': window, 'future_prices': y}
        else:
            return None

class BitcoinTestContainer():
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

        data = np.column_stack((close, diff, sma15, close-sma15, sma15-sma60, rsi, atr))
        data = np.nan_to_num(data)
        return data

    def process(self, train_df, test_df):
        self.train_data = self.featurize(train_df)
        self.test_data = self.featurize(test_df)

        scaler = preprocessing.StandardScaler()
        scaler.fit(self.train_data)
        self.train_data = scaler.transform(self.train_data) # [num_periods, features]
        self.test_data = scaler.transform(self.test_data) # [num_periods, features]

    @property
    def train_length(self):
        return self.train_data.shape[0]

    @property
    def test_length(self):
        return self.test_data.shape[0]

    @property
    def num_features(self):
        return self.train_data.shape[1]

class PortfolioVectorMemory():
    def __init__(self, num_periods, num_assets):
        self.PVM = np.ones((num_periods, num_assets))
        self.PVM = self.PVM / (num_assets + 1) # since we need to include the cash

    def read_batch(self, time, batch_size=10):
        return self.PVM[time:time+batch_size, :] # [batch_size, num_assets]

    def input_batch(self, time, weights):
        # param: weights [batch_size, num_assets+1]
        batch_size = weights.shape[0]
        weights = weights[:, 1:]
        self.PVM[time:time+batch_size, :] = weights