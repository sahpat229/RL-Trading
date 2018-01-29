import numpy as np
import pandas as pd

class DataContainer():
    def __init__(self, csv_file_name=None, hdf_file_name=None, key=None):
        if hdf_file_name is not None:
            self.pd_data = pd.read_hdf(hdf_file_name, key=key)
        else:
            # make a Pandas DataFrame using the csv data
            pass

    def yield_data(self, history_length=10, batch_size=10):
        asset_names = list(self.pd_data.columns.levels[0])
        closing_prices = [self.pd_data[asset_name, 'close'].values for asset_name in asset_names]
        high_prices = [self.pd_data[asset_name, 'high'].values for asset_name in asset_names]
        low_prices = [self.pd_data[asset_name, 'high'].values for asset_name in asset_names]

        all_closing = np.stack(closing_prices) # [num_assets, num_periods]
        all_high = np.stack(high_prices) # [num_assets, num_periods]
        all_low = np.stack(low_prices) # [num_assets, num_periods]
        self.all_prices = np.stack([all_closing, all_high, all_low],
                                   axis=2) # [num_assets, num_periods, 3]

        time = 0 # end time = history_length - 1
        while time <= self.all_prices.shape[1] - history_length - batch_size + 1:
            yield self.make_batch(time=time,
                                  history_length=history_length,
                                  batch_size=batch_size)
            time += 1

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

class PortfolioVectorMemory():
    def __init__(self, num_periods, num_assets):
        self.PVM = np.ones((num_periods, num_assets))
        self.PVM = self.PVM / (num_assets + 1) # since we need to include the cash

    def read_batch(self, time, batch_size=10):
        return self.PVM[time:time+batch_size, :] # [batch_size, num_assets]

    def input_batch(self, time, weights):
        # param: weights [batch_size, num_assets+1]
        weights = weights[:, 1:]
        self.PVM[time:time+batch_size, :] = weights