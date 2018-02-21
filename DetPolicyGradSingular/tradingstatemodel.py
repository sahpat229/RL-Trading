import datacontainer as dc
import numpy as np

class State():
    def __init__(self, asset_features, coins, terminated):
        self.asset_features = asset_features # [1, num_features]
        self.coins = coins
        self.terminated = terminated

    @property
    def prices(self):
        return self.asset_features[:, 0]

    @property
    def features(self):
        asset_features = self.asset_features.flatten()
        #print("asset_features:", asset_features, "coins:", [self.coins])
        return np.concatenate((asset_features, [self.coins]),
                              axis=0)
    
class TradingStateModel():
    def __init__(self, datacontainer, episode_length, is_training, commission_percentage, coin_boundary):
        self.datacontainer = datacontainer
        self.episode_length = episode_length
        self.is_training = is_training
        self.commission_percentage = commission_percentage
        self.coin_boundary = coin_boundary

    def initialize(self):
        """
        Returns the initial state and reward
        """
        self.time, self.end_time = self.datacontainer.initial_time(train=self.is_training,
                                                                   episode_length=self.episode_length)
        num_coins = np.random.uniform(low=0,
                                      high=self.coin_boundary)
        self.state = State(asset_features=self.datacontainer.get_asset_features(train=self.is_training,
                                                                                time=self.time),
                           coins=num_coins,
                           terminated=False)
        return self.state, 0

    def step(self, action):
        """
        Returns the next state and reward received due to action (which is the next portfolio allocation vector)
        """
        new_num_coins = action
        self.time += 1
        if self.time == self.end_time:
            terminated = True
        else:
            terminated = False
        new_state = State(asset_features=self.datacontainer.get_asset_features(train=self.is_training,
                                                                               time=self.time),
                          coins=new_num_coins,
                          terminated=terminated)
        reward = self.reward(curr_state=self.state,
                             new_state=new_state,
                             commission_percentage=self.commission_percentage)
        self.state = new_state
        return new_state, reward

    def reward(self, curr_state, new_state, commission_percentage):
        commission_rate = commission_percentage / 100.0
        reward = new_state.coins * (new_state.prices[0]/curr_state.prices[0] - 1)
        return reward