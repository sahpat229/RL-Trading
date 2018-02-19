import datacontainer as dc
import numpy as np

class State():
    def __init__(self, asset_features, portfolio_allocation, terminated):
        self.asset_features = asset_features # [num_assets, num_features]
        self.portfolio_allocation = portfolio_allocation
        self.terminated = terminated

    @property
    def prices(self):
        return self.asset_features[:, 0]
    
class TradingStateModel():
    def __init__(self, datacontainer, episode_length, is_training, commission_percentage):
        self.datacontainer = datacontainer
        self.episode_length = episode_length
        self.is_training = is_training
        self.commission_percentage = commission_percentage

    def initialize(self):
        """
        Returns the initial state and reward
        """
        self.time, self.end_time = self.datacontainer.initial_time(train=self.is_training,
                                                                   episode_length=self.episode_length)
        initial_portfolio = np.zeros(self.datacontainer.num_assets)
        initial_portfolio[0] = 1

        self.state = State(asset_features=self.datacontainer.get_features(train=self.is_training,
                                                                          time=self.time),
                           portfolio_allocation=initial_portfolio,
                           terminated=False)
        return self.state, 0

    def step(self, action):
        """
        Returns the next state and reward received due to action (which is the next portfolio allocation vector)
        """
        new_portfolio = action
        self.time += 1
        if self.time == self.end_time:
            terminated = True
        else:
            terminated = False

        new_state = State(asset_features=self.datacontainer.get_features(train=self.is_training,
                                                                         time=self.time),
                          portfolio_allocation=new_portfolio,
                          terminated=terminated)
        reward = self.reward(curr_state=self.state,
                             new_state=new_state,
                             commission_percentage=self.commission_percentage)
        self.state = new_state
        return new_state, reward

    def reward(self, curr_state, new_state, commission_percentage):
        total_diff = np.sum(np.abs(new_state.portfolio_allocation - curr_state.portfolio_allocation))
        price_ratio = new_state.prices / curr_state.prices
        commission_rate = commission_percentage / 100
        reward = (1 - commission_rate) * np.dot(new_state.portfolio_allocation, price_ratio)
        return reward - 1