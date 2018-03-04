import datacontainer as dc
import numpy as np

class State():
    def __init__(self, asset_features, portfolio_allocation, price, terminated):
        """
        Param asset_features is all the technical indicators we want to provide to our model
            Should be of the form [num_assets, num_history_length, num_features]

        Param portfolio_allocation is a_t^(tilda)
        Param terminated is whether the episode is finished or not
        """
        self.asset_features = asset_features
        self.portfolio_allocation = portfolio_allocation
        self.terminated = terminated
        self.price = price

    @property
    def returns(self):
        """
        Gets all the returns [X_t-history_length+1 ... X_t].  Assumes that returns is the first feature
            in asset_features
        """
        return self.asset_features[:, :, 0] # [num_assets, num_history_length]
    
    @property
    def features(self):
        asset_features = self.asset_features.flatten()
        return np.concatenate((asset_features, self.portfolio_allocation),
                              axis=0)    
    
class TradingStateModel():
    def __init__(self, datacontainer, episode_length, history_length, is_training, commission_percentage):
        self.datacontainer = datacontainer
        self.episode_length = episode_length
        self.history_length = history_length
        self.is_training = is_training
        self.commission_percentage = commission_percentage

    def reset(self):
        """
        Returns the initial state and reward
        """
        self.time, self.end_time = self.datacontainer.initial_time(train=self.is_training,
                                                                   episode_length=self.episode_length,
                                                                   history_length=self.history_length)
        initial_portfolio = np.zeros(self.datacontainer.num_assets)
        initial_portfolio[0] = 1
        self.state = State(asset_features=self.datacontainer.get_asset_features(train=self.is_training,
                                                                                time=self.time,
                                                                                history_length=self.history_length),
                           portfolio_allocation=initial_portfolio,
                           price=self.datacontainer.get_prices(train=self.is_training,
                                                               time=self.time),
                           terminated=False)
        return self.state

    def step(self, action):
        """
        Returns the next state and reward received due to action (which is the next portfolio allocation vector)
        """
        action = action / np.sum(action)

        self.time += 1
        if self.time == self.end_time:
            terminated = True
        else:
            terminated = False

        reward, after_price_changes = self.reward(old_portfolio=self.state.portfolio_allocation,
                                                  new_portfolio=action,
                                                  price_returns=self.datacontainer.get_asset_features(train=self.is_training,
                                                                                                      time=self.time)[:, 0],
                                                  commission_percentage=self.commission_percentage)
        new_state = State(asset_features=self.datacontainer.get_asset_features(train=self.is_training,
                                                                               time=self.time,
                                                                               history_length=self.history_length),
                          portfolio_allocation=after_price_changes,
                          price=self.datacontainer.get_prices(train=self.is_training,
                                                              time=self.time),
                          terminated=terminated)
        self.state = new_state
        return new_state, reward, new_state.terminated, None # To match OpenAI Gym protocol

    def reward(self, old_portfolio, new_portfolio, price_returns, commission_percentage):
        """
        param old_portfolio is {a_t^tilda}_i [num_assets]
        param new_portfolio is {a_t}_i [num_assets]
        param price_returns is {X_(t+1)}_i [num_assets]
        param commission_percentage is delta_i
        """
        commission_rate = commission_percentage / 100.0

        pnl = np.dot(new_portfolio, price_returns)
        tc = commission_rate * np.sum(np.abs(new_portfolio - old_portfolio))
        print("Comission:", tc)
        reward = np.log(1 + pnl - tc)

        after_price_changes = new_portfolio * (1 + price_returns) / (1 + pnl - tc) # {a_(t+1)^tilda}_i [num_assets]
        return reward, after_price_changes

    # def reward(self, curr_state, new_state, commission_percentage):
    #     diffs = np.abs(new_state.portfolio_allocation - curr_state.portfolio_allocation)
    #     commission_rate = commission_percentage / 100.0
    #     after_commission = new_state.portfolio_allocation - (commission_rate * diffs)
    #     price_ratio = new_state.prices / curr_state.prices
    #     after_price_changes = price_ratio * after_commission

    #     new_portfolio = after_price_changes / np.sum(after_price_changes)
    #     reward = np.sum(after_price_changes) - 1
    #     return reward, new_portfolio