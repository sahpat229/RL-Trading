from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from tqdm import tqdm # Progress bars

import numpy as np
import random

class State():
    def __init__(self, price, num_coins, max_coins, terminated):
        self.price = price
        self.num_coins = self.one_hot_encode(num_coins=num_coins,
                                             max_coins=max_coins)
        self.terminated = terminated

    def one_hot_encode(self, num_coins, max_coins):
        coins = np.zeros(max_coins+1)
        coins[num_coins] = 1
        return coins

    @property
    def feature(self):
        return np.concatenate((self.price, self.num_coins))
    
class TradingStateModel():
    def __init__(self, bitcoin_container, model, episode_length=200, gamma=0.95, starting_coins=0, 
                 max_coins=4, epochs=100, replay_buffer=None, batch_size=10):
        self.container = bitcoin_container
        self.model = model # Neural network model
        self.episode_length = episode_length
        self.gamma = gamma
        self.num_coins = starting_coins
        self.max_coins = 4
        self.epochs = epochs
        self.replay_buffer=replay_buffer
        self.batch_size = batch_size

    def initialize(self, train=True):
        # returns the initial state
        self.is_training = train
        if train is True:
            self.time_step = random.randint(0, self.container.train_length - self.episode_length)
            self.end_time = self.time_step + self.episode_length
            self.data = self.container.train_data
            return State(price=self.container.train_data[self.time_step, :], 
                         num_coins=self.num_coins, 
                         max_coins=self.max_coins,
                         terminated=False), 0
        else:
            self.time_step = 0
            self.end_time = self.container.test_length - 1
            self.data = self.container.test_data
            return State(price=self.container.test_data[self.test_data, :], 
                         num_coins=self.num_coins, 
                         max_coins=self.max_coins,
                         terminated=False), 0

    def step(self, action):
        # action in range [0, 4]
        self.num_coins = action
        self.time_step += 1
        if self.time_step == self.end_time:
            terminated = True
        else:
            terminated = False
        return State(price=self.data[self.time_step, :], 
                     num_coins=self.num_coins, 
                     max_coins=self.max_coins,
                     terminated=terminated), self.reward()

    def reward(self):
        assert self.time_step < self.data.shape[0]
        rew = self.num_coins * (self.data[self.time_step, 0]/self.data[self.time_step-1, 0] - 1) # 0 index is for close data
        return rew

    def train(self):
        epsilon = 1.00
        for _ in range(self.epochs):
            print("Epoch:", _)
            state, reward = self.initialize(train=True)
            rewards = []
            for i in tqdm(range(self.episode_length)):
                if random.random() < epsilon:
                    action = np.random.randint(0, 4)
                else:
                    action = np.argmax(self.model.predict(state))
                trans_state, reward = self.step(action)
                rewards.append(reward)
                self.replay_buffer.store(old_state=state,
                                         action=action,
                                         reward=reward,
                                         new_state=trans_state)
                if self.replay_buffer.full:
                    transitions = self.replay_buffer.sample(self.batch_size)
                    batch_x = []
                    batch_y = []
                    for transition in transitions:
                        old_state, action, reward, new_state = transition
                        y = self.model.predict(old_state)
                        if new_state.terminated:
                            y[action] = reward
                        else:
                            y[action] = reward + self.gamma*np.amax(self.model.predict(new_state))
                        batch_x.append(old_state)
                        batch_y.append(y)
                    self.model.grad_step(batch_x, batch_y)
                state = trans_state
            print("Avg reward:", np.mean(rewards))

        if epsilon > 0.1:
            epsilon -= 1.0 / self.epochs

class QApproximator():
    def __init__(self, num_features, num_actions):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_dim=num_features))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_actions, activation='linear'))
        adam = Adam()
        self.model.compile(optimizer=adam,
                           loss='mean_squared_error')

    def predict(self, state):
        return self.model.predict(np.array([state.feature]), batch_size=1)[0]

    def grad_step(self, batch_x, batch_y):
        batch_values = np.array([state.feature for state in batch_x])
        batch_targets = np.array(batch_y)
        self.model.fit(x=batch_values,
                       y=batch_targets,
                       batch_size=len(batch_values),
                       verbose=0)

class ReplayBuffer():
    def __init__(self, buffer_size=200):
        self.buffer = []
        self.buffer_size = 200
        self.replace_index = 0

    def store(self, old_state, action, reward, new_state):
        tup = [old_state, action, reward, new_state]
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(tup)
        else:
            self.buffer[self.replace_index % self.buffer_size] = tup
            self.replace_index += 1

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    @property
    def full(self):
        return len(self.buffer) == self.buffer_size
    