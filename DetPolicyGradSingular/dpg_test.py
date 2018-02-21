import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf

from datacontainer import TestContainer
from dpg import DDPG
from networks import ActorNetwork, CriticNetwork
from replaybuffer import ReplayBuffer
from tradingstatemodel import TradingStateModel 

NUM_EPISODES = 1000
EPISODE_LENGTH = 50
COMMISSION_PERCENTAGE = 0.0
BATCH_SIZE = 32
BATCH_NORM = True
BUFFER_SIZE=1000000
COIN_BOUNDARY = 5

tc = TestContainer(num_assets=1, num_samples=5000)
#tc.plot_prices(train=True)
tsm = TradingStateModel(datacontainer=tc,
                        episode_length=EPISODE_LENGTH,
                        is_training=True,
                        commission_percentage=COMMISSION_PERCENTAGE,
                        coin_boundary=COIN_BOUNDARY)
print("FLATTENED:", tc.num_flattened_features)

sess = tf.Session()
actor_target = ActorNetwork(sess=sess,
                            batch_size=BATCH_SIZE,
                            batch_norm=BATCH_NORM,
                            dropout=0.5,
                            history_length=50,
                            datacontainer=tc,
                            epochs=50,
                            is_target=True,
                            coin_boundary=COIN_BOUNDARY)
actor_trainer = ActorNetwork(sess=sess,
                             batch_size=BATCH_SIZE,
                             batch_norm=BATCH_NORM,
                             dropout=0.5,
                             history_length=50,
                             datacontainer=tc,
                             epochs=50,
                             is_target=False,
                             coin_boundary=COIN_BOUNDARY)
critic_target = CriticNetwork(sess=sess,
                              batch_size=BATCH_SIZE,
                              batch_norm=BATCH_NORM,
                              dropout=0.5,
                              history_length=50,
                              datacontainer=tc,
                              epochs=50,
                              is_target=True)
critic_trainer = CriticNetwork(sess=sess,
                               batch_size=BATCH_SIZE,
                               batch_norm=BATCH_NORM,
                               dropout=0.5,
                               history_length=50,
                               datacontainer=tc,
                               epochs=50,
                               is_target=False)
rpb = ReplayBuffer(buffer_size=BUFFER_SIZE)
dpg = DDPG(sess=sess,
           batch_size=BATCH_SIZE,
           num_episodes=NUM_EPISODES,
           actor_target=actor_target,
           actor_trainer=actor_trainer,
           critic_target=critic_target,
           critic_trainer=critic_trainer,
           trading_state_model=tsm,
           replay_buffer=rpb,
           datacontainer=tc,
           gamma=0.95,
           tau=0.01)
dpg.train()