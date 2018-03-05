import matplotlib
#matplotlib.use('Agg')
import gym
import numpy as np
import tensorflow as tf

from datacontainer import BitcoinTestContainer, TestContainer, DataContainer
from dpg_gym_tf import DDPG, OrnsteinUhlenbeckActionNoise
from networks_tf import ActorNetwork, CriticNetwork
from replaybuffer import ReplayBuffer
from tradingstatemodel import TradingStateModel

from baselines.deepq.replay_buffer import PrioritizedReplayBuffer

BUFFER_SIZE = 100000
BATCH_SIZE = 32
BATCH_NORM = True
LEARNING_RATE = 1e-3
DROPOUT = 0.5
NUM_EPISODES = 10000
EPISODE_LENGTH = 250
HISTORY_LENGTH = 50
GAMMA = 0.99
TAU = 0.001
COMMISSION_PERCENTAGE = 0.0

# env = gym.make('Pendulum-v0')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# boundary = env.action_space.high[0]

# dc = TestContainer(num_assets=3,
#                    num_samples=2000)
dc = DataContainer(hdf_file_name='../data/hdfs/poloniex_30m.hf')
#dc = BitcoinTestContainer(csv_file_name='../data/csvs/output.csv')
env = TradingStateModel(datacontainer=dc,
                        episode_length=EPISODE_LENGTH,
                        is_training=True,
                        history_length=HISTORY_LENGTH,
                        commission_percentage=COMMISSION_PERCENTAGE)
asset_features_shape = [dc.num_assets, HISTORY_LENGTH, dc.num_asset_features]
action_dim = dc.num_assets

actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
# rpb = ReplayBuffer(buffer_size=BUFFER_SIZE)
# conf = {
#   'size': BUFFER_SIZE,
#   'batch_size': BATCH_SIZE,
#   'learn_start': 1000,
#   'steps': NUM_EPISODES * EPISODE_LENGTH
# }
# rpb = Experience(conf)
rpb = PrioritizedReplayBuffer(size=BUFFER_SIZE,
                              alpha=0.6)

sess = tf.Session()
actor = ActorNetwork(sess=sess,
                     asset_features_shape=asset_features_shape, 
                     action_dim=action_dim, 
                     action_bound=1, 
                     learning_rate=LEARNING_RATE, 
                     tau=TAU, 
                     batch_size=BATCH_SIZE)
critic = CriticNetwork(sess=sess,
                       asset_features_shape=asset_features_shape, 
                       action_dim=action_dim, 
                       learning_rate=LEARNING_RATE, 
                       tau=TAU,
                       gamma=GAMMA, 
                       num_actor_vars=actor.get_num_trainable_vars())
ddpg = DDPG(sess=sess, 
            batch_size=BATCH_SIZE, 
            num_episodes=NUM_EPISODES, 
            episode_length=EPISODE_LENGTH,
            actor=actor,
            critic=critic,
            env=env, 
            replay_buffer=rpb, 
            gamma=GAMMA,
            tau=TAU, 
            actor_noise=actor_noise)
ddpg.train()
