import matplotlib
#matplotlib.use('Agg')
import gym
import numpy as np
import tensorflow as tf

from dpg_gym import DDPG, OrnsteinUhlenbeckActionNoise
from networks import ActorNetwork, CriticNetwork
from replaybuffer import ReplayBuffer 
from rank_based import Experience

BUFFER_SIZE = 100000
BATCH_SIZE = 32
BATCH_NORM = True
LEARNING_RATE = 1e-3
DROPOUT = 0.5
NUM_EPISODES = 1000000
EPISODE_LENGTH = 250
GAMMA = 0.99
TAU = 0.001

env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
boundary = env.action_space.high[0]

actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
rpb = ReplayBuffer(buffer_size=BUFFER_SIZE)
rpb = Experience()

sess = tf.Session()
actor_trainer = ActorNetwork(sess=sess,
                             batch_size=BATCH_SIZE,
                             batch_norm=BATCH_NORM,
                             learning_rate=LEARNING_RATE,
                             dropout=DROPOUT,
                             is_target=False,
                             state_boundary=boundary,
                             state_dimension=state_dim,
                             action_dimension=action_dim)
actor_target = ActorNetwork(sess=sess,
                            batch_size=BATCH_SIZE,
                            batch_norm=BATCH_NORM,
                            learning_rate=LEARNING_RATE,
                            dropout=DROPOUT,
                            is_target=True,
                            state_boundary=boundary,
                            state_dimension=state_dim,
                            action_dimension=action_dim)
critic_trainer = CriticNetwork(sess=sess,
                               batch_size=BATCH_SIZE,
                               batch_norm=BATCH_NORM,
                               learning_rate=LEARNING_RATE,
                               dropout=DROPOUT,
                               is_target=False,
                               state_dimension=state_dim,
                               action_dimension=action_dim)
critic_target = CriticNetwork(sess=sess,
                              batch_size=BATCH_SIZE,
                              batch_norm=BATCH_NORM,
                              learning_rate=LEARNING_RATE,
                              dropout=DROPOUT,
                              is_target=True,
                              state_dimension=state_dim,
                              action_dimension=action_dim)
ddpg = DDPG(sess=sess, 
            batch_size=BATCH_SIZE, 
            num_episodes=NUM_EPISODES, 
            episode_length=EPISODE_LENGTH,
            actor_target=actor_target,
            actor_trainer=actor_trainer,
            critic_target=critic_target,
            critic_trainer=critic_trainer,
            env=env, 
            replay_buffer=rpb, 
            gamma=GAMMA,
            tau=TAU, 
            actor_noise=actor_noise)
ddpg.train()
