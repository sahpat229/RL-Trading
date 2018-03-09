import matplotlib
matplotlib.use('Agg')

import gym
import matplotlib.pyplot as plt
import numpy as np 
import os
import random
import tensorflow as tf 

from networks_tf import ActorNetwork, CriticNetwork
from tradingstatemodel import  TradingStateModel
from tqdm import tqdm
from utils import convert_features, softmax

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class DDPG():
    def __init__(self, sess, batch_size, num_episodes, episode_length, actor, critic,
                 env, replay_buffer, gamma, tau, actor_noise, tensorboard_directory,
                 infer_directory):
        self.sess = sess
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.actor = actor
        self.critic = critic
        self.env = env
        self.rpb = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.actor_noise = actor_noise
        self.infer_directory = infer_directory

        self.sess.run(tf.global_variables_initializer())
        self.actor.assign_target_network()
        self.critic.assign_target_network()
        if not os.path.exists(tensorboard_directory):
            os.makedirs(tensorboard_directory)
        else:
            for file in os.listdir(tensorboard_directory):
                file_path = os.path.join(tensorboard_directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
        if not os.path.exists(infer_directory):
            os.makedirs(infer_directory)
        else:
            for file in os.listdir(infer_directory):
                file_path = os.path.join(infer_directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
        self.writer = tf.summary.FileWriter(tensorboard_directory, sess.graph)
        self.build_summaries()

    def build_summaries(self):
        self.episode_reward = tf.placeholder(dtype=tf.float32,
                                             shape=None)
        self.qfunc_loss = tf.placeholder(dtype=tf.float32,
                                         shape=None)
        self.actions = tf.placeholder(dtype=tf.float32,
                                      shape=[self.actor.a_dim])
        self.prices = tf.placeholder(dtype=tf.float32,
                                     shape=[self.actor.a_dim])
        self.individual_reward = tf.placeholder(dtype=tf.float32,
                                                shape=None)
        self.individual_pnl = tf.placeholder(dtype=tf.float32,
                                             shape=None)
        self.individual_tc = tf.placeholder(dtype=tf.float32,
                                            shape=None)
        self.individual_estimated_q = tf.placeholder(dtype=tf.float32,
                                                     shape=None)
        ep_reward = tf.summary.scalar("Episode Reward", self.episode_reward)
        qfunc_loss = tf.summary.scalar("Qfunc Loss", self.qfunc_loss)
        actions = [tf.summary.scalar("Action-"+str(index), self.actions[index]) for
            index in range(self.actor.a_dim)]
        prices = [tf.summary.scalar("Price-"+str(index), self.prices[index]) for
            index in range(self.actor.a_dim)]
        individual_reward = tf.summary.scalar("Individual Reward", self.individual_reward)
        individual_pnl = tf.summary.scalar("Individiual Pnl", self.individual_pnl)
        individual_tc = tf.summary.scalar("Individual Tc", self.individual_tc)

        self.episode_summaries = tf.summary.merge([ep_reward])
        self.individual_summaries = tf.summary.merge(actions + prices + [individual_reward, 
                                                     individual_pnl, individual_tc])
        self.batch_summaries = tf.summary.merge([qfunc_loss])

    def train(self):
        global_step = 0
        training_rewards = []
        for episode in range(1, self.num_episodes+1):
            state = self.env.reset()
            episode_rewards = 0
            episode_ave_max_q = 0
            for time_step in range(self.episode_length):
                action = self.actor.predict(asset_inputs=np.array([state.asset_features]),
                                            portfolio_inputs=np.array([state.portfolio_allocation]))[0]
                #print("ACTION before:", softmax(action))
                noise = self.actor_noise()
                noise = 0
                #print("NOISE:", noise)
                action += noise
                action = softmax(action) # take softmax here
                #print("ACTION after:", action)
                trans_state, reward, terminal, info = self.env.step(action)
                episode_rewards += reward

                # self.rpb.store_w_terminal(old_state=state,
                #                           action=action,
                #                           reward=reward,
                #                           terminal=terminal,
                #                           new_state=trans_state)
                self.rpb.add(obs_t=state.features,
                             action=action,
                             reward=reward,
                             obs_tp1=trans_state.features,
                             done=terminal)
                # if self.rpb.ready(self.batch_size):
                if len(self.rpb._storage) >= self.batch_size:
                    # batch_states, batch_actions, batch_rewards, batch_terminal, batch_trans_state \
                    #     = self.rpb.sample_batch(batch_size=self.batch_size)
                    experiences = self.rpb.sample(batch_size=self.batch_size, beta=0.5)
                    batch_states, batch_actions, batch_rewards, batch_trans_state, batch_terminal, \
                        weights, rank_e_id = experiences
                    batch_asset_features, batch_portfolio = convert_features(features=batch_states,
                                                                             asset_features_shape=self.actor.asset_features_shape,
                                                                             portfolio_features_shape=[self.actor.a_dim])
                    batch_trans_asset_features, batch_trans_portfolio = \
                        convert_features(features=batch_trans_state,
                                         asset_features_shape=self.actor.asset_features_shape,
                                         portfolio_features_shape=[self.actor.a_dim])
                    weights = np.expand_dims(weights, axis=1)

                    target_actions = self.actor.predict_target(asset_inputs=batch_trans_asset_features,
                                                               portfolio_inputs=batch_trans_portfolio) # [batch_size, action_dim]
                    target_q = self.critic.predict_target(asset_inputs=batch_trans_asset_features,
                                                          portfolio_inputs=batch_trans_portfolio, # [batch_size, 1]
                                                          action=target_actions)
                    batch_y = []
                    for ind in range(self.batch_size):
                        if batch_terminal[ind]:
                            batch_y.append([batch_rewards[ind]])
                        else:
                            batch_y.append(batch_rewards[ind] + self.gamma*target_q[ind])
                    batch_y = np.array(batch_y) # [batch_size, 1]
                    loss, out, _ = self.critic.train(asset_inputs=batch_asset_features,
                                                     portfolio_inputs=batch_portfolio,
                                                     action=batch_actions,
                                                     predicted_q_value=batch_y,
                                                     weights=weights)
                    deltas = np.squeeze(np.abs(out - batch_y))
                    deltas[deltas==0] = 0.001
                    self.rpb.update_priorities(idxes=rank_e_id,
                                               priorities=deltas)
                    policy_actions = self.actor.predict(asset_inputs=batch_asset_features,
                                                        portfolio_inputs=batch_portfolio) # [batch_size, num_assets]
                    policy_actions = softmax(policy_actions, axis=-1) # take softmax here
                    action_grads = self.critic.action_gradients(asset_inputs=batch_asset_features,
                                                                portfolio_inputs=batch_portfolio,
                                                                actions=policy_actions)[0]
                    self.actor.train(asset_inputs=batch_asset_features,
                                     portfolio_inputs=batch_portfolio,
                                     a_gradient=np.array(action_grads))
                    self.critic.update_target_network()
                    self.actor.update_target_network()

                    summary = self.sess.run(self.batch_summaries,
                                            feed_dict={
                                                self.qfunc_loss: loss
                                            })
                    self.writer.add_summary(summary, global_step)

                summary = self.sess.run(self.individual_summaries,
                                        feed_dict={
                                            self.actions: action,
                                            self.prices: state.price,
                                            self.individual_reward: reward,
                                            self.individual_pnl: info['pnl'],
                                            self.individual_tc: info['tc']
                                        })
                self.writer.add_summary(summary, global_step)

                global_step += 1
                state = trans_state

                if terminal:
                    print("Episode number:", episode)
                    summary = self.sess.run(self.episode_summaries, feed_dict={self.episode_reward: episode_rewards})
                    self.writer.add_summary(summary, episode)
                    print("Reward:", episode_rewards)
                    break

                elif time_step == (self.episode_length - 1):
                    print("Episode number:", episode)
                    summary = self.sess.run(self.episode_summaries, feed_dict={self.episode_reward: episode_rewards})
                    self.writer.add_summary(summary, episode)
                    print("Reward:", episode_rewards)
                    break

            # summary = self.sess.run(self.summary_ops, feed_dict={self.episode_reward: episode_rewards})
            # self.writer.add_summary(summary, global_step)

        #     epsiode_rewards.append(np.sum(rewards))
        #     if epsilon > 0.1:
        #         epsilon -= 2.0 / self.num_episodes

            if (episode % 50) == 0:
                self.infer(train=False, episode=episode)

        # plt.plot(epsiode_rewards)
        # plt.savefig("./episode_rewards.png")

        # self.infer(train=False, episode=episode)

    def infer(self, train, episode):
        if not train:
            episode_length = self.env.datacontainer.train_length - 1 - self.env.history_length
            tsm = TradingStateModel(datacontainer=self.env.datacontainer,
                                    episode_length=episode_length,
                                    history_length=self.env.history_length,
                                    is_training=True,
                                    commission_percentage=self.env.commission_percentage)
            state = tsm.reset()
            prices = [state.price] # [episode_length]
            rewards = [0] # [episode_length]
            allocations = [state.portfolio_allocation] # [episode_length]

            for _ in range(episode_length):
                batch_asset_features, batch_portfolio = convert_features(features=np.array([state.features]),
                                                                         asset_features_shape=self.actor.asset_features_shape,
                                                                         portfolio_features_shape=[self.actor.a_dim])   
                action = self.actor.predict_target(asset_inputs=batch_asset_features,
                                                   portfolio_inputs=batch_portfolio)[0]
                trans_state, reward, terminal, info = tsm.step(action)
                prices.append(trans_state.price)
                rewards.append(reward)
                allocations.append(trans_state.portfolio_allocation)
                state = trans_state

            prices = np.array(prices)
            rewards = np.array(rewards)
            allocations = np.array(allocations)

            f, axarr = plt.subplots(3, sharex=True)
            axarr[0].set_ylabel('Price')
            for ind in range(self.env.datacontainer.num_assets):
                axarr[0].plot(prices[:, ind])

            axarr[1].set_ylabel('Cumulative Reward')
            axarr[1].plot(np.cumsum(rewards))

            axarr[2].set_ylabel('Action')
            for ind in range(self.env.datacontainer.num_assets):
                axarr[2].plot(allocations[:, ind])

            dataset = 'Train' if train else 'Test'
            title = '{}, Total Reward: {}'.format(dataset,
                                                  np.sum(rewards))
            plt.savefig(os.path.join(self.infer_directory, str(episode)+".png"))
