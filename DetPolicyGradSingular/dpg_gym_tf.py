import matplotlib
matplotlib.use('Agg')

import gym
import matplotlib.pyplot as plt
import numpy as np 
import random
import tensorflow as tf 

from networks_tf import ActorNetwork, CriticNetwork
from tradingstatemodel import  TradingStateModel
from tqdm import tqdm

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
                 env, replay_buffer, gamma, tau, actor_noise):
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

        self.sess.run(tf.global_variables_initializer())
        self.actor.assign_target_network()
        self.critic.assign_target_network()
        self.writer = tf.summary.FileWriter("./tensorboard", sess.graph)
        self.build_summaries()

    def build_summaries(self):
        self.episode_reward = tf.placeholder(dtype=tf.float32,
                                             shape=None)
        tf.summary.scalar("Reward", self.episode_reward)
        self.summary_ops = tf.summary.merge_all()

    def train(self):
        global_step = 0
        training_rewards = []
        for episode in range(1, self.num_episodes+1):
            state = self.env.reset().features
            state = np.reshape(state, (self.actor.s_dim, ))
            episode_rewards = 0
            episode_ave_max_q = 0
            for time_step in range(self.episode_length):
                action = self.actor.predict(inputs=np.array([state]))[0] + self.actor_noise() # [action_dim]
                trans_state, reward, terminal, info = self.env.step(action)
                trans_state = trans_state.features
                trans_state = np.reshape(trans_state, (self.actor.s_dim, ))
                episode_rewards += reward

                # self.rpb.store_w_terminal(old_state=state,
                #                           action=action,
                #                           reward=reward,
                #                           terminal=terminal,
                #                           new_state=trans_state)
                self.rpb.add(obs_t=state,
                             action=action,
                             reward=reward,
                             obs_tp1=trans_state,
                             done=terminal)
                # if self.rpb.ready(self.batch_size):
                if len(self.rpb._storage) >= self.batch_size:
                    # batch_states, batch_actions, batch_rewards, batch_terminal, batch_trans_state \
                    #     = self.rpb.sample_batch(batch_size=self.batch_size)
                    experiences = self.rpb.sample(batch_size=self.batch_size, beta=0.5)
                    print("LEN:", len(experiences))
                    batch_states, batch_actions, batch_rewards, batch_trans_state, batch_terminal, \
                        weights, rank_e_id = experiences
                    weights = np.expand_dims(weights, axis=1)

                    print(batch_states.shape, batch_actions.shape, batch_rewards.shape, batch_trans_state.shape,
                          batch_terminal.shape, weights.shape, len(rank_e_id))    

                    target_actions = self.actor.predict_target(inputs=batch_trans_state) # [batch_size, action_dim]
                    target_q = self.critic.predict_target(inputs=batch_trans_state, # [batch_size, 1]
                                                          action=target_actions)
                    batch_y = []
                    for ind in range(self.batch_size):
                        if batch_terminal[ind]:
                            batch_y.append([batch_rewards[ind]])
                        else:
                            batch_y.append(batch_rewards[ind] + self.gamma*target_q[ind])
                    batch_y = np.array(batch_y) # [batch_size, 1]
                    out, _ = self.critic.train(inputs=batch_states,
                                               action=batch_actions,
                                               predicted_q_value=batch_y,
                                               weights=weights)
                    deltas = np.squeeze(np.abs(out - batch_y))
                    deltas[deltas==0] = 0.001
                    self.rpb.update_priorities(idxes=rank_e_id,
                                               priorities=deltas)
                    policy_actions = self.actor.predict(inputs=batch_states) # [batch_size, num_assets]
                    action_grads = self.critic.action_gradients(inputs=batch_states,
                                                                actions=policy_actions)[0]
                    self.actor.train(inputs=batch_states,
                                     a_gradient=np.array(action_grads))
                    self.critic.update_target_network()
                    self.actor.update_target_network()

                global_step += 1
                state = trans_state

                if terminal:
                    print("Episode number:", episode)
                    summary = self.sess.run(self.summary_ops, feed_dict={self.episode_reward: episode_rewards})
                    self.writer.add_summary(summary, episode)
                    print("Reward:", episode_rewards)
                    break

                elif time_step == (self.episode_length - 1):
                    print("Episode number:", episode)
                    summary = self.sess.run(self.summary_ops, feed_dict={self.episode_reward: episode_rewards})
                    self.writer.add_summary(summary, episode)
                    print("Reward:", episode_rewards)
                    break

            # summary = self.sess.run(self.summary_ops, feed_dict={self.episode_reward: episode_rewards})
            # self.writer.add_summary(summary, global_step)

        #     epsiode_rewards.append(np.sum(rewards))
        #     if epsilon > 0.1:
        #         epsilon -= 2.0 / self.num_episodes

            if (episode % 25) == 0:
                self.infer(train=False, episode=episode)

        # plt.plot(epsiode_rewards)
        # plt.savefig("./episode_rewards.png")

        # self.infer(train=False, episode=episode)

    def infer(self, train, episode):
        if not train:
            episode_length = self.env.datacontainer.test_length - 1
            tsm = TradingStateModel(datacontainer=self.env.datacontainer,
                                    episode_length=episode_length,
                                    is_training=True,
                                    commission_percentage=self.env.commission_percentage,
                                    coin_boundary=self.env.coin_boundary)
            state = tsm.reset()
            prices = [state.price] # [episode_length]
            rewards = [0] # [episode_length]
            coins = [state.coins] # [episode_length]

            for _ in tqdm(range(episode_length)):
                action = self.actor.predict_target(inputs=np.array([state.features]))[0]
                #action = self.random_action()
                trans_state, reward, terminal, info = tsm.step(action)
                prices.append(trans_state.price)
                rewards.append(reward)
                coins.append(trans_state.coins)
                state = trans_state

            prices = np.array(prices)
            rewards = np.array(rewards)
            coins = np.array(coins)

            f, axarr = plt.subplots(3, sharex=True)
            axarr[0].set_ylabel('Price')
            axarr[0].plot(prices)

            axarr[1].set_ylabel('Cumulative Reward')
            axarr[1].plot(np.cumsum(rewards))

            axarr[2].set_ylabel('Action')
            axarr[2].plot(coins)

            dataset = 'Train' if train else 'Test'
            title = '{}, Total Reward: {}'.format(dataset,
                                                  np.sum(rewards))
            plt.savefig("./infer_ims/"+str(episode)+".png")
