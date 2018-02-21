import matplotlib.pyplot as plt
import numpy as np 
import random
import tensorflow as tf 

from networks import Network, ActorNetwork, CriticNetwork
from tradingstatemodel import  TradingStateModel
from tqdm import tqdm

class DDPG():
    def __init__(self, sess, batch_size, num_episodes, actor_target, actor_trainer,
                 critic_target, critic_trainer, trading_state_model, replay_buffer,
                 datacontainer, gamma, tau):
        self.sess = sess
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.actor_target = actor_target
        self.actor_trainer = actor_trainer
        self.critic_target = critic_target
        self.critic_trainer = critic_trainer
        self.tsm = trading_state_model
        self.rpb = replay_buffer
        self.datacontainer = datacontainer
        self.gamma = gamma
        self.tau = tau

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(Network.assign_target_graph("actor-trainer", "actor-target"))
        self.sess.run(Network.assign_target_graph("critic-trainer", "critic-target"))

    def random_action(self):
        return np.random.uniform(low=-self.tsm.coin_boundary,
                                 high=self.tsm.coin_boundary)

    def train(self):
        epsilon = 1.00
        epsiode_rewards = []
        for episode in range(1, self.num_episodes+1):
            state, reward = self.tsm.initialize()
            rewards = []
            for _ in tqdm(range(self.tsm.episode_length)):
                if random.random() < epsilon:
                    action = self.random_action()
                else:
                    action = self.actor_trainer.select_action(inputs=np.array([state.features]))[0][0]
                trans_state, reward = self.tsm.step(action)
                rewards.append(reward)
                self.rpb.store(old_state=state,
                               action=action,
                               reward=reward,
                               new_state=trans_state)
                if self.rpb.ready(self.batch_size):
                    transitions = self.rpb.sample(batch_size=self.batch_size,
                                                  recurrent=False)
                    batch_states = [] # [batch_size, num_assets, num_features]
                    batch_actions = [] # [batch_size, num_assets]
                    batch_y = [] # [batch_size, 1]
                    for transition in transitions:
                        old_state, action, reward, new_state = transition
                        target_action = self.actor_target.select_action(inputs=np.array([new_state.features]))
                        target_q = self.critic_target.get_q_value(inputs=np.array([new_state.features]),
                                                                  actions=target_action)[0]
                        y = reward + self.gamma * target_q
                        #print("Y:", y, "Target_q:", target_q, "Target_action:", target_action, "reward:", reward)
                        batch_y.append(y)
                        batch_states.append(old_state.features)
                        batch_actions.append([action])
                    self.critic_trainer.train_step(inputs=np.array(batch_states),
                                                   actions=np.array(batch_actions),
                                                   predicted_q_value=np.array(batch_y))
                    policy_actions = self.actor_trainer.select_action(inputs=np.array(batch_states)) # [batch_size, num_assets]
                    action_grads = self.critic_trainer.get_action_gradients(inputs=np.array(batch_states),
                                                                            actions=policy_actions)[0]
                    self.actor_trainer.train_step(inputs=np.array(batch_states),
                                                  action_gradient=np.array(action_grads))
                    ActorNetwork.update_actor(self.sess, self.tau)
                    CriticNetwork.update_critic(self.sess, self.tau)
                state = trans_state

            epsiode_rewards.append(np.sum(rewards))
            if epsilon > 0.1:
                epsilon -= 2.0 / self.num_episodes

            if (episode % 1) == 0:
                self.infer(train=False, episode=episode)

        plt.plot(epsiode_rewards)
        plt.savefig("./episode_rewards.png")

        self.infer(train=False, episode=episode)

    def infer(self, train, episode):
        if not train:
            episode_length = self.datacontainer.test_length - 1
            tsm = TradingStateModel(datacontainer=self.datacontainer,
                                    episode_length=episode_length,
                                    is_training=train,
                                    commission_percentage=self.tsm.commission_percentage,
                                    coin_boundary=self.tsm.coin_boundary)
            state, reward = tsm.initialize()
            prices = [state.prices] # [episode_length, num_assets]
            rewards = [reward] # [episode_length]
            coins = [state.coins] # [episode_length, num_assets]

            for _ in tqdm(range(episode_length)):
                action = self.actor_target.select_action(inputs=np.array([state.features]))[0][0]
                #action = self.random_action()
                trans_state, reward = tsm.step(action)
                prices.append(trans_state.prices)
                rewards.append(reward)
                coins.append(trans_state.coins)
                state = trans_state

            prices = np.array(prices)
            rewards = np.array(rewards)
            coins = np.array(coins)

            f, axarr = plt.subplots(3, sharex=True)
            axarr[0].set_ylabel('Price')
            for ind in range(prices.shape[1]):
                axarr[0].plot(prices[:, ind])

            axarr[1].set_ylabel('Cumulative Reward')
            axarr[1].plot(np.cumsum(rewards))

            axarr[2].set_ylabel('Action')
            axarr[2].plot(coins)

            dataset = 'Train' if train else 'Test'
            title = '{}, Total Reward: {}'.format(dataset,
                                                  np.sum(rewards))
            plt.savefig("./infer"+str(episode)+".png")