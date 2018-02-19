import matplotlib.pyplot as plt
import numpy as np 
import random
import tensorflow as tf 

from networks import Network, ActorNetwork, CriticNetwork

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

    def random_action(self, num_dimensions):
        numbers = np.random.uniform(size=num_dimensions-1)
        numbers = np.concatenate((numbers, [0, 1]),
                                 axis=0)
        numbers = np.sort(numbers)
        return np.diff(numbers)

    def train(self):
        epsilon = 1.00
        epsiode_rewards = []
        for episode in range(1, self.num_episodes+1):
            state, reward = self.tsm.initialize()
            rewards = []
            while not state.terminated:
                if random.random() < epsilon:
                    action = self.random_action(num_dimensions=self.datacontainer.num_assets)
                else:
                    action = self.actor_trainer.select_action(inputs=np.array([state.asset_features]))
                    action = np.squeeze(action)
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
                        target_action = self.actor_target.select_action(inputs=np.array([new_state.asset_features]))
                        target_q = self.critic_target.get_q_value(inputs=np.array([new_state.asset_features]),
                                                                  actions=target_action)[0]
                        y = reward + self.gamma * target_q
                        #print("Y:", y, "Target_q:", target_q, "Target_action:", target_action, "reward:", reward)
                        batch_y.append(y)
                        batch_states.append(old_state.asset_features)
                        batch_actions.append(action)
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
        plt.plot(epsiode_rewards)
        plt.show()