import numpy as np
import tensorflow as tf
import tflearn

from keras.layers import Activation, Input, Concatenate, Conv2D, Dense, Reshape, Dropout, Add
from keras.models import Model

def determine_shape(input_amount, kernel_size, padding, stride):
    return int((input_amount - kernel_size + 2*padding)/stride + 1)

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, asset_features_shape, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.asset_features_shape = asset_features_shape #[num_assets, num_features]
        self.a_dim = action_dim # [num_assets]
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.asset_inputs, self.portfolio_inputs, self.scaled_out = self.create_actor_network()
        self.soft_out = Activation('softmax')(self.scaled_out)

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_asset_inputs, self.target_portfolio_inputs, self.target_scaled_out = self.create_actor_network()
        self.target_soft_out = Activation('softmax')(self.target_scaled_out)

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        self.assign_target_network_params = \
            [self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.soft_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        asset_inputs = Input(shape=self.asset_features_shape) # [batch_size, num_assets, features]
        portfolio_inputs = Input(shape=[self.a_dim]) # [batch_size, num_assets]
        asset_inputs_reshaped = Reshape((np.prod(self.asset_features_shape),))(asset_inputs)

        net = Concatenate(axis=-1)([asset_inputs_reshaped, portfolio_inputs])
        net = Dense(23)(net)
        net = Activation('relu')(net)
        net = Dropout(0.5)(net)
        net = Dense(self.a_dim)(net)
        return asset_inputs, portfolio_inputs, net

    def train(self, asset_inputs, portfolio_inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.asset_inputs: asset_inputs,
            self.portfolio_inputs: portfolio_inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, asset_inputs, portfolio_inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.asset_inputs: asset_inputs,
            self.portfolio_inputs: portfolio_inputs
        })

    def predict_target(self, asset_inputs, portfolio_inputs):
        return self.sess.run(self.target_soft_out, feed_dict={
            self.target_asset_inputs: asset_inputs,
            self.target_portfolio_inputs: portfolio_inputs
        })

    def assign_target_network(self):
        self.sess.run(self.assign_target_network_params)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, asset_features_shape, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.asset_features_shape = asset_features_shape
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.asset_inputs, self.portfolio_inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_asset_inputs, self.target_portfolio_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.assign_target_network_params = \
            [self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # weights for prioritized experience replay
        self.weights = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.square(self.predicted_q_value - self.out)
        self.loss = tf.multiply(self.weights, self.loss)
        self.loss = tf.reduce_mean(self.loss)

        #self.loss = tflearn.mean_square(self.predicted_q_value, self.out)

        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        asset_inputs = Input(shape=self.asset_features_shape) # [batch_size, num_assets, history_length, features]
        portfolio_inputs = Input(shape=[self.a_dim]) # []
        action_inputs = Input(shape=[self.a_dim])
        asset_inputs_reshaped = Reshape((np.prod(self.asset_features_shape),))(asset_inputs)

        net = Concatenate(axis=-1)([asset_inputs_reshaped, portfolio_inputs])
        net = Dense(23)(net)
        net = Activation('relu')(net)
        net = Dropout(0.5)(net)

        actions = Dense(23)(action_inputs)
        actions = Activation('relu')(actions)
        net = Add()([net, actions])
        out = Dense(1)(net)

        return asset_inputs, portfolio_inputs, action_inputs, out

    def train(self, asset_inputs, portfolio_inputs, action, predicted_q_value, weights):
        return self.sess.run([self.loss, self.out, self.optimize], feed_dict={
            self.asset_inputs: asset_inputs,
            self.portfolio_inputs: portfolio_inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.weights: weights
        })

    def predict(self, asset_inputs, portfolio_inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.asset_inputs: asset_inputs,
            self.portfolio_inputs: portfolio_inputs,
            self.action: action
        })

    def predict_target(self, asset_inputs, portfolio_inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_asset_inputs: asset_inputs,
            self.target_portfolio_inputs: portfolio_inputs,
            self.target_action: action
        })

    def action_gradients(self, asset_inputs, portfolio_inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.asset_inputs: asset_inputs,
            self.portfolio_inputs: portfolio_inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def assign_target_network(self):
        self.sess.run(self.assign_target_network_params)