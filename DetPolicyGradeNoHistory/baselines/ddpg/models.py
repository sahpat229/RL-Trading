import tensorflow as tf
import tensorflow.contrib as tc

from keras.layers import Activation, Input, Concatenate, Conv2D, Dense, Reshape

def determine_shape(input_amount, kernel_size, padding, stride):
    return int((input_amount - kernel_size + 2*padding)/stride + 1)

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, num_asset_features, num_actions, asset_features_shape, portfolio_features_shape,
                 name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.num_asset_features = num_asset_features
        self.num_actions = num_actions
        self.asset_features_shape = asset_features_shape
        self.portfolio_features_shape = portfolio_features_shape

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.layers.dense(obs, 23)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.num_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.softmax(x)
        return x

class Critic(Model):
    def __init__(self, num_asset_features, num_actions, asset_features_shape, portfolio_features_shape, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.num_asset_features = num_asset_features
        self.num_actions = num_actions
        self.asset_features_shape = asset_features_shape
        self.portfolio_features_shape = portfolio_features_shape

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.layers.dense(obs, 23)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            actions = tf.layers.dense(action, 23)
            if self.layer_norm:
                actions = tc.layers.layer_norm(actions, center=True, scale=True)

            x = x + actions
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
