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

            asset_inputs = tf.slice(obs,
                                    [0, 0],
                                    [-1, self.num_asset_features])
            portfolio_inputs = tf.slice(obs,
                                        [0, self.num_asset_features],
                                        [-1, self.num_actions])
            asset_inputs = Reshape(self.asset_features_shape)(asset_inputs)
            portfolio_inputs = Reshape(self.portfolio_features_shape)(portfolio_inputs)
            x = tc.layers.conv2d(inputs=asset_inputs,
                                 num_outputs=3,
                                 kernel_size=[1, 3],
                                 padding='VALID',
                                 activation_fn=None)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tc.layers.conv2d(inputs=x,
                                 num_outputs=20,
                                 kernel_size=[1, determine_shape(input_amount=self.asset_features_shape[1],
                                                                 kernel_size=3,
                                                                 padding=0,
                                                                 stride=1)],
                                 padding='VALID',
                                 activation_fn=None)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = Concatenate(axis=-1)([portfolio_inputs, x])
            x = tc.layers.conv2d(inputs=x,
                                 num_outputs=1,
                                 kernel_size=[1, 1],
                                 padding='VALID',
                                 activation_fn=None)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = Reshape((self.num_actions,))(x)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
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

            asset_inputs = tf.slice(obs,
                                    [0, 0],
                                    [-1, self.num_asset_features])
            portfolio_inputs = tf.slice(obs,
                                        [0, self.num_asset_features],
                                        [-1, self.num_actions])
            asset_inputs = Reshape(self.asset_features_shape)(asset_inputs)
            portfolio_inputs = Reshape(self.portfolio_features_shape)(portfolio_inputs)
            x = tc.layers.conv2d(inputs=asset_inputs,
                                 num_outputs=3,
                                 kernel_size=[1, 3],
                                 padding='VALID',
                                 activation_fn=None)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tc.layers.conv2d(inputs=x,
                                 num_outputs=20,
                                 kernel_size=[1, determine_shape(input_amount=self.asset_features_shape[1],
                                                                 kernel_size=3,
                                                                 padding=0,
                                                                 stride=1)],
                                 padding='VALID',
                                 activation_fn=None)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = Concatenate(axis=-1)([portfolio_inputs, x])
            x = tc.layers.conv2d(inputs=x,
                                 num_outputs=1,
                                 kernel_size=[1, 1],
                                 padding='VALID',
                                 activation_fn=None)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = Reshape((self.num_actions,))(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
