import numpy as np
import tensorflow as tf

class Network():
    def __init__(self, sess, batch_size, batch_norm=False, learning_rate=1e-3,
                 dropout=0.5, history_length=50, datacontainer=None, epochs=50,
                 is_target=True):
        self.sess = sess
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.learning_rate=learning_rate
        self.dropout = dropout
        self.datacontainer = datacontainer
        self.epochs = epochs
        self.target = "target" if is_target else "trainer"

    @staticmethod
    def update_target_graph(from_scope, to_scope, tau):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op = to_var.assign(tf.scalar_mul(tau, from_var) + tf.scalar_mul(1-tau, to_var))
            op_holder.append(op)
        return op_holder

    @staticmethod
    def assign_target_graph(from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op = to_var.assign(from_var)
            op_holder.append(op)
        return op_holder

class ActorNetwork(Network):
    """
    Non-recurrent Actor Network, computes mu(s)
    """
    def __init__(self, sess, batch_size, batch_norm=False, learning_rate=1e-3,
                 dropout=0.5, history_length=50, datacontainer=None, epochs=50,
                 is_target=True, coin_boundary=5):
        super().__init__(sess=sess,
                         batch_size=batch_size,
                         batch_norm=batch_norm,
                         learning_rate=learning_rate,
                         dropout=dropout,
                         history_length=history_length,
                         datacontainer=datacontainer,
                         epochs=epochs,
                         is_target=is_target)
        self.coin_boundary = coin_boundary
        self.build_model()

    def build_model(self):
        self.scope = "actor-"+self.target
        with tf.variable_scope(self.scope) as scope:
            self.inputs = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.datacontainer.num_flattened_features])
            print(self.inputs.shape)
            self.is_training = tf.placeholder(dtype=tf.bool,
                                              shape=None)
            if self.batch_norm:
                use_bias = False
            else:
                use_bias = True

            net = tf.layers.dense(inputs=self.inputs,
                                  units=200,
                                  activation=None,
                                  use_bias=use_bias,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  trainable=True)
            if self.batch_norm:
                net = tf.layers.batch_normalization(inputs=net,
                                                    training=self.is_training,
                                                    trainable=True,
                                                    scale=True)
            net = tf.nn.relu(net)
            net = tf.layers.dense(inputs=net,
                                  units=100,
                                  activation=None,
                                  use_bias=use_bias,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  trainable=True)
            if self.batch_norm:
                net = tf.layers.batch_normalization(inputs=net,
                                                    training=self.is_training,
                                                    trainable=True,
                                                    scale=True)
            net = tf.nn.relu(net)
            w_init = tf.random_uniform_initializer(minval=-3e-3,
                                                   maxval=3e-3)
            net = tf.layers.dense(inputs=net,
                                  units=1,
                                  activation=tf.nn.tanh,
                                  use_bias=True,
                                  kernel_initializer=w_init,
                                  trainable=True)
            self.output = tf.scalar_mul(self.coin_boundary, net)

            network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               self.scope)
            self.action_gradient = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.datacontainer.num_assets])
            unnormalized_action_gradients = tf.gradients(self.output,
                                                         network_params,
                                                         -self.action_gradient)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), unnormalized_action_gradients))
            # Optimization Op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                           self.scope)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).\
                    apply_gradients(zip(self.actor_gradients, network_params))

    def train_step(self, inputs, action_gradient):
        self.sess.run(self.optimizer,
                      feed_dict={
                          self.inputs: inputs,
                          self.action_gradient: action_gradient,
                          self.is_training: True
                      })

    def select_action(self, inputs):
        output = self.sess.run(self.output,
                               feed_dict= {
                                   self.inputs: inputs,
                                   self.is_training: False
                                   })
        return output # size is [batch_size, num_assets]

    @staticmethod
    def update_actor(sess, tau):
        ops = Network.update_target_graph("actor-trainer", "actor-target", tau)
        sess.run(ops)

class CriticNetwork(Network):
    def __init__(self, sess, batch_size, batch_norm=False, learning_rate=1e-3,
                 dropout=0.5, history_length=50, datacontainer=None, epochs=50,
                 is_target=True):
        super().__init__(sess=sess,
                         batch_size=batch_size,
                         batch_norm=batch_norm,
                         learning_rate=learning_rate,
                         dropout=dropout,
                         history_length=history_length,
                         datacontainer=datacontainer,
                         epochs=epochs,
                         is_target=is_target)
        self.build_model()

    def build_model(self):
        self.scope = "critic"+self.target
        with tf.variable_scope(self.scope) as scope:
            self.inputs = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.datacontainer.num_flattened_features])
            self.actions = tf.placeholder(dtype=tf.float32,
                                          shape=[None, 1])
            self.is_training = tf.placeholder(dtype=tf.bool,
                                              shape=None)

            if self.batch_norm:
                use_bias = False
            else:
                use_bias = True

            net = tf.layers.dense(inputs=self.inputs,
                                  units=200,
                                  activation=None,
                                  use_bias=use_bias,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  trainable=True)
            if self.batch_norm:
                net = tf.layers.batch_normalization(inputs=net,
                                                    training=self.is_training,
                                                    trainable=True,
                                                    scale=True)
            net = tf.nn.relu(net)
            states = tf.layers.dense(inputs=net,
                                     units=100,
                                     activation=None,
                                     use_bias=False,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     trainable=True)
            actions = tf.layers.dense(inputs=self.actions,
                                      units=100,
                                      activation=None,
                                      use_bias=True,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      trainable=True)
            net = states + actions
            w_init = tf.random_uniform_initializer(minval=-3e-3,
                                                   maxval=3e-3)
            net = tf.layers.dense(inputs=net,
                                  units=1,
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=w_init,
                                  trainable=True)
            self.output = net
            self.predicted_q_value = tf.placeholder(dtype=tf.float32,
                                                    shape=[None, 1])
            self.loss = tf.square(self.output - self.predicted_q_value)
            self.loss = tf.reduce_mean(self.loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                           self.scope)
            with tf.control_dependencies(update_ops):
                model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    self.scope)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     var_list=model_variables)
            self.action_gradients = tf.gradients(self.output, self.actions)

    def train_step(self, inputs, actions, predicted_q_value):
        loss, _ = self.sess.run([self.loss, self.optimizer],
                                feed_dict={
                                    self.inputs: inputs,
                                    self.actions: actions,
                                    self.predicted_q_value: predicted_q_value,
                                    self.is_training: True
                                })
        #print("LOSS:", loss)
        return loss

    def get_q_value(self, inputs, actions):
        q_value = self.sess.run(self.output,
                                feed_dict={
                                    self.inputs: inputs,
                                    self.actions: actions,
                                    self.is_training: False
                                })
        return q_value

    def get_action_gradients(self, inputs, actions):
        action_gradients = self.sess.run(self.action_gradients,
                                         feed_dict={
                                            self.inputs: inputs,
                                            self.actions: actions,
                                            self.is_training: False
                                         })
        return action_gradients

    @staticmethod
    def update_critic(sess, tau):
        ops = Network.update_target_graph("critic-trainer", "critic-target", tau)
        sess.run(ops)