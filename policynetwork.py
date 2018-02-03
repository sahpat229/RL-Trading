import numpy as np
import tensorflow as tf

class PolicyNetwork():
    def __init__(self, sess, num_assets, history_length, input_batch_size, commission_rate, mode,
                 num_epochs, use_batch_norm, data_container, PVM):
        self.sess = sess
        self.num_assets = num_assets
        self.history_length = history_length
        self.input_batch_size = input_batch_size
        self.commission_rate = commission_rate
        self.num_epochs = num_epochs
        self.data_container = data_container
        self.PVM = PVM
        if mode == 'EIEE':
            self.build_cnn(use_batch_norm)

    @staticmethod
    def determine_size(input_size, kernel_width, kernel_padding=0, stride=1):
        return (input_size - kernel_width + 2*kernel_padding) / stride + 1

    def build_cnn(self, use_batch_norm):
        # Batch size is the episode length
        self.batch_size = tf.placeholder(dtype=tf.int32,
                                         shape=[])
        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.num_assets, self.history_length, 3]) # X_t's
        self.previous_weights = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.num_assets, 1, 1]) # w_(t-1)'s
        self.future_prices = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.num_assets]) # y_(t+1)'s
        self.is_training = tf.placeholder(dtype=tf.bool,
                                     shape=None)
        use_bias = True
        activation = tf.nn.relu
        if use_batch_norm:
            use_bias = False
            activation = None

        network = tf.layers.conv2d(inputs=self.input,
                                   filters=2,
                                   kernel_size=[1, 3],
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-9),
                                   use_bias=use_bias,
                                   activation=activation)
        if use_batch_norm:
            network = tf.layers.batch_normalization(inputs=network,
                                                    training=self.is_training,
                                                    trainable=True,
                                                    scale=True)
            network = tf.nn.relu(network)

        network = tf.layers.conv2d(inputs=network,
                                   filters=20,
                                   kernel_size=[1, PolicyNetwork.determine_size(input_size=self.history_length,
                                                                                kernel_width=3)],
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-9),
                                   use_bias=use_bias,
                                   activation=activation)
        if use_batch_norm:
            network = tf.layers.batch_normalization(inputs=network,
                                                    training=self.is_training,
                                                    trainable=True,
                                                    scale=True)
            network = tf.nn.relu(network)

        network = tf.concat(values=[network, self.previous_weights],
                            axis=3)
        network = tf.layers.conv2d(inputs=network,
                                   filters=1,
                                   kernel_size=[1, 1],
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-9),
                                   activation=None)
        network = tf.squeeze(network, axis=[2, 3]) # remove width and channels dimensions, which are both of size 1
        self.cash_bias = tf.get_variable(name='cash_bias',
                                         shape=[1, 1],
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer())
        cash_bias = tf.tile(input=self.cash_bias,
                            multiples=[self.batch_size, 1]) # replicate the cash bias across the batch
        network = tf.concat(values=[cash_bias, network],
                            axis=1) # shape is now [self.batch_size, num_assets+1]
        self.voting_scores = tf.nn.softmax(network)

        future_prices = tf.concat(values=[tf.ones([self.batch_size, 1]), self.future_prices],
                                  axis=1)

        future_weight_prime = tf.div((future_prices * self.voting_scores), 
            tf.expand_dims(tf.reduce_sum(future_prices * self.voting_scores, axis=1),
            axis=1)) # w_(t+1)'
        # shape: [batch_size, num_assets+1]

        # if we index future_weight_prime from t = 1 to N, and index our network's output weights
        # from t = 2 to N+1, we can overcome the problem of needing to run the portfolio model an extra time

        mu = 1 - tf.reduce_sum(tf.abs(future_weight_prime[:self.batch_size-1] - self.voting_scores[1:]),
                               axis=1)*self.commission_rate # these are the mu_(t+1)'s per time step
        # size: [self.batch_size-1, 1]
        mu = tf.concat([tf.ones(1), mu],
                       axis=0)
        # size: [self.batch_size, 1]
        self.reward = tf.log(mu * tf.reduce_sum(future_prices * self.voting_scores,
                                                axis=1))
        # size: [self.batch_size, 1]

        self.loss = -tf.reduce_mean(self.reward) + tf.losses.get_regularization_loss()

    def train_init(self):
        model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, var_list=model_variables)
        self.sess.run(tf.global_variables_initializer())
        #self.sess.run(tf.local_variables_initializer())

    def train_step(self, feed_dict):
        batched_weights, bias, reward, loss = self.sess.run([self.voting_scores, self.cash_bias, self.reward, self.loss], 
                                                            feed_dict=feed_dict)
        print("Bias:", bias)
        print("Weights", batched_weights[0, :])
        return batched_weights, reward, loss

    def train(self):
        self.train_init()
        data_generator = self.data_container.yield_data(history_length=self.history_length,
                                                        batch_size=self.input_batch_size)

        epochs = 0
        while epochs < self.num_epochs:
            try:
                batched_data, time = next(data_generator)
                previous_weights = self.PVM.read_batch(time=time,
                                                       batch_size=self.input_batch_size)
                previous_weights = previous_weights[:, :, np.newaxis, np.newaxis]
                feed_dict = {self.input: batched_data['batch_current_prices'],
                             self.future_prices: batched_data['batch_future_prices'],
                             self.previous_weights: previous_weights,
                             self.is_training: True,
                             self.batch_size: self.input_batch_size}
                batched_weights, reward, loss = self.train_step(feed_dict=feed_dict)
                self.PVM.input_batch(time=time+1,
                                     weights=batched_weights)
                print("reward:", np.mean(reward))
            except StopIteration:
                epochs += 1
                print("Increased epoch")
                data_generator = self.data_container.yield_data(history_length=self.history_length,
                                                                batch_size=self.batch_size)

    def infer(self, test_batch_size):
        data_generator = self.data_container.yield_data(history_length=self.history_length,
                                                        batch_size=test_batch_size)
        batched_data, time = next(data_generator)
        previous_weights = self.PVM.read_batch(time=time,
                                               batch_size=test_batch_size)
        previous_weights = previous_weights[:, :, np.newaxis, np.newaxis]
        feed_dict = {self.input: batched_data['batch_current_prices'],
                     self.future_prices: batched_data['batch_future_prices'],
                     self.previous_weights: previous_weights,
                     self.is_training: True,
                     self.batch_size: test_batch_size}
        batched_weights, reward, loss = self.train_step(feed_dict=feed_dict)
        self.PVM.input_batch(time=time+1,
                             weights=batched_weights)