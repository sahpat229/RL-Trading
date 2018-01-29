import tensorflow as tf

class PolicyNetwork():
    def __init__(self, num_assets, history_length, batch_size, commission_rate, mode):
        self.num_assets = num_assets
        self.history_length = history_length
        self.batch_size = batch_size
        self.commission_rate = commission_rate
        if mode == 'EIEE':
            self.build_cnn()

    @classmethod
    def determine_size(input_size, kernel_width, kernel_padding=0, stride=1):
        return (input_size - kernel_width + 2*kernel_padding) / stride

    def build_cnn(self):
        # Batch size is the episode length
        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=[self.batch_size, self.num_assets, self.history_length, 3]) # X_t's
        self.previous_weights = tf.placeholder(dtype=tf.float32,
                                               shape=[self.batch_size, self.num_assets, 1, 1]) # w_(t-1)'s
        self.future_prices = tf.placeholder(dtype=tf.float32,
                                            shape=[self.batch_size, self.num_assets]) # y_(t+1)'s
        network = tf.layers.conv2d(inputs=self.input,
                                   filters=2,
                                   kernel_size=[1, 3],
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.nn.l2_regularizer(scale=5e-9),
                                   activation=tf.nn.relu)
        network = tf.layers.conv2d(inputs=network,
                                   filters=20,
                                   kernel_size=[1, PolicyNetwork.determine_size(input_size=self.history_length,
                                                                                kernel_width=3)],
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.nn.l2_regularizer(scale=5e-9),
                                   activation=tf.nn.relu)
        network = tf.concat(values=[network, self.previous_weights],
                            axis=3)
        network = tf.layers.conv2d(inputs=network,
                                   filters=1,
                                   kernel_size=[1, 1],
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.nn.l2_regularizer(scale=5e-9),
                                   activation=None)
        network = tf.squeeze(network) # remove width and channels dimensions, which are both of size 1
        cash_bias = tf.get_variable(name='cash_bias',
                                    shape=[1, 1],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
        cash_bias = tf.tile(input=cash_bias,
                            multiples=[self.batch_size, 1]) # replicate the cash bias across the batch
        network = tf.concat(values=[cash_bias, network],
                            axis=1) # shape is now [self.batch_size, num_assets+1]
        self.voting_scores = tf.nn.softmax(network)

        self.future_prices = tf.concat(values=[tf.ones([self.batch_size, 1]), self.future_prices],
                                       axis=1)

        future_weight_prime = (self.future_prices * self.voting_scores) / tf.reduce_sum(self.future_prices * self.voting_scores,
                                                                                        axis=1) # w_(t+1)'
        # shape: [batch_size, num_assets+1]

        # if we index future_weight_prime from t = 1 to N, and index our network's output weights
        # from t = 2 to N+1, we can overcome the problem of needing to run the portfolio model an extra time

        mu = 1 - tf.reduce_sum(tf.abs(future_weight_prime[:, self.batch_size-1] - self.voting_scores[1, :]),
                               axis=1)*self.commission_rate # these are the mu_(t+1)'s per time step
        # size: [self.batch_size-1, 1]
        mu = tf.concat([tf.ones(1), mu],
                       axis=0)
        # size: [self.batch_size, 1]
        reward = tf.log(mu * tf.reduce_sum(self.future_prices * self.voting_scores,
                                           axis=1))
        # size: [self.batch_size, 1]

        self.loss = -tf.reduce_mean(reward) + tf.losses.get_regularization_loss()

    def train_init(self):
        model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, var_list=model_variables)
        self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

    def train(self):
        