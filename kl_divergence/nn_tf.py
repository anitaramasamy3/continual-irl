import tensorflow as tf

# NUM_STATES = 12

class Policy_Network:

	def __init__(self, num_states, params, sess, load = None):
		
		self.keep_prob = 0.8
		self.learning_rate = 0.001
		with tf.variable_scope('Policy_Network', reuse = tf.AUTO_REUSE) as scope:
			self.weights1 = self.get_weights([num_states, params[0]], 'layer1_weights')
			self.biases1 = self.get_bias_variable([params[0]], 'layer1_biases')
			
			self.weights2 = self.get_weights([params[0], params[1]], 'layer2_weights')
			self.biases2 = self.get_bias_variable([params[1]], 'layer2_biases')
			
			self.weights3 = self.get_weights([params[1], 3], 'layer3_weights')
			self.biases3 = self.get_bias_variable([3], 'layer3_biases')
		self.output, self.loss, self.optimizer = self.create_graph(num_states)

		self. var_list = [self.weights1, self.biases1, self.weights2, self.biases2, self.weights3, self.biases3]
		
		
		init = tf.global_variables_initializer()
		
		self.sess = sess
		self.sess.run(init)
		self.saver = tf.train.Saver()
		if load != None:
			self.restore_model(load)

		self.fisher = []

	def get_weights(self,shape, name):
		initializer = tf.contrib.layers.xavier_initializer()
		weights = tf.get_variable(shape=shape,name=name,initializer=initializer)
		return weights

	def get_bias_variable(self,shape,name, constant=0.0):
	    return tf.get_variable(shape = shape, name=name, initializer = tf.constant_initializer(constant))

	def neural_network(self,state):
		
		hidden1 = tf.matmul(state, self.weights1) + self.biases1
		hidden1 = tf.nn.relu(hidden1)
		# hidden1 = tf.nn.dropout(hidden1, self.keep_prob)

		hidden2 = tf.matmul(hidden1, self.weights2) + self.biases2
		hidden2 = tf.nn.relu(hidden2)
		# hidden2 = tf.nn.dropout(hidden2, self.keep_prob)

		hidden3 = tf.matmul(hidden2, self.weights3) + self.biases3

		return hidden3


	def create_graph(self, num_states):
		self.X = tf.placeholder(tf.float32, [None, num_states])
		self.Y = tf.placeholder(tf.float32, [None,3])
		# with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
		output = self.neural_network(self.X)
		loss = tf.losses.mean_squared_error(output, self.Y)

		optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

		return output, loss, optimizer

	def train(self, input_vector, labels):
		self.keep_prob = 0.8
		
		actions, step_loss, _ = self.sess.run([self.output, self.loss, self.optimizer], feed_dict = {self.X: input_vector, self.Y: labels})
		# print(step_loss)
		return actions, step_loss

	def predict(self, input_vector):
		self.keep_prob = 1
		
		actions = self.sess.run([self.output], feed_dict = {self.X: input_vector})
		return actions

	def save_weights(self, model_path):
		self.saver.save(self.sess, model_path)

	def restore_model(self, load):
		self.saver.restore(self.sess, load)

	# def compute_fisher(self, input, output):

	# 	for v in range(len(self.var_list)):
 #            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))



# sess = tf.InteractiveSession()

# var = tf.train.list_variables('./saved-models_red/evaluatedPolicies/1-164-150-100-50000-1100.h5')

# print(var)
