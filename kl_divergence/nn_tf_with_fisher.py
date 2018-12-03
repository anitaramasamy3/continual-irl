import tensorflow as tf
import numpy as np

# NUM_STATES = 12

class Policy_Network:

	def __init__(self, num_states, params, sess, name,load = None, prev_model=None, restore_prev_weights=False):
		
		self.keep_prob = 0.8
		self.learning_rate = 0.001
		with tf.variable_scope('Policy_Network'+name, reuse = tf.AUTO_REUSE) as scope:
			self.weights1 = self.get_weights([num_states, params[0]], 'layer1_weights'+name)
			self.biases1 = self.get_bias_variable([params[0]], 'layer1_biases'+name)
			
			self.weights2 = self.get_weights([params[0], params[1]], 'layer2_weights'+name)
			self.biases2 = self.get_bias_variable([params[1]], 'layer2_biases'+name)
			
			self.weights3 = self.get_weights([params[1], 3], 'layer3_weights'+name)
			self.biases3 = self.get_bias_variable([3], 'layer3_biases'+name)
		self.prev_model = prev_model
		self.lambda1 = 1
		self.lambda2 = 0.2
		self.var_list = [self.weights1, self.biases1, self.weights2, self.biases2, self.weights3, self.biases3]
		self.output, self.loss, self.optimizer = self.create_graph(num_states)

		# init = tf.global_variables_initializer()
		
		self.sess = sess
		# self.sess.run(init)
		# restore_vars = [ var for var in tf.global_variables() if 'Adam' not in var.name ]	
		self.saver = tf.train.Saver()
		if load!=None:
			self.saver.restore(self.sess,load)
		
		if restore_prev_weights:
			
			self.restore_model()
		self.F = []

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
		if self.prev_model != None:
			self.kl_term = self.calculate_kl_term(self.X, output)
		else:
			self.kl_term = tf.constant(0, dtype=tf.float32)
		self.ms_loss = tf.losses.mean_squared_error(output, self.Y) 
		loss = self.lambda1*self.ms_loss + self.lambda2*self.kl_term

		# self.gradients_ms = tf.gradients(ms_loss, self.var_list)
		# self.gradients_kl = tf.gradients(self.kl_term, self.var_list)
		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

		return output, loss, optimizer

	def train(self, input_vector, labels):
		self.keep_prob = 0.8 
		actions, step_loss, _, kl, ms = self.sess.run([self.output, self.loss, self.optimizer, self.kl_term, self.ms_loss], feed_dict = {self.X: input_vector, self.Y: labels})
		# print(step_loss, self.lambda2*kl, step_loss-self.lambda2*kl)
		# print(kl)
		# for v in range(len(self.var_list)):
		# 	print(np.mean(gkl[v],axis=0))
		# 	print(np.mean(gms[v], axis=0))
		# 	print('---------------------------------------')
		return actions, step_loss, kl, ms

	def predict(self, input_vector):
		self.keep_prob = 1
		
		actions = self.sess.run([self.output], feed_dict = {self.X: input_vector})
		return actions

	def save_weights(self, model_path):
		self.saver.save(self.sess, model_path)

	def restore_model(self):
		for v in range(len(self.var_list)):
			self.sess.run(self.var_list[v].assign(self.prev_model.var_list[v]))

	def calculate_kl_term(self, state, current_model_output):
		brown_model_output = self.prev_model.neural_network(state)
		brown_model_prob = tf.nn.softmax(brown_model_output)

		kl_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=brown_model_prob, logits = current_model_output))
		return kl_t


	# def compute_fisher(self, input_vector, labels, num_samples):
	# 	self.F_accum = []
	# 	for v in range(len(self.var_list)):
	# 		self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

	# 	for i in range(num_samples):
	# 		ders = self.sess.run(tf.gradients(self.loss, self.var_list), feed_dict = {self.X: input_vector[i:i+1], self.Y: labels[i:i+1]})

	# 		for v in range(len(self.F_accum)):
	# 			self.F_accum[v] += np.square(ders[v])
	# 		# print(ders[v].shape)
	# 	print('Fisher terms')
	# 	# print(len(self.F))
	# 	# print(self.F)
	# 	# for i in range(6):
	# 		# print(self.F[i].shape)
	# 	for v in range(len(self.F_accum)):
	# 		self.F_accum[v] /= num_samples

	# def star(self):

	# 	self.star_vars = []

	# 	for v in range(len(self.var_list)):
	# 		self.star_vars.append(self.var_list[v].eval())

	# def update_ewc_loss(self, lam):
 #        # elastic weight consolidation
 #        # lam is weighting for previous task(s) constraints
 #        # self.ewc_loss = self.loss
	# 	for v in range(len(self.var_list)):
	# 		self.loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
	# 	print('update_loss')
	# 	self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
	# 	# self.sess.run(tf.variables_initializer(self.optimizer.variables()))
	# 	# self.optimizer.minimize(self.loss)
        

# sess = tf.InteractiveSession()

# var = tf.train.list_variables('./saved-models_red/evaluatedPolicies/1-164-150-100-50000-1100.h5')

# print(var)
