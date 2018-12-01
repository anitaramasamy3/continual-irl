from nn_tf_with_fisher import Policy_Network
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
saved_model = 'saved-models_red/evaluatedPolicies/1-164-150-100-50000-1100.h5'
model = Policy_Network(12, [164, 150], sess, saved_model)


input_vector = [[0,1,0,1,1,1,0,0,1,0,0,0], [0,1,1,0,1,1,0,0,1,0,0,0]]
input_vector = np.array(input_vector)
output = np.array([[0,1,0], [1,0,0]])
model.compute_fisher(input_vector, output)
model.star_vars()
print(model.star_vars)