"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

from keras.models import Sequential
from keras import backend as k
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback
import collections
import numpy as np
k.set_learning_phase(0)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def neural_net(num_sensors, params, load=''):
    model = Sequential()

    # First layer.
    model.add(Dense(
        params[0], init='lecun_uniform', input_shape=(num_sensors,)
    ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second layer.
    model.add(Dense(params[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(3, init='lecun_uniform')) #!
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    if load:
        model.load_weights(load)

    return model

def compute_fisher(model, inputs, sess):
    output_tensor = model.output
    listOfVariables = model.trainable_weights
    # print(listOfVariables)
    gradients = k.gradients(output_tensor, listOfVariables)
    # trainingExample = np.random.random((1,8))
    # sess = tf.InteractiveSession()
    # sess.run(tf.initialize_all_variables())
    F = collections.defaultdict(int)
    for input_data  in inputs:
        evaluated_gradients = sess.run(gradients,feed_dict={model.input:input_data})
        for i,v in enumerate(listOfVariables):
            F[v] += np.square(evaluated_gradients[i])
    for i,v in enumerate(listOfVariables):
        F[v] /= len(inputs)

    return F

# def get_star_weights(model):
#     weights = {}
#     for layer in model.layers:
#         layer.get_weights()

def lstm_net(num_sensors, load=False):
    model = Sequential()
    model.add(LSTM(
        output_dim=512, input_dim=num_sensors, return_sequences=True
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=512, input_dim=512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=3, input_dim=512)) #!
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return model



# saved_model = 'saved-models_yellow/evaluatedPolicies/9-164-150-100-50000-100000.h5' # use the saved model to get the FE
# model = neural_net(self.num_states, [164, 150], saved_model)
# F = compute_fisher(model, )