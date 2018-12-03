"""
Once a model is learned, use this to play it. that is run/exploit a policy to get the feature expectations of the policy
"""

from flat_game import carmunk
import numpy as np
# from nn import neural_net, compute_fisher
from nn_tf_with_fisher import Policy_Network
import sys
import time
import tensorflow as tf

NUM_STATES = 12
GAMMA = 0.9


def play(model, weights, sess=None):

    car_distance = 0
    game_state = carmunk.GameState(weights, [1,0,0,0])

    _, state, __ = game_state.frame_step((2))
    # state = state + [1,0,0,0]

    featureExpectations = np.zeros(len(weights))
    
    # Move.
    #time.sleep(15)
    while True:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(state)))
        # F = compute_fisher(model, [state], sess)
        # print(F)
        #print ("Action ", action)

        # Take action.
        immediateReward , state, readings = game_state.frame_step(action)
        # state = state + [1,0,0,0]
        #print ("immeditate reward:: ", immediateReward)
        #print ("readings :: ", readings)
        #start recording feature expectations only after 100 frames
        if car_distance > 100:
            featureExpectations += (GAMMA**(car_distance-101))*np.array(readings)
        #print ("Feature Expectations :: ", featureExpectations)
        # Tell us something.
        if car_distance % 2000 == 0:
            print("Current distance: %d frames." % car_distance)
            break


    return featureExpectations

if __name__ == "__main__": # ignore
    # BEHAVIOR = sys.argv[1]
    # ITERATION = sys.argv[2]
    # FRAME = sys.argv[3]
    # saved_model = 'saved-models_'+BEHAVIOR+'/evaluatedPolicies/'+str(ITERATION)+'-164-150-100-50000-'+str(FRAME)+'.h5'
    # saved_model = 'saved-models_red/evaluatedPolicies/2-164-150-100-50000-100000.h5'
    sess = tf.InteractiveSession()
    # sess.run(tf.initialize_all_variables())
    saved_model = 'saved-models_red/evaluatedPolicies/4-164-150-100-50000-100000.h5'
    # saved_model = 'saved-models_yellow/evaluatedPolicies/8-164-150-100-50000-100000.h5'
    brown_weights = [-0.26275824,  0.03635492,  0.09312051,  0.00469211, -0.18295909,  0.6987476, -0.59225824, -0.2201157 ]

    # weights = [-0.79380502 , 0.00704546 , 0.50866139 , 0.29466834, -0.07636144 , 0.09153848 ,-0.02632325 ,-0.09672041]
    # weights = [-0.06099233, -0.20316265, -0.1427778,  -0.16924885,  0.25280695, -0.0025343, 0.30678838, -0.86483369]
    model = Policy_Network(NUM_STATES, [164, 150], sess, 'red',saved_model,None, False)
    
    print (play(model, brown_weights))
