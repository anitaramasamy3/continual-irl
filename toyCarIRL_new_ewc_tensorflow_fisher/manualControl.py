"""
Manually control the agent to provide expert trajectories.
The main aim is to get the feature expectaitons respective to the expert trajectories manually given by the user
Use the arrow keys to move the agent around
Left arrow key: turn Left
right arrow key: turn right
up arrow key: dont turn, move forward
down arrow key: exit 

Also, always exit using down arrow key rather than Ctrl+C or your terminal will be tken over by curses
"""
from flat_game import carmunk
import numpy as np
from nn_tf_with_fisher import Policy_Network
import curses # for keypress
import tensorflow as tf
import numpy as np

NUM_STATES = 12
GAMMA = 0.9 # the discount factor for RL algorithm



def play(screen):
    sess = tf.InteractiveSession()
    saved_model = 'saved-models_brown/evaluatedPolicies/1-164-150-100-50000-100000.h5'
    model = Policy_Network(NUM_STATES, [164, 150], sess, saved_model)

    car_distance = 0
    weights = [-0.26275824,  0.03635492,  0.09312051,  0.00469211, -0.18295909,  0.6987476, -0.59225824, -0.2201157 ]  #brown
    # weights = [-0.06099233, -0.20316265, -0.1427778,  -0.16924885,  0.25280695, -0.0025343, 0.30678838, -0.86483369]
    # weights = [1, 1, 1, 1, 1, 1, 1, 1]# just some random weights, does not matter in calculation of the feature expectations
    game_state = carmunk.GameState(weights, [0,0,1,0])
    _, state, __ = game_state.frame_step((2))
    featureExpectations = np.zeros(len(weights))
    Prev = np.zeros(len(weights))
    replay = []
    while True:
        car_distance += 1
        event = screen.getch()

        if event == curses.KEY_LEFT:
            action = 1
        elif event == curses.KEY_RIGHT:
            action = 0
        elif event == curses.KEY_DOWN:
            break
        else:
            action = 2

        # Take action. 
        #start recording feature expectations only after 100 frames
        immediateReward , new_state, readings = game_state.frame_step(action)
        replay.append((state, action, immediateReward, new_state))
        state = new_state

        if car_distance > 100:
            featureExpectations += (GAMMA**(car_distance-101))*np.array(readings)
            
        
        # Tell us something.
        changePercentage = (np.linalg.norm(featureExpectations - Prev)*100.0)/np.linalg.norm(featureExpectations)

        print (car_distance)
        print ("percentage change in Feature expectation ::", changePercentage)
        Prev = np.array(featureExpectations)

        if car_distance % 300 == 0:
            break

    Xtrain, Ytrain = process_minibatch(replay, model)
    np.save('xtrain_brown.npy',Xtrain)
    np.save('ytrain_brown.npy',Ytrain)
    return featureExpectations

def process_minibatch(minibatch, model):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        old_qval = model.predict(old_state_m)
        # Get prediction on new state.
        newQ = model.predict(new_state_m)
        # Get our best move. I think?
        maxQ = np.max(newQ)
        y = np.zeros((1, 3)) #3
        y[:] = old_qval[:]
        # Check for terminal state.
        #if reward_m != -500:  # non-terminal state
            #update = (reward_m + (GAMMA * maxQ))
        #else:  # terminal state
            #update = reward_m
        if new_state_m[0][7] == 1:  #terminal state
            update = reward_m
        else:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(NUM_STATES,))
        y_train.append(y.reshape(3,)) #3

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train

if __name__ == "__main__":
    screen = curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    screen.keypad(1)
    screen.addstr("Play the game")
    result = play(screen)
    curses.endwin()
    print (result)
