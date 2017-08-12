import numpy as np
#import cPickle as pickle
import tensorflow as tf
#%matplotlib inline
import matplotlib.pyplot as plt
import math

#from modelAny import *

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

import gym
env = gym.make('CartPole-v0')

# hyperparameters
H = 8 # number of hidden layer neurons
learning_rate = 1e-2
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

model_bs = 3 # Batch size when learning from model
real_bs = 3 # Batch size when learning from real environment

# model initialization
D = 4 # input dimensionality

#Policy network
tf.reset_default_graph()
observations = tf.placeholder(tf.float32, [None,4] , name="input_x")
W1 = tf.get_variable("W1", shape=[4, H],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages) 
newGrads = tf.gradients(loss,tvars)
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

#Model network
mH = 256 #Model layer size
 
input_data = tf.placeholder(tf.float32, [None, 5])
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [mH, 50])
    softmax_b = tf.get_variable("softmax_b", [50])
     
previous_state = tf.placeholder(tf.float32, [None, 5], name = "previous_state") #Input?
W1M = tf.get_variable("W1M", shape = [5, mH], #4 state variables + 1 action
                                 initializer = tf.contrib.layers.xavier_initializer())
B1M = tf.Variable(tf.zeros([mH]), name = "B1M")
layer1M = tf.nn.relu(tf.matmul(previous_state, W1M) + B1M)
W2M = tf.get_variable("W2M", shape = [mH, mH],
                                 initializer = tf.contrib.layers.xavier_initializer())
B2M = tf.Variable(tf.zeros([mH]), name = "B2M")
layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M)
 
#Output weights
wO = tf.get_variable("wO", shape=[mH, 4],
            initializer=tf.contrib.layers.xavier_initializer())
wR = tf.get_variable("wR", shape=[mH, 1],
            initializer=tf.contrib.layers.xavier_initializer())
wD = tf.get_variable("wD", shape=[mH, 1],
            initializer=tf.contrib.layers.xavier_initializer())
 
#Output biases
bO = tf.Variable(tf.zeros([4]),name="bO")
bR = tf.Variable(tf.zeros([1]),name="bR")
bD = tf.Variable(tf.ones([1]),name="bD")
 
#Predicted state
predicted_observation = tf.matmul(layer2M,wO,name="predicted_observation") + bO
predicted_reward = tf.matmul(layer2M,wR,name="predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M,wD,name="predicted_done") + bD)
 
#Actual state
true_observation = tf.placeholder(tf.float32,[None,4],name="true_observation")
true_reward = tf.placeholder(tf.float32,[None,1],name="true_reward")
true_done = tf.placeholder(tf.float32,[None,1],name="true_done")
 
predicted_state = tf.concat([predicted_observation,predicted_reward,predicted_done], 1)
 
observation_loss = tf.square(true_observation - predicted_observation)
 
reward_loss = tf.square(true_reward - predicted_reward)
 
done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1-predicted_done, 1-true_done)
done_loss = -tf.log(done_loss) #Again, this looks like binary logistic regression
 
model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)
 
modelAdam = tf.train.AdamOptimizer(learning_rate = learning_rate)
updateModel = modelAdam.minimize(model_loss)


def reset_grad_buffer(gradient_buffer):
    for i, gradient in enumerate(gradient_buffer):
        gradient_buffer[i] = gradient * 0
    return gradient_buffer

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#This function uses our model to produce a new state when given a previous state and action
def step_model(session, xs, action): #xs are previous state
    to_feed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5]) #What the shit is this?
    prediction = session.run([predicted_state], feed_dict = { previous_state: to_feed })
    reward = prediction[0][:, 4] #Row 4 is reward
    observation = prediction[0][:, 0:4] #Rows 0, 1, 2, and 3 are observation
    observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4) #Clipping pieces of observation to min and max possible values
    observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
    done_p = np.clip(prediction[0][:, 5], 0, 1) #Row 5 is done. Clip to 0 and 100% probability
    if done_p > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False
    return observation, reward, done

#Training the policy and model

xs,drs,ys,ds = [],[],[],[] #Fuck you, but I guess this is state, reward, action, done
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
init = tf.global_variables_initializer()
batch_size = real_bs

draw_from_model = False #When set to True, will use model for observations
train_the_model = True #Whether to train the model
train_the_policy = False #Whether to train the policy
switch_point = 1 #FUUUUU #The point at which it switches between model and actual?

#Launch the graph
with tf.Session() as session:
    rendering = False
    session.run(init)
    observation = env.reset()
    x = observation
    gradient_buffer = session.run(tvars)
    grad_buffer = reset_grad_buffer(gradient_buffer)
    
    while episode_number <= 5000:
        #Start displaying the environment once performance is acceptably high
        if(reward_sum / batch_size > 150 and draw_from_model == False) or rendering == True:
            env.render()
            rendering = True
    
        x = np.reshape(observation, [1, 4]) #What was the original shape here?
        
        tfprob = session.run(probability, feed_dict = {observations: x}) #Given x, produce y (0 is left, 1 is right?)
        action = 1 if np.random.uniform() < tfprob else 0 #Randomly select action proportional to confidence
        
        #Record various intermediates (needed later for backprop)
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)
        
        #Step the model or real environment and get new measurements
        if draw_from_model == False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = step_model(session, xs, action)
            
        reward_sum += reward
        
        ds.append(done * 1) #Why are we multiplying this by one??
        drs.append(reward) #Record reward (has to be done after we call step() to get reward for previous action
    
        if done:
            
            if draw_from_model == False:
                real_episodes += 1
            episode_number += 1
            
            #Stack together all inputs, hidden states, action gradients, and reward for this episode
            epx = np.vstack(xs) #Inputs
            epy = np.vstack(ys) #Action gradients
            epr = np.vstack(drs) #Reward
            epd = np.vstack(ds) #Donedness
            #Where are these hidden states he's talking about??
            xs, drs, ys, ds = [], [], [], [] #Reset array memory
            
            if train_the_model == True:
                
                actions = np.array([np.abs(y - 1) for y in epy][:-1]) #Why is this being inverted
                state_prevs = epx[:-1,:] #Presumably cutting off the last action because we don't have a subsequent state and reward to backprop
                state_prevs = np.hstack([state_prevs, actions])
                state_nexts = epx[1:, :] #Ignore first episode
                rewards = np.array(epr[1:, :])
                dones = np.array(epd[1:, :])
                state_nexts_all = np.hstack([state_nexts, rewards, dones]) #s1, r, s1(d)
                #s, a, r, s1
                #prompt state, taken action, received reward, subsequent state
                
                feed_dict = {previous_state: state_prevs, true_observation: state_nexts, true_done: dones, true_reward: rewards }
                loss, p_state, _ = session.run([model_loss, predicted_state, updateModel], feed_dict)
                
            if train_the_policy == True:
                discounted_epr = discount_rewards(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr) #Advantage appears to be normalized differentiation from average reward
                t_grad = session.run(newGrads, feed_dict = {observations: epx, input_y: epy, advantages: discounted_epr})
                
                #If gradients become too large, end training process
                if np.sum(t_grad[0] == t_grad[0]) == 0:
                    print("gradients too large", np.sum(t_grad[0] == t_grad[0]))
                    break
                for i, gradient in enumerate(t_grad):
                    grad_buffer[i] += gradient
                
            if switch_point + batch_size == episode_number:
                switch_point = episode_number
                if train_the_policy == True:
                    session.run(updateGrads, feed_dict = {W1Grad: grad_buffer[0], W2Grad: grad_buffer[1]})
                    grad_buffer = reset_grad_buffer(gradient_buffer)
                    
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01 #Gradually increment total reward
                if draw_from_model == False:
                    print("World performance: Episode %f. Reward %f. Action: %f. Mean reward %f." % (real_episodes, reward_sum / real_bs, action, running_reward / real_bs))
                    if reward_sum / batch_size  > 200:
                        print("break: reward_sum / batch_size" + str(reward_sum) + ", " + str(batch_size))
                        break
                reward_sum = 0
                
                #Once the model has been trained on 100 episodes, we start alternating between training the policy
                #from the model and training the model form the real environment.
                if episode_number > 100:
                    draw_from_model = not draw_from_model
                    train_the_model = not train_the_model
                    train_the_policy = not train_the_policy
                    
            if draw_from_model == True:
                observation = np.random.uniform(-0.1, 0.1, [4]) #Generate reasonable starting point
                batch_size = model_bs
            else:
                observation = env.reset()
                batch_size = model_bs
                
                
print(real_episodes)