import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
#%matplotlib inline Unsupported in python 3? Should be unnecessary

try:
	xrange = xrange
except:
	xrange = range

environment = gym.make('CartPole-v0')

gamma = 0.99 #discount rate

def discount_rewards(reward):
	#print("reward", reward)
	#Take 1D float array of rewards and compute discounted reward
	discounted_reward = np.zeros_like(reward)
	running_add = 0
	for t in reversed(xrange(0, reward.size)):
		running_add = running_add * gamma + reward[t]
		discounted_reward[t] = running_add
	#print("discounted reward", discounted_reward)
	return discounted_reward

class Agent():
	def __init__(self, alpha, s_size, a_size, h_size): #state, action, hidden?
		#These lines establish the feed-forward part of the network
		#The agent produces the action
		self.state_in = tf.placeholder(shape = [None, s_size], dtype = tf.float32)
		hidden = slim.fully_connected(
			self.state_in,
			h_size,
			biases_initializer = None, activation_fn = tf.nn.relu
			)
		self.output = slim.fully_connected(
			hidden,
			a_size,
			activation_fn = tf.nn.softmax,
			biases_initializer = None
			)
		#print("output", self.output)
		self.chosen_action = tf.argmax(self.output, 1) #action to choose
		
		#The next six lines establish the training procedure.
		#We feed the reward and chosen action into the network
		#to compute the loss, and use it to update the network.
		self.reward_holder = tf.placeholder(shape = [None], dtype = tf.float32) #Holds time-indexed episode of discounted rewards
		self.action_holder = tf.placeholder(shape = [None], dtype = tf.int32) #Holds time-indexed episode of actions
		self.indexes = tf.range(0, 
			tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder #Within a flattened m x K dimensional matrix, the index responsible for each action taken
		self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes) #Outputs for all actions in the episode
		self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
		
		tvars = tf.trainable_variables() #Contains fully connected layer weights
		self.gradient_holders = [] #Create empty list
		#Add gradient placeholders for each fully connected layer
		for fc_layer_index, _ in enumerate(tvars): #_ is actual variable
			placeholder = tf.placeholder(tf.float32, name = str(fc_layer_index) + '_holder')
			self.gradient_holders.append(placeholder)
		
		self.gradients = tf.gradients(self.loss, tvars)
		
		optimizer = tf.train.AdamOptimizer(learning_rate = alpha)
		self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
		

tf.reset_default_graph()

agent = Agent(1e-2, 4, 2, 8) #Load the agent

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999 #Max length of episode (anything over 200 is unclamped, since that limit is built into the environment
update_frequency = 5 #How often to backpropagate

init = tf.global_variables_initializer()

with tf.Session() as session:
	writer = tf.summary.FileWriter("tf_logs", session.graph)
	
	session.run(init)
	
	total_reward = []
	total_length = []
	
	grad_buffer = session.run(tf.trainable_variables())
	for index, grad in enumerate(grad_buffer): #Zero out grad buffer
		grad_buffer[index] = grad * 0
	
	for i in range(total_episodes):
		s = environment.reset()
		running_reward = 0
		ep_history = []
		for j in range(max_ep):
			#Probabilistically pick and action given our network outputs
			a_dist = session.run(agent.output, feed_dict = {
				agent.state_in:[s]
				})
			a = np.random.choice(a_dist[0], p = a_dist[0]) #Choose action with probability proportional to output weight
			a = np.argmax(a_dist == a) #chosen action
			
			#environment.render()
			s1, r, d, _ = environment.step(a) #Get our reward for taking an action given a bandit
			ep_history.append([s, a, r, s1])
			s = s1
			running_reward += r
			if d == True: #If episode is completed
				#Update the network
				ep_history = np.array(ep_history)
				ep_history[:,2] = discount_rewards(ep_history[:,2]) #Set rewards to discounted rewards
				feed_dict = {
					agent.reward_holder: ep_history[:,2], #Total sequence of discounted rewards received in last episode
					agent.action_holder: ep_history[:,1], #Total sequence of actions taken in last episode
					agent.state_in: np.vstack(ep_history[:,0]) #Get all rows, state column
					}

				grads = session.run(agent.gradients, feed_dict = feed_dict) #Gradients for both layers, one for each weight
				#grad buffer aggregates gradients for multiple episodes
				for index, grad in enumerate(grads): #Index is FC layer number
					#print("assigning grad buffer index at ", index)
					grad_buffer[index] += grad
					
				if i % update_frequency == 0 and i != 0:
					feed_dict = dictionary = dict(zip(agent.gradient_holders, grad_buffer)) #Generate feed dict from gradient buffer
					_ = session.run(agent.update_batch, feed_dict = feed_dict)
					for index, grad in enumerate(grad_buffer): #Zero out grad buffer
						grad_buffer[index] = grad * 0
						
				total_reward.append(running_reward)
				total_length.append(j)
				break
			
		print("episode #" + str(i) + " reward: ", str(running_reward))
		#Update our running tally of scores
		if i % 100 == 0:
			print("Average episode reward as of episode #" + str(i) + ": " + str(np.mean(total_reward[-100:])))
			if np.mean(total_reward[-100:]) > 195:
				print("Cartpole solved!")
	
