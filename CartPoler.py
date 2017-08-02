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
		print("indexes: ", self.indexes)
		self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes) #Outputs for all actions in the episode
		self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
		
		tvars = tf.trainable_variables()
		self.gradient_holders = []
		for idx, var in enumerate(tvars):
			placeholder = tf.placeholder(tf.float32, name = str(idx) + '_holder')
			self.gradient_holders.append(placeholder)
		
		self.gradient_holders = tf.gradients(self.loss, tvars)
		
		optimizer = tf.train.AdamOptimizer(learning_rate = alpha)
		self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
		

tf.reset_default_graph()

agent = Agent(1e-2, 4, 2, 8) #Load the agent

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999 #Max length of episode
update_frequency = 5 #And this? Really? No comments??

init = tf.global_variables_initializer()

with tf.Session() as session:
	writer = tf.summary.FileWriter("tf_logs", session.graph)
	
	session.run(init)
	
	total_reward = []
	total_length = []
	
	grad_buffer = session.run(tf.trainable_variables())
	for ix, grad in enumerate(grad_buffer):
		grad_buffer[ix] = grad * 0
	
	for i in range(total_episodes):
		s = environment.reset()
		running_reward = 0
		ep_history = []
		for j in range(max_ep):
			#Probabilistically pick and action given our network outputs
			a_dist = session.run(agent.output, feed_dict = {
				agent.state_in:[s]
				})
			print("a_dist", a_dist)
			a = np.random.choice(a_dist[0], p = a_dist[0]) #Choose action with probability proportional to output weight
			a = np.argmax(a_dist == a) #chosen action
			
			s1, r, d, _ = environment.step(a) #Get our reward for taking an action given a bandit
			ep_history.append([s, a, r, s1])
			s = s1
			running_reward += r
			if d == True: #If episode is completed
				#Update the network
				#print("unarrayed", type(ep_history))
				ep_history = np.array(ep_history)
				#print("arrayed", type(ep_history))
				ep_history[:,2] = discount_rewards(ep_history[:,2]) #Set rewards to discounted rewards
				#print("unstacked: ", ep_history[:,0])
				#print("reward", ep_history[:,1])
				feed_dict = { 
					agent.reward_holder: ep_history[:,2], #Total sequence of discounted rewards received in last episode
					agent.action_holder: ep_history[:,1], #Total sequence of actions taken in last episode
					agent.state_in: np.vstack(ep_history[:,0]) #Get all rows, state column
					}
				#print("stacked: ", np.vstack(ep_history[:,0]))

				grads = session.run(agent.gradient_holders, feed_dict = feed_dict)

				for idx, grad in enumerate(grads):
					grad_buffer[idx] += grad
					
				if i % update_frequency == 0 and i != 0:
					feed_dict = dictionary = dict(zip(agent.gradient_holders, grad_buffer))
					_ = session.run(agent.update_batch, feed_dict = feed_dict)
					#print(agent.indexes.eval())
					#print(session.run(agent.indexes, feed_dict = feed_dict))
					for ix, grad in enumerate(grad_buffer):
						grad_buffer[ix] = grad * 0
						
				total_reward.append(running_reward)
				total_length.append(j)
				break
			
			
		print("output: ", session.run(agent.output, feed_dict = feed_dict))
		print("chosen action: ", session.run(agent.chosen_action, feed_dict = feed_dict))
		print("actions: ", session.run(agent.action_holder, feed_dict = feed_dict))
		print("indexes: ", session.run(agent.indexes, feed_dict = feed_dict))
		print("responsible outputs: ", session.run(agent.responsible_outputs, feed_dict = feed_dict))

		#Update our running tally of scores
		if i % 100 == 0:
			print("Average episode reward: " + str(np.mean(total_reward[-100:])))
	


#print("copasetic")