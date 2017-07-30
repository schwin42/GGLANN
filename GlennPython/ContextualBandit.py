import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class ContextualBandit():
	def __init__(self):
		self.state = 0
		#List out our bandits. Currently, arms 4, 2, and 1 (respectively) are the most optimal)
		self.bandits = np.array([[0.2, 0, 0, -5], [0.1, -5, 1, -.25], [-5, 5, 5, 5]])
		self.num_bandits = self.bandits.shape[0] #number of top level arrays
		self.num_arms = self.bandits.shape[1] #number of next level elements

	def get_bandit(self):
		self.state = np.random.randint(0, len(self.bandits)) #Returns a random state for each episode
		return self.state

	def pull_arm(self, action):
		arm_threshhold = self.bandits[self.state, action] #number need to beat for reward
		#Get a random number
		result = np.random.randn(1)
		if result > arm_threshhold:
			#return a positive reward
			return 1
		else:
			#return a negative reward
			return -1

class Agent():
	def __init__(self, lr, s_size, a_size):
		#These lines established the feed-forward part of the network.
		#The agent takes a state and produces an action.
		self.state_in = tf.placeholder(shape = [1], dtype = tf.int32) #Environment state input
		state_in_OH = slim.one_hot_encoding(self.state_in, s_size)
		output = slim.fully_connected(state_in_OH, a_size, #\ #Is backslash typo??
			biases_initializer = None, activation_fn = tf.nn.sigmoid, 
			weights_initializer = tf.ones_initializer())
		self.output = tf.reshape(output, [-1])
		self.chosen_action = tf.argmax(self.output, 0)
		
		#The next six lines establish the training procedure.
		#We feed the reward and chosen action into the network to
		#compute the loss, and use it to update the network.
		self.reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)
		self.action_holder = tf.placeholder(shape = [1], dtype = tf.int32)
		self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
		self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
		self.update = optimizer.minimize(self.loss)

tf.reset_default_graph() #Clear the tensorflow graph

contextual_bandit = ContextualBandit() #Load the bandits
agent = Agent(lr = 0.001, s_size = contextual_bandit.num_bandits, 
			  a_size = contextual_bandit.num_arms) #Load the agent
weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network

total_episodes = 10000 #Set total episodes to train agent on
total_reward = np.zeros([contextual_bandit.num_bandits, contextual_bandit.num_arms]) #Set scoreboard for bandits to zero
e = 0.1 #Exploration rate (chance of taking random action)

init = tf.global_variables_initializer()

#Launch the tensorflow graph
with tf.Session() as session:
	session.run(init)
	for i in range(total_episodes):
		state = contextual_bandit.get_bandit() #Get a state from the environment

		#Choose either a random action or one from our network
		if np.random.rand(1) < e:
			action = np.random.randint(contextual_bandit.num_arms)
		else:
			action = session.run(agent.chosen_action, feed_dict = {
				agent.state_in: [state]})

		reward = contextual_bandit.pull_arm(action) #Get our reward for taking an action given a bandit

		#Update the network
		feed_dict = {
			agent.reward_holder: [reward], 
			agent.action_holder: [action],
			agent.state_in: [state]
			}

		_, ww = session.run([agent.update, weights], feed_dict = feed_dict)

		#Update our running tally of scores
		total_reward[state, action] += reward
		if i % 500 == 0:
			print ("Mean reward for each of the " + str(contextual_bandit.num_bandits) + " bandits: " + str(np.mean(total_reward,axis=1)))

for a in range(contextual_bandit.num_bandits):
	print ("The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most promising....")
	if np.argmax(ww[a]) == np.argmin(contextual_bandit.bandits[a]):
		print("...and it was right!")
	else:
		print("...and it was wrong!")
