import tensorflow as tf

node1 = tf.constant(3.0, dtype = tf.float32) #Initialized when called
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + provides a shortcut for tf.add(a, b)
add_and_triple = adder_node * 3.

W = tf.Variable([.3], dtype = tf.float32) #Instantiate a new instance of the variable class
b = tf.Variable([-.3], dtype = tf.float32) #Only initalized once tf.global_variables_initializer() is called
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b
square_error = tf.square(linear_model - y)
loss = tf.reduce_sum(square_error)
optimizer = tf.train.GradientDescentOptimizer(0.01) #Param is learning rate
train = optimizer.minimize(loss)

init = tf.global_variables_initializer() #init is a handle to the global variables sub-graph
session = tf.Session()
session.run(init)

#print("node3: ", node3)
#print("session.run(node3): ", session.run([node3]))

#print(session.run(adder_node, {a: 3, b: 4.5}))
#print(session.run(adder_node, {a: [1, 3], b: [2, 4]}))

#print(session.run(add_and_triple, {a: 3, b: 4.5}))

#fix_W = tf.assign(W, [-1.])
#fix_b = tf.assign(b, [1.])
#session.run([fix_W, fix_b])
#print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

episodes = 1000

for i in range(episodes):
	session.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(session.run([W, b]))


#Multistep session run attempt
#a, b = session.run(adder_node, {a: node1, b: node2})
#print(