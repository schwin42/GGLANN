import tensorflow as tf

node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + provides a shortcut for tf.add(a, b)

session = tf.Session()
#print("node3: ", node3)
#print("session.run(node3): ", session.run([node3]))

print(session.run(adder_node, {a: 3, b: 4.5}))
print(session.run(adder_node, {a: [1, 3], b: [2, 4]}))

#a, b = session.run(adder_node, {a: node1, b: node2})
#print(