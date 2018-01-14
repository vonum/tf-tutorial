import tensorflow as tf

def compute_graph(node):
  sess = tf.Session()
  result = sess.run(node)
  sess.close()

  return result

# define the data flow graph
a = tf.constant(5.0)
b = tf.constant(3.0)
c = tf.constant(7.0)

d = tf.add(a, c)
e = tf.divide(c, b)

f = tf.multiply(d, e)

# create a session and run the computation
result = compute_graph(f)
print("result = {}".format(result))

# Example 1
a = tf.constant(1.0)
b = tf.constant(2.0)

c = a + b
d = a * b

f = c + d
e = c * d

g = f / e

result = compute_graph(g)
print("result = {}".format(result))

# Example 2
a = tf.constant(1.0)
b = tf.constant(2.0)

c = a * b
d = tf.sin(c)
e = b / d

result = compute_graph(e)
print("result = {}".format(result))
