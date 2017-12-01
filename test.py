import tensorflow as tf

a = tf.Variable([[1,2],[3,4]])
a = tf.layers.dense(a,3)
print a
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(a)
