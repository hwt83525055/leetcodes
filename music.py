import tensorflow as tf
import numpy as np


# # create 100 random dots
# x_data = np.random.rand(100)
# y_data = x_data*0.1 + 0.2
#
# #init a serialize model
# with tf.name_scope('layer'):
#     with tf.name_scope('bias'):
#         b = tf.Variable(0., name='b')
#         tf.summary.scalar('b', b)
#     with tf.name_scope('k'):
#         k = tf.Variable(0., name='k')
#         tf.summary.scalar('k', k)
# y = k*x_data + b
#
#
# #define a two times
# loss = tf.reduce_mean(tf.square(y_data-y))
# #define a tiduxiajiangfa
# optimizer = tf.train.GradientDescentOptimizer(0.2)
# # zuixiaohua daijiahanshu
# train = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
#
# merge = tf.summary.merge_all()
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter('logs_new/', sess.graph)
#     sess.run(init)
#     for step in range(201):
#         summary, _ = sess.run([merge, train])
#         writer.add_summary(summary, step)
#         if step%20 == 0:
#             print(step, sess.run([k, b]))


#
# def add_layer(inputs, in_size, out_size, activation_function=None):
#     Weights = tf.Variable(tf.random_normal(in_size, out_size))
#     biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
#     Wx_plus_b = tf.matmul(inputs, Weights) + biases
#     if not activation_function:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_function(Wx_plus_b)
#     return outputs
#
# # x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, None]
# x_data = tf.linspace(-1.0, 1.0, 300)
# # noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
# noise = tf.random_normal(x_data.shape, 0.0, 0.1, dtype=tf.float32)
# # y_data = np.square(x_data) - 0.5 + noise
# y_data = tf.square(x_data) - 0.5 + noise
#
# xs = tf.placeholder(tf.float32, [None, 1])
# ys = tf.placeholder(tf.float32, [None, 1])
# l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# prediction = add_layer(l1, 10, 1, activation_function=None)
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                      reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for _ in range(1000):
#         sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
#         print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))


