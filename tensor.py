import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create 100 random dots
# x_data = np.random.rand(100)
# y_data = x_data*0.1 + 0.2
#
# #init a serialize model
# b = tf.Variable(0.)
# k = tf.Variable(0.)
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
# with tf.Session() as sess:
#     sess.run(init)
#     for step in range(201):
#         sess.run(train)
#         if step%20 == 0:
#             print(step, sess.run([k, b]))

# above gradient descent algorithm



######## new
'''3.1非线性回归'''
# #创建200个点 均匀分布在-0.5到0.5之间
# x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis] # 200 hang  1lie
# noise = np.random.normal(0, 0.02, x_data.shape) #噪音，和x_data 形状一样
# y_data = np.square(x_data) + noise
#
# #define two placeholders
#
# x = tf.placeholder(tf.float32, [None, 1])  # 任意行 一列
# y = tf.placeholder(tf.float32, [None, 1])
#
# #define神经网络中间层
# Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
# biases_L1 = tf.Variable(tf.zeros([1, 10]))
# Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# L1 = tf.nn.tanh(Wx_plus_b_L1)
#
#
# #define the print
#
# Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
# biases_L2 = tf.Variable(tf.zeros([1, 1]))
# Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
# prediction = tf.nn.tanh(Wx_plus_b_L2)
#
# #daijiahanshu
# loss = tf.reduce_mean(tf.square(y-prediction))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for _ in range(2000):
#         sess.run(train_step, feed_dict={x:x_data, y:y_data})
#     #draw
#     prediction_value = sess.run(prediction, feed_dict={x:x_data})
#     #draw
#     plt.figure()
#     plt.scatter(x_data, y_data)
#     plt.plot(x_data, prediction_value, 'r-', lw=5)
#     plt.show()


'''3.2 shibie'''


