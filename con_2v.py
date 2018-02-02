# import tensorflow as tf
# #case 2
# input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
# filter = tf.Variable(tf.random_normal([1, 1, 5, 1]))
#
# op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# #case 3
# input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
# filter = tf.Variable(tf.random_normal([3,3,5,1]))
#
# op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# #case 4
# input = tf.Variable(tf.random_normal([1,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,1]))
#
# op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# #case 5
# input = tf.Variable(tf.random_normal([1,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,1]))
#
# op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
# #case 6
# input = tf.Variable(tf.random_normal([1,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,7]))
#
# op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
# #case 7
# input = tf.Variable(tf.random_normal([1,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,7]))
#
# op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
# #case 8
# input = tf.Variable(tf.random_normal([10,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,7]))
#
# op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print("case 2")
#     print(sess.run(op2))
#     print("case 3")
#     print(sess.run(op3))
#     print("case 4")
#     print(sess.run(op4))
#     print("case 5")
#     print(sess.run(op5))
#     print("case 6")
#     print(sess.run(op6))
#     print("case 7")
#     print(sess.run(op7))
#     print("case 8")
#     print(sess.run(op8))


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#
# # 输入数据(图像)
# x_image = tf.placeholder(tf.float32, shape=[5, 5])
# x = tf.reshape(x_image, [1, 5, 5, 1])
# # filter
# W_cpu = np.array([[1, 1, 1], [0, -1, 0], [0, -1, 1]], dtype=np.float32)
# W = tf.Variable(W_cpu)
# W = tf.reshape(W, [3, 3, 1, 1])
#
# # 步长，卷积类型
# strides = [1, 1, 1, 1]
# padding = 'VALID'
#
# # 卷积
# # x的4个参数是[batch, in_height, in_width, in_channels]，代表[训练时图片的数量， 图片的高度，图片的宽度，图像通道数]
# # y的4个参数是[filter_heigth, filter_width, in_channels, out_channels ],代表[卷积核的高度，卷积核的宽度，图像通道数， 卷积核个数]
# # strides:卷积时每一维的步长，这是一个一维的向量，长度为4
# # padding':VALID表示without padding SAME with zero padding
# y = tf.nn.conv2d(x, W, strides, padding)
#
# x_data = np.array(
#     [
#         [1, 0, 0, 0, 0],
#         [2, 1, 1, 2, 1],
#         [1, 1, 2, 2, 0],
#         [2, 2, 1, 0, 0],
#         [2, 1, 2, 1, 1]
#     ]
# )
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     x = sess.run(x, feed_dict={x_image: x_data})
#     W = sess.run(W, feed_dict={x_image: x_data})
#     y = sess.run(y, feed_dict={x_image: x_data})
# print("The shape of X:", x.shape)
# print(x.reshape(5, 5))
# print("")
#
# print("The shape of W:", W.shape)
# print(W.reshape(3, 3))
# print('')
#
# print("The shape of y:", y.shape)
# print(y.reshape(3, 3))
# print("")
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')  # label 0-9

W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
b_conv1 = bias_variable([32], 'b_conv1')

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b_conv2 = bias_variable([64], 'b_conv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024], 'b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10], 'W_fc2')
b_fc2 = bias_variable([10], 'b_fc2')

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'my_saver/cnn1.ckpt')
    batch = mnist.test.next_batch(50)
    x_k = batch[0]
    y_k = batch[1]
    print(sess.run(accuracy, feed_dict={x: x_k, y_: y_k, keep_prob: 1.0}))
