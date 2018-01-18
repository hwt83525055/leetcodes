import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# 定义批次的大小
batch_size = 200
#计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size

# define two placeholders
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10]) #label 0-9
keep_prob = tf.placeholder(tf.float32)
# keep_prob  baifenzhi  duoshao  shenjingyuan  gongzuo

#创建一个简单的神经网络
W = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b = tf.Variable(tf.zeros([2000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W)+b)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2)+b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3)+b3)
L3_drop = tf.nn.dropout(L3, keep_prob)


W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)


prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

#定义二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))


#gradient descent
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#init
init = tf.global_variables_initializer()

#求准确率     tf.argmax 求y最大的值的位置 ,匹配的程度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
#结果存放在一个布尔型列表中

#把预测的对比之后转换为32位浮点型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(200):
        for batch in range(n_batch):
            #huoqu yige pici
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print('test准确率为:', test_acc)
        print('train准确率:', train_acc)