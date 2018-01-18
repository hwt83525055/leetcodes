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

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

#定义二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))

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
    for epoch in range(21):
        for batch in range(n_batch):
            #huoqu yige pici
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('准确率为%f:', acc)