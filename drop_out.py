import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 定义批次的大小
batch_size = 50
#计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('min', tf.reduce_max(var))
        tf.summary.scalar('max', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


with tf.name_scope('input'):
# define two placeholders
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input') #label 0-9
    keep_prob = tf.placeholder(tf.float32)
# keep_prob  baifenzhi  duoshao  shenjingyuan  gongzuo

#创建一个简单的神经网络
with tf.name_scope('layer'):
    with tf.name_scope('weight'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('bisas'):
        b = tf.Variable(tf.zeros([10]) + 0.1, name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)
# L1 = tf.nn.tanh(tf.matmul(x, W) + b)
# L1_drop = tf.nn.dropout(L1, keep_prob)
#
# W1 = tf.Variable(tf.truncated_normal([1000, 500], stddev=0.1))
# b1 = tf.Variable(tf.truncated_normal([500]) + 0.1)
# L2 = tf.nn.tanh(tf.matmul(L1_drop, W1) + b1)
# L2_drop = tf.nn.dropout(L2, keep_prob)
#
# W3 = tf.Variable(tf.truncated_normal([500, 200], stddev=0.1))
# b3 = tf.Variable(tf.truncated_normal([200]) + 0.1)
# L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
# L3_drop = tf.nn.dropout(L3, keep_prob)

# W4 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
# b4 = tf.Variable(tf.truncated_normal([10]) + 0.1)



#定义二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

#gradient descent
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

#init
init = tf.global_variables_initializer()

#求准确率     tf.argmax 求y最大的值的位置 ,匹配的程度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
#结果存放在一个布尔型列表中


merge = tf.summary.merge_all()

#把预测的对比之后转换为32位浮点型
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(200):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merge, train_step], feed_dict={x:batch_xs, y:batch_ys, keep_prob:1})
        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1})
        # acc_train = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:1})
        print('准确率为:', acc)




