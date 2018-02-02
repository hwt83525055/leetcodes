import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist = input("MNIST_data/")
max_steps = 1001
image_sum = 3000
Dir = "Data/"

sess = tf.Session()

embedding = tf.Variable(tf.stack(mnist.test.images[:image_sum]), trainable=False, name='embedding')

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

    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
    # -1  undefined
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

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

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    tf.summary.scalar(loss, 'loss')

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

if tf.gfile.Exists(Dir + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(Dir + 'projector/projector/metadata.tsv')
with open(Dir + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.arg_max(mnist.test.lables[:], 1))
    for _ in range(image_sum):
        f.write(str(labels[i]) + '\n')

merge = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(Dir + 'projector/projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = Dir + 'projector/projector/metadata.tsv'
embed.sprite.image_path = Dir + 'projector/data/mnist_10k_sprite.png'
