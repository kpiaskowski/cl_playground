
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=True)

batch_size = 6

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=None)

net = tf.layers.dense(X, 128, tf.nn.relu)
net = tf.layers.dense(net, 128, tf.nn.relu)
net = tf.layers.dense(net, 10, None)

loss = tf.losses.softmax_cross_entropy(Y, net)
train_op = tf.train.AdamOptimizer().minimize(loss)
writer = tf.summary.FileWriter()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        images, labels = mnist.train.next_batch(batch_size)
        cost, _ = sess.run([loss, train_op], feed_dict={X: images, Y: labels})
        print(i, cost)

