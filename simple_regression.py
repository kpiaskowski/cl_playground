import os
from random import shuffle, randint

import clusterone
import numpy as np
import tensorflow as tf

"""
Ultra simple project to test on clusterone, check how to run a simple NN regression model
inputs: vectors of 5 numbers, x1...x4
outputs: single scalar, calculated as follows: y = x1 + 2x2 +0x3 - 0.5x4
"""

PATH_TO_LOCAL_LOGS = '/home/carlo/logs'

flags = tf.app.flags
flags.DEFINE_string("log_dir",
                    clusterone.get_logs_path(root=PATH_TO_LOCAL_LOGS),
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to clusterone without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")





FLAGS = flags.FLAGS
FLAGS.log_dir = '/tblogs' if FLAGS.log_dir == '/logs/' else FLAGS.log_dir
import os
print('making flags', FLAGS.log_dir)
os.mkdir(FLAGS.log_dir)

# params
num_examples = 100
batch_size = 1
num_batches = num_examples // batch_size
epochs = 5
lrate = 0.001

# data creation
x = np.random.rand(num_examples, 4)
y = x[:, 0] + 2 * x[:, 1] - 0.5 * x[:, 3]

# network
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, None)

logits = tf.layers.dense(X, 1, activation=None)

# loss
loss = tf.losses.mean_squared_error(labels=Y, predictions=logits)
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

# train op
train_op = tf.train.AdamOptimizer(lrate).minimize(loss)
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(FLAGS.log_dir)
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        shuffle([x, y])
        for b in range(num_batches):
            batch_x = x[b * batch_size:(b + 1) * batch_size, :]
            batch_y = y[b * batch_size:(b + 1) * batch_size]

            _, cost, summary = sess.run([train_op, loss, merged], feed_dict={X: batch_x, Y: batch_y})
            print(e, b, cost)
            train_writer.add_summary(summary, global_step=e * num_batches + b)

        try:
            print('standard logs', os.listdir(FLAGS.log_dir))
            print('tb logs', os.listdir('/tblogs/'))
        except Exception as e:
            print(e)

    # checking results
    x = np.random.rand(10, 4)
    y = x[:, 0] + 2 * x[:, 1] - 0.5 * x[:, 3]
    out = sess.run(logits, feed_dict={X: x})
    print('Checking')
    for i in zip(y, out):
        print('GT: {}, predicted: {}'.format(i[0], i[1][0]))
