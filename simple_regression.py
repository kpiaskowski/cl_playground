import os
from random import shuffle

import clusterone
import numpy as np
import tensorflow as tf

"""
Ultra simple project to test on clusterone, check how to run a simple NN regression model
inputs: vectors of 5 numbers, x1...x4
outputs: single scalar, calculated as follows: y = x1 + 2x2 +0x3 - 0.5x4
"""

PATH_TO_LOCAL_LOGS = '/home/carlo/logs'
PATH_TO_LOCAL_DATA = '/media/carlo/My Files/DL Playground/cluster_one_dataset'

flags = tf.app.flags
flags.DEFINE_string("data_dir",
                    clusterone.get_data_path(
                        dataset_name="kpiaskowski/test_dataset",
                        local_root=PATH_TO_LOCAL_DATA,
                        local_repo="test_dataset",
                        path='train' # path: string, path inside the repository, e.g. train.
                    ), "Path to data. Returns <local_root>/<local_repo>/<path> or /data/<dataset_name>/<path>")
flags.DEFINE_string("log_dir", clusterone.get_logs_path(root=PATH_TO_LOCAL_LOGS), "Path to logs")

FLAGS = flags.FLAGS

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

    # checking results
    x = np.random.rand(10, 4)
    y = x[:, 0] + 2 * x[:, 1] - 0.5 * x[:, 3]
    out = sess.run(logits, feed_dict={X: x})
    print('Checking')
    for i in zip(y, out):
        print('GT: {}, predicted: {}'.format(i[0], i[1][0]))
