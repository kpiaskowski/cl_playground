import sys
from builtins import float
from random import shuffle, randint

import numpy as np
import tensorflow as tf
import clusterone


"""
Ultra simple project to test on clusterone, check how to run a simple NN regression model
inputs: vectors of 5 numbers, x1...x4
outputs: single scalar, calculated as follows: y = x1 + 2x2 +0x3 - 0.5x4
"""

local_logs_path = '/home/carlo/logs'
logs_path = clusterone.get_logs_path(local_logs_path)
print('LOG PATH (local or clusterone', logs_path)

# # params
# num_examples = 10000
# batch_size = 1
# num_batches = num_examples // batch_size
# epochs = 5
# lrate = 0.001
#
# # data creation
# x = np.random.rand(num_examples, 4)
# y = x[:, 0] + 2 * x[:, 1] - 0.5 * x[:, 3]
#
# # network
# X = tf.placeholder(tf.float32, [None, 4])
# Y = tf.placeholder(tf.float32, None)
#
# logits = tf.layers.dense(X, 1, activation=None)
#
# # loss
# loss = tf.losses.mean_squared_error(labels=Y, predictions=logits)
# tf.summary.scalar('loss', loss)
# merged = tf.summary.merge_all()
#
# # train op
# train_op = tf.train.AdamOptimizer(lrate).minimize(loss)
#
# with tf.Session() as sess:
#     train_writer = tf.summary.FileWriter('summaries/' + str(randint(0, 1000000)))
#     sess.run(tf.global_variables_initializer())
#
#     for e in range(epochs):
#         shuffle([x, y])
#         for b in range(num_batches):
#             batch_x = x[b * batch_size:(b + 1) * batch_size, :]
#             batch_y = y[b * batch_size:(b + 1) * batch_size]
#
#             _, cost, summary = sess.run([train_op, loss, merged], feed_dict={X: batch_x, Y: batch_y})
#             print(e, b, cost)
#
#     # checking results
#     x = np.random.rand(10, 4)
#     y = x[:, 0] + 2 * x[:, 1] - 0.5 * x[:, 3]
#     out = sess.run(logits, feed_dict={X: x})
#     print('Checking')
#     for i in zip(y, out):
#         print('GT: {}, predicted: {}'.format(i[0], i[1][0]))