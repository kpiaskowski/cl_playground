import tensorflow as tf


# distributed
tasks = ["localhost:2222", "localhost:2223"]
jobs = {"local": tasks}
cluster = tf.train.ClusterSpec(jobs)
server1 = tf.train.Server(cluster, job_name="local", task_index=0)
server2 = tf.train.Server(cluster, job_name="local", task_index=1)
tf.reset_default_graph()

with tf.device("/job:local/task:0"):
    var1 = tf.Variable(0.0, name='var1')
with tf.device("/job:local/task:1"):
    var2 = tf.Variable(0.0, name='var2')

# (This will initialize both variables)

sess1 = tf.Session(server1.target)
sess2 = tf.Session(server2.target)

sess1.run(tf.global_variables_initializer())
# sess1.run(tf.global_variables_initializer())
# sess2.run(tf.global_variables_initializer())

print('DISTRIBUTED')
print("Initial value of var in session 1:", sess1.run(var2))
print("Initial value of var in session 2:", sess2.run(var2))


def run_with_location_trace(sess, op):
    # From https://stackoverflow.com/a/41525764/7832197
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(op, options=run_options, run_metadata=run_metadata)
    for device in run_metadata.step_stats.dev_stats:
      print(device.device)
      for node in device.node_stats:
        print("  ", node.node_name)

run_with_location_trace(sess2, var1)

