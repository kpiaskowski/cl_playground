import argparse
import sys

import numpy as np
import tensorflow as tf
FLAGS = None

# dataset_path = '/media/carlo/My Files/DL Playground/cluster_one_dataset/test_dataset/train'
batch_size = 10
# filenames = [os.path.join(dataset_path, name) for name in os.listdir(dataset_path)]

# ten trainer dziala lokalnie, minimalizuje losowa cos


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model...
            X = tf.placeholder(tf.float32, [None, 4])
            Y = tf.placeholder(tf.float32, None)
            logits = tf.layers.dense(X, 1, activation=None)

            loss = tf.losses.mean_squared_error(labels=Y, predictions=logits)
            tf.summary.scalar('loss', loss)

            global_step = tf.contrib.framework.get_or_create_global_step()

            train_op = tf.train.AdamOptimizer(0.01).minimize(
                loss, global_step=global_step)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.

        # writer_wr = tf.summary.FileWriter()

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir=None, #"/tmp/train_logs", todo ten probuje wczytac zapisane modele
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # t_handle = mon_sess.run(iterator.string_handle())
                # Run a training step asynchronously.
                # See <a href="../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.

                # data generation
                batch_x = np.random.rand(batch_size, 4)
                batch_y = batch_x[:, 0] + 2 * batch_x[:, 1] - 0.5 * batch_x[:, 3]
                cost, _ = mon_sess.run([loss, train_op], feed_dict={X: batch_x, Y: batch_y})
                print(cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
