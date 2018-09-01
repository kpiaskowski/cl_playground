
import tensorflow as tf
import os

dataset_path = '/media/carlo/My Files/DL Playground/cluster_one_dataset/test_dataset/train'
batch_size = 10
filenames = [os.path.join(dataset_path, name) for name in os.listdir(dataset_path)]


def load_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_scaled = image_decoded / 255
    return image_scaled

dataset = tf.data.Dataset.from_tensor_slices((tf.constant(filenames)))
dataset = dataset.shuffle(buffer_size=5000)
dataset = dataset.map(load_image, 8)
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.prefetch(batch_size)
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.repeat()

handle = tf.placeholder(tf.string, shape=[])
iter = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
images = iter.get_next()
iterator = dataset.make_one_shot_iterator()

with tf.Session() as sess:
    t_handle = sess.run(iterator.string_handle())
    sess.run(tf.global_variables_initializer())

    o = sess.run(images, feed_dict={handle: t_handle})
    print(o.shape)