import os

import numpy as np
import tensorflow as tf


class DataProvider:
    def __init__(self, root_path, batch_size):
        self.batch_size = batch_size
        self.n_imgs = 2
        self.train_dirs = [os.path.join(root_path, 'train', name) for name in os.listdir(os.path.join(root_path, 'train'))]
        self.val_dirs = [os.path.join(root_path, 'val', name) for name in os.listdir(os.path.join(root_path, 'val'))]

    # todo
    def create_dataset(self, paths):
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths)))
        dataset = dataset.shuffle(20000)
        dataset = dataset.flat_map(lambda dir: tf.data.Dataset.list_files(dir + '/*.png').
                                   shuffle(30).
                                   map(self.load_images, num_parallel_calls=8).
                                   map(lambda img, filename: tuple(tf.py_func(self.decode_name, [img, filename], [tf.float32, tf.float32], stateful=False)),
                                       num_parallel_calls=8).
                                   apply(tf.contrib.data.batch_and_drop_remainder(self.n_imgs)))
        dataset = dataset.prefetch(self.batch_size)
        dataset = dataset.shuffle(150)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        dataset = dataset.repeat()
        return dataset

    def dataset(self):
        t_d = self.create_dataset(self.train_dirs)
        v_d = self.create_dataset(self.val_dirs)

        h = tf.placeholder(tf.string, shape=[])
        iter = tf.data.Iterator.from_string_handle(h, t_d.output_types, t_d.output_shapes)
        images, angles = iter.get_next()

        t_iter = t_d.make_one_shot_iterator()
        v_iter = v_d.make_one_shot_iterator()

        return h, t_iter, v_iter, images, angles

    def load_images(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_scaled = image_decoded / 255
        return image_scaled, filename

    def decode_name(self, img, filename):
        decoded = filename.decode()
        angles = np.float32(decoded.split('/')[-1].split('_')[:2])
        return img, angles