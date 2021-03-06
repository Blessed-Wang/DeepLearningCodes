import os
import tensorflow as tf

_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
path_to_zip = tf.keras.utils.get_file('./facades.tar.gz', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
