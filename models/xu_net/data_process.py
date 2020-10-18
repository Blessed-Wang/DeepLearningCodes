from glob import glob

import cv2
import numpy as np
import tensorflow as tf

from models.xu_net.XuNet import XuNet
from models.xu_net.hyperparameters import BATCH_SIZE, BUFFER_SIZE
from models.xu_net.utils import hpf

PATH = './data/BOSS/'

train_cover_img_paths = glob(PATH + 'train/cover/*.pgm')
train_stego_img_paths = glob(PATH + 'train/stego/*.pgm')
train_cover = [cv2.imread(file_path, 0) for file_path in train_cover_img_paths]
train_stego = [cv2.imread(file_path, 0) for file_path in train_stego_img_paths]
train_cover = np.array([cv2.resize(img, (512, 512)) for img in train_cover])
train_stego = np.array([cv2.resize(img, (512, 512)) for img in train_stego])

test_cover_img_paths = glob(PATH + 'test/cover/*.pgm')
test_stego_img_paths = glob(PATH + 'test/stego/*.pgm')
test_cover = [cv2.imread(file_path, 0) for file_path in test_cover_img_paths]
test_stego = [cv2.imread(file_path, 0) for file_path in test_stego_img_paths]
test_cover = np.array([cv2.resize(img, (512, 512)) for img in test_cover])
test_stego = np.array([cv2.resize(img, (512, 512)) for img in test_stego])
train_pos_label = np.ones(train_cover.shape[0], dtype=np.float32)
train_nag_label = np.zeros(train_stego.shape[0], dtype=np.float32)
X_train = np.concatenate([train_cover, train_stego], axis=0)
y_train = np.concatenate([train_pos_label, train_nag_label], axis=0)
print(X_train.shape, y_train.shape)
test_pos_label = np.ones(test_cover.shape[0], dtype=np.float32)
test_nag_label = np.zeros(test_stego.shape[0], dtype=np.float32)
X_test = np.concatenate([test_cover, test_stego], axis=0)
y_test = np.concatenate([test_pos_label, test_nag_label], axis=0)

X_train = np.array([hpf(img) for img in X_train])
X_test = np.array([hpf(img) for img in X_test])

X_train = X_train.astype(np.float32) / 255.
X_test = X_test.astype(np.float32) / 255.
y_train = tf.one_hot(y_train, 2)
y_test = tf.one_hot(y_test, 2)

print(y_train.shape, y_test.shape)