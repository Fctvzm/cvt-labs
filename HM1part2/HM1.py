import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import sys,argparse
import time
from datetime import timedelta
import math
import random


def upload_images(train_path, image_sizeW, image_sizeH, classes):
    images = []
    labels = []
    cls = []

    for name in classes:   
        index = classes.index(name)
        path = os.path.join(train_path, name, '*g')
        files = glob.glob(path)
        for f in files:
            image = cv2.imread(f)
            image = cv2.resize(image, (image_sizeW, image_sizeH), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            cls.append(name)
    images = np.array(images)
    labels = np.array(labels)
    cls = np.array(cls)

    return images, labels, cls


class DataSet(object):

    def __init__(self, images, labels, cls):
    	self._num_examples = images.shape[0]

    	self._images = images
    	self._labels = labels
    	self._cls = cls
    	self._epochs_done = 0
    	self._index_in_epoch = 0

    @property
    def images(self):
    	return self._images

    @property
    def labels(self):
    	return self._labels

    @property
    def cls(self):
    	return self._cls

    @property
    def num_examples(self):
    	return self._num_examples

    @property
    def epochs_done(self):
    	return self._epochs_done

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
          self._epochs_done += 1
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._cls[start:end]


def read_train_test(train_path, image_sizeW, image_sizeH, classes, test_size):
	class DataSets(object):
	   pass
	data_sets = DataSets()

	images, labels, cls = upload_images(train_path, image_sizeW, image_sizeH, classes)
	images, labels, cls = shuffle(images, labels, cls)  

	if isinstance(test_size, float):
		test_size = int(test_size * images.shape[0])

	test_image = images[:test_size]
	test_labels = labels[:test_size]
	test_cls = cls[:test_size]

	train_images = images[test_size:]
	train_labels = labels[test_size:]
	train_cls = cls[test_size:]

	data_sets.train = DataSet(train_images, train_labels, train_cls)
	data_sets.test = DataSet(test_image, test_labels, test_cls)

	return data_sets


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

batch_size = 64

classes = ['neg','pos']
num_classes = len(classes)

test_size = 0.2
img_sizeW = 128
img_sizeH = 128
num_channels = 3
train_path='dataset2'

data = read_train_test(train_path, img_sizeW, img_sizeH, classes, test_size = test_size)
print("Reading input data is completed")


session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_sizeH, img_sizeW, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


#filters
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 128

def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

def create_biases(size):
	return tf.Variable(tf.constant(0.05, shape = [size]))

def create_convolutional_layer(input, n_channels, filter_size, n_filters):  
    weights = create_weights(shape = [filter_size, filter_size, n_channels, n_filters])
    biases = create_biases(n_filters)

    layer = tf.nn.conv2d(input = input, filter = weights, strides = [1, 1, 1, 1], padding = 'SAME')
    layer += biases

    layer = tf.nn.max_pool(value = layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    layer = tf.nn.relu(layer)

    return layer

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer

def create_fullconnected_layer(input, n_inputs, n_outputs, use_relu = True):
    weights = create_weights(shape = [n_inputs, n_outputs])
    biases = create_biases(n_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
    	layer = tf.nn.relu(layer)
    
    return layer

#layers

layer_conv1 = create_convolutional_layer(x, num_channels, filter_size_conv1, num_filters_conv1)
layer_conv2 = create_convolutional_layer(layer_conv1, num_filters_conv1, filter_size_conv2, num_filters_conv2)
layer_conv3= create_convolutional_layer(layer_conv2, num_filters_conv2, filter_size_conv3, num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fullconnected_layer(layer_flat, layer_flat.get_shape()[1:4].num_elements(),
                fc_layer_size)

layer_fc2 = create_fullconnected_layer(layer_fc1, fc_layer_size, num_classes, use_relu = False) 


y_pred = tf.nn.softmax(layer_fc2, name = 'y_pred')
y_pred_cls = tf.argmax(y_pred, dimension = 1)

session.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc2, labels = y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



session.run(tf.global_variables_initializer())

def print_accuracy(epoch, train_dic, test_dic, loss):
    acc = session.run(accuracy, feed_dict = train_dic)
    val_acc = session.run(accuracy, feed_dict = test_dic)
    message = "Epoch {0} Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Loss: {3:.3f}"
    print(message.format(epoch + 1, acc, val_acc, loss))

total_iterations = 0

saver = tf.train.Saver(save_relative_paths = True)

def train(n_iteration):
    global total_iterations
    
    for i in range(total_iterations, total_iterations + n_iteration):
        x_batch, y_true_batch, cls_batch = data.train.next_batch(batch_size)
        x_test_batch, y_test_batch, y_cls_batch = data.test.next_batch(batch_size)

        
        train_dic = {x: x_batch, y_true: y_true_batch}
        test_dic = {x: x_test_batch, y_true: y_test_batch}

        session.run(optimizer, feed_dict = train_dic)

        if i % int(data.train.num_examples/batch_size) == 0: 
            loss = session.run(cost, feed_dict = test_dic)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            print_accuracy(epoch, train_dic, test_dic, loss)
            saver.save(session, './model.ckpt') 

    total_iterations += n_iteration

train(150)


