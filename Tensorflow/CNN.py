#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 04:36:08 2018

@author: makise
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()
tf.set_random_seed(1)
np.random.seed(1)


BATCH_SIZE = 50
LR = 0.001     
## Prepare Data

mnist = input_data.read_data_sets('./mnist', one_hot = True)

test_x = mnist.test.images[:20]

test_y = mnist.test.labels[:20]

## plot example

print(mnist.train.images.shape)
print(mnist.train.labels.shape)

plt.imshow(mnist.train.images[5].reshape((28,28)),cmap='gray')

plt.title('%i' % np.argmax(mnist.train.labels[0]))

plt.show()

## Tensorflow Graph

tf_x = tf.placeholder(tf.float32, [None, 28*28]) /255.

image = tf.reshape(tf_x, [-1,28,28,1])

tf_y = tf.placeholder(tf.int32, [None, 10])

conv1 = tf.layers.conv2d(inputs = image,
                          filters = 16,
                          kernel_size = 5,
                          strides = 1,
                          padding = 'same',
                          activation = tf.nn.relu
                          )

pool1 = tf.layers.max_pooling2d(inputs = conv1,
                                pool_size = 2,
                                strides = 2)

conv2 = tf.layers.conv2d(inputs = pool1,
                         filters = 32,
                         kernel_size = 5,
                         strides = 1,
                         padding = 'same',
                         activation = tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs = conv2,
                                pool_size = 2,
                                strides = 2)

flat = tf.reshape(pool2,[-1,7*7*32])

output = tf.layers.dense(flat,10)

loss = tf.losses.softmax_cross_entropy(onehot_labels = tf_y, logits = output)

train_op = tf.train.AdamOptimizer(learning_rate = LR).minimize(loss)

accuracy = tf.metrics.accuracy(labels = tf.argmax(tf_y,axis = 1), predictions = tf.argmax(output, axis = 1),)[1]


sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 

sess.run(init_op)

for step in range(600):
    
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    
    _, loss_ = sess.run([train_op, loss], {tf_x:b_x, tf_y:b_y})
    
    if step % 50 == 0:
        
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)





