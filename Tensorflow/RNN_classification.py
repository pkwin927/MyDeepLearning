#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:16:32 2018

@author: makise
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow.examples.tutorials.mnist import input_data


tf.set_random_seed(1)
np.random.seed(1)

## Hyper Parameters

Batch_size = 64
Time_step = 28
Input_size = 28
Learning_rate = 0.01

# data

mnist = input_data.read_data_sets('./mnist', one_hot = True)

test_x = mnist.test.images[:2000]

test_y = mnist.test.labels[:2000]


## Model

tf_x = tf.placeholder(tf.float32, [None, Time_step * Input_size])

image = tf.reshape(tf_x, [-1, Time_step, Input_size])

tf_y = tf.placeholder(tf.int32, [None, 10])

rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units = 64)

outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
        rnn_cell,
        image,
        initial_state = None,
        dtype = tf.float32,
        time_major = False,
        )

output = tf.layers.dense(outputs[:,-1,:], 10)


loss = tf.losses.softmax_cross_entropy(onehot_labels = tf_y, logits = output)

train_op = tf.train.AdamOptimizer(Learning_rate).minimize(loss)

accuracy = tf.metrics.accuracy(labels = tf.argmax(tf_y, axis = 1), predictions = tf.argmax(output, axis = 1),)[1]

sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess.run(init_op)

for step in range(1200):
    
    b_x, b_y = mnist.train.next_batch(Batch_size)
    
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    
    if step % 50 == 0:
        
        accuracy_ = sess.run(accuracy, {tf_x:test_x, tf_y:test_y})
        
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)












