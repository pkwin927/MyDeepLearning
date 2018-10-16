#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 05:28:31 2018

@author: makise
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

n_data = np.ones((100, 2))
x0 = np.random.normal(5 * n_data,1)      # class0 x shape=(100, 2)
y0 = np.zeros(100)                      # class0 y shape=(100, 1)
x1 = np.random.normal(-5*n_data, 1)     # class1 x shape=(100, 2)
y1 = np.ones(100)                       # class1 y shape=(100, 1)
x2 = np.random.normal(0*n_data,1)
y2 = np.full((100,),2)
x = np.vstack((x0, x1, x2))  # shape (300, 3) + some noise
y = np.hstack((y0, y1 ,y2))  # shape (300, )

# plot data
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)

tf_y = tf.placeholder(tf.int32, y.shape)

layer1 = tf.layers.dense(tf_x, units = 10,activation= tf.nn.relu)

layer2 = tf.layers.dense(layer1, units = 100,activation= tf.nn.relu)

output = tf.layers.dense(layer2, 3)

loss = tf.losses.sparse_softmax_cross_entropy(labels = tf_y, logits = output)

#accuracy = tf.metrics.accuracy()

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf_y, predictions=tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)

sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess.run(init_op)

plt.ion()   # something about plotting
for step in range(1000):
    # train and net output
    
    i, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})
    if step % 50 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()















 