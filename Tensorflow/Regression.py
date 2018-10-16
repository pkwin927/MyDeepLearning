#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 05:20:36 2018

@author: makise
"""

import tensorflow as tf
#import tensorlayer as tl
import matplotlib.pyplot as plt
import numpy as np




tf.set_random_seed(1)
np.random.seed(1)

x = np.linspace(-2,2,1000)[:, np.newaxis]

noise = np.random.normal(0,0.1, size = x.shape)

y = np.power(x,3) + noise  # y = x^2 + noise

plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)

tf_y = tf.placeholder(tf.float32, y.shape)

layer1 = tf.layers.dense(tf_x, 10, tf.nn.relu)

layer2 = tf.layers.dense(layer1, 50, tf.nn.relu)

output = tf.layers.dense(layer2,1)

loss = tf.losses.mean_squared_error(tf_y, output)   
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)


sess = tf.Session()                                 
sess.run(tf.global_variables_initializer())         

plt.ion()   

for step in range(1000):

    i, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 50 == 0:

        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

sess.close()








