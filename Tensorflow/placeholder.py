#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 04:50:46 2018

@author: makise
"""

import tensorflow as tf

x1 = tf.placeholder(dtype = tf.float32,shape = None)

y1 = tf.placeholder(dtype = tf.float32, shape = None)

z1 = x1 + y1

x2 = tf.placeholder(dtype = tf.float32, shape = [2,1])

y2 = tf.placeholder(dtype = tf.float32, shape = [1,2])

z2 = tf.matmul(x2,y2)

sess = tf.Session()


## When only one operation to run
z1_value = sess.run(z1,feed_dict={x1:1,y1:2})

print(z1_value)

## When run together

z1_value ,z2_value = sess.run([z1,z2], feed_dict = {x1:1, y1:2, x2:[[2],[2]], y2:[[3],[4]]})

print(z1_value)
print(z2_value)

sess.close()