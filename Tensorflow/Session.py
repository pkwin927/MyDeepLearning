#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 04:28:42 2018

@author: makise
"""

import tensorflow as tf
import tensorlayer as tl

m1 = tf.constant([[2,2]])

m2 = tf.constant([[3],[3]])


operation = tf.matmul(m1,m2)
operation1 = tf.multiply(m1,m2)

print(operation)


## method1 use session

sess = tf.Session()

result = sess.run(operation)

result1 = sess.run(operation1)

print(result)
print(result1)

sess.close()

##method2  use session

with tf.Session() as sess:
    
    result = sess.run(operation)

    result1 = sess.run(operation1)
    
    print(result)
    print(result1)


