#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 05:04:00 2018

@author: makise
"""

import tensorflow as tf


## Method1
var = tf.Variable(0)

add_operation = tf.add(var,1) ## or add_operation = var + 1

update_operation = tf.assign(var,add_operation)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
    
for i in range(3):
    
    sess.run(update_operation)
    
    print(sess.run(var))

sess.close()


