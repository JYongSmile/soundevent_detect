#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 2:04
# @Author  : Barry_J
# @Email   : s.barry1994@foxmail.com
# @File    : noise_deep_graphs.py
# @Software: PyCharm

# 新程序：

# 新程序：

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from swallowsound.swallowsound_input_data import read_data_sets

import tensorflow as tf

dir = '/tmp/tensorflow/noise/input_data'


# 定义变量和卷积函数
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x5(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 5, 1],
                          strides=[1, 1, 5, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层+可视化
############################################################
def deepnn(x):
    with tf.name_scope('resh_img'):
        x_image = tf.reshape(x, [-1, 1, 250, 1], name='Reshape')

    # 第一层卷积层：
    with tf.name_scope('cov_layer1'):
        with tf.name_scope('conv1'):
            with tf.name_scope('W_conv1'):
                W_conv1 = weight_variable([1, 5, 1, 32])
                tf.summary.histogram('cov_layer1', W_conv1)
            with tf.name_scope('b_conv1'):
                b_conv1 = bias_variable([32])
                tf.summary.histogram('cov_layer1', b_conv1)
            with tf.name_scope('h_conv1'):
                h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
                tf.summary.histogram('cov_layer1', h_conv1)
        with tf.name_scope('pool_1'):
            h_pool1 = max_pool_1x5(h_conv1)
            tf.summary.histogram('cov_layer1', h_pool1)

    # 第二层卷积层：
    with tf.name_scope('cov_layer2'):
        with tf.name_scope('conv2'):
            with tf.name_scope('W_conv2'):
                W_conv2 = weight_variable([1, 5, 32, 64])
                tf.summary.histogram('cov_layer2', W_conv2)
            with tf.name_scope('b_conv2'):
                b_conv2 = bias_variable([64])
                tf.summary.histogram('cov_layer2', b_conv2)
            with tf.name_scope('h_conv2'):
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
                tf.summary.histogram('cov_layer2', h_conv2)
        with tf.name_scope('pool_2'):
            h_pool2 = max_pool_1x5(h_conv2)
            tf.summary.histogram('cov_layer2', h_pool2)

    # 第三层卷积层：
    with tf.name_scope('cov_layer3'):
        with tf.name_scope('conv3'):
            with tf.name_scope('W_conv3'):
                W_conv3 = weight_variable([1, 5, 64, 128])
                tf.summary.histogram('cov_layer3', W_conv3)
            with tf.name_scope('b_conv3'):
                b_conv3 = bias_variable([128])
                tf.summary.histogram('cov_layer3', b_conv3)
            with tf.name_scope('h_conv3'):
                h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
                tf.summary.histogram('cov_layer3', h_conv3)
        with tf.name_scope('pool_3'):
            h_pool3 = max_pool_1x5(h_conv3)
            tf.summary.histogram('cov_layer3', h_pool3)

    ############################################################
    #
    # 定义全连接层：
    # 第一层全连接层：
    with tf.name_scope('fc_layer1'):
        with tf.name_scope('fc1'):
            with tf.name_scope('W_fc1'):
                W_fc1 = weight_variable([1 * 2 * 128, 1024])
                tf.summary.histogram('fc_layer1', W_fc1)
            with tf.name_scope('b_fc1'):
                b_fc1 = bias_variable([1024])
                tf.summary.histogram('fc_layer1', b_fc1)
            with tf.name_scope('h_pool3_flat'):
                h_pool3_flat = tf.reshape(h_pool3, [-1, 1 * 2 * 128])
                tf.summary.histogram('fc_layer1', h_pool3_flat)
            with tf.name_scope('h_fc1'):
                h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
                tf.summary.histogram('fc_layer1', h_fc1)
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                tf.summary.histogram('fc_layer1', h_fc1_drop)

    # 第二层全连接层：
    with tf.name_scope('fc_layer2'):
        with tf.name_scope('fc2'):
            with tf.name_scope('W_fc2'):
                W_fc2 = weight_variable([1024, 2])
                tf.summary.histogram('fc_layer2', W_fc2)
            with tf.name_scope('b_fc2'):
                b_fc2 = bias_variable([2])
                tf.summary.histogram('fc_layer2', b_fc2)
            with tf.name_scope('y_conv'):
                y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                tf.summary.histogram('fc_layer2', y_conv)
                return y_conv, keep_prob



# Import data
num_classes = 2
swallowsound = read_data_sets(dir,
                              gzip_compress=False,
                              train_imgaes='train-images-idx3-ubyte',
                              train_labels='train-labels-idx1-ubyte',
                              test_imgaes='t10k-images-idx3-ubyte',
                              test_labels='t10k-labels-idx1-ubyte',
                              one_hot=True,
                              validation_size=50,
                              num_classes=num_classes,
                              MSB=True)

# Create the model
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 250], name='x_input')
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2], name='y_input')

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
# cross_entropy = tf.reduce_mean(cross_entropy)
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('loss',cross_entropy)
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# AdamOptimizer(1e-4)
# GradientDescentOptimizer(0.5)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy',accuracy)

sess = tf.Session()
merged =tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs1/', sess.graph)

sess.run(tf.global_variables_initializer())



print('训练开始！')
for i in range(10000):
    batch = swallowsound.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 50 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('输入第 %d 批batch, 此时的训练精度为：%g' % (i, train_accuracy))
        result = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        train_writer.add_summary(result, i)

print('........')
print('........')
print('........')
print('........')
print('测试开始！')

print('测试精度为： %g' % accuracy.eval(session=sess, feed_dict={x: swallowsound.test.images, y_: swallowsound.test.labels,
                                                           keep_prob: 0.5}))
print('测试结束！')

# saver=tf.train.Saver()
# with tf.Session()as sess:
#     sess.run(tf.global_variables_initializer())
#     save_path=saver.save(sess,"mynet/save_net.ckpt")
#     print("save to my path",save_path)


#D:\pycharm\swallowsound\swallowsound\logs
#tensorboard  --logdir=D:\pycharm\swallowsound\swallowsound\logs\