#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we build model 1 of training data
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from data_preprocess1 import train_dataset, validation_dataset, test_dataset
import tensorflow as tf
import numpy as np

def f(x, w):
    """
    :param x: input data
    :param w: parameter
    :return: predict tensor
    """
    return tf.nn.softmax(tf.matmul(x, w))

def compute_loss(true, pred):
    """
    :param true: true labels
    :param pred: output tensor
    :return: loss
    """
    return tf.reduce_mean(tf.reduce_sum(tf.keras.metrics.categorical_crossentropy(true, pred), axis=-1))

# compute accuracy
# we use the function tf.keras.metrics.categorical_accuracy() to compute the accuracy
# [1, 0, 0] -> 0
# [0.2, 0.7, 0.1] -> 0.7 -> 1
# acc: 0

def compute_accuracy(true, pred):
    """
    :param true: true labels
    :param pred: output tensor
    :return: accuracy
    """
    return tf.reduce_mean(tf.keras.metrics.categorical_accuracy(true, pred))

# we define a function:
def check(W):
    """
    :param W: the parameter of the logistic regression
    :return: training loss; training accuracy; testing loss; testing accuracy
    """
    loss = []
    acc = []
    for x, y in train_dataset:
        output = f(x, W)
        loss.append(compute_loss(y, output))
        acc.append(compute_accuracy(y, output))
    train_loss = sum(loss) / len(loss)
    train_acc = sum(acc) / len(acc)
    # we can use zip function to separate data and labels
    test_x, test_y = zip(*test_dataset)  # tf.float32 -> text_x; [10000, 784]

    output = f(test_x, W)
    test_loss = compute_loss(test_y, output)
    test_acc = compute_accuracy(test_y, output)
    # tensor.numpy() can convert the tensor to numpy format
    return train_loss.numpy(), train_acc.numpy(), test_loss.numpy(), test_acc.numpy()


def logistic():
    epoch = 0
    # define the parameter and convert to tensor
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-2)
    W = tf.Variable(tf.convert_to_tensor(np.random.normal(size=(32*32*3, 18)), dtype=tf.float32))
    # set the loop conditions
    while epoch < 1000:
        epoch += 1
        # training
        for x, y in train_dataset: # go through all the training data
            # we come to the most important part, using tf.GradientTape
            # tensorflow will compute the gradient of the parameter by tf.GradientTape
            with tf.GradientTape() as tape:
                # ask the tape to keep track on W
                tape.watch(W)
                # go forward we get the output
                output = tf.nn.softmax(tf.matmul(x, W))
                # compute training loss
                loss = compute_loss(y, output) # scalar
            # ask for the gradient
            # tape.gradient(target=loss, source=parameter)
            grads = tape.gradient(target=loss, sources=W)
            # optimize parameter
            optimizer.apply_gradients(zip([grads], [W]))
        # every 5 epochs, we compute the training loss on the whole training data and the testing loss
        if epoch % 5 == 1:
            train_loss, train_acc, test_loss, test_acc = check(W)
            # print the results
            print('epoch: %d, training loss: %.2f, training accuracy: %.2f, testing loss: %.2f, testing accuracy: %.2f'
                  % (epoch, train_loss / 100, float(train_acc), test_loss / 796, float(test_acc)))


if __name__ == "__main__":
    logistic()