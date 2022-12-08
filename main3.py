#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we complete main part of final project main 3
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tqdm import tqdm
from data_preprocess3 import train_dataset, validation_dataset, test_dataset, batch_size
from model2 import pretrained_resnet18

# hyper-parameter
step_size = 1e-4
training_epochs = 50

def train(model, train_dataset, validation_dataset, test_dataset, step_size, epochs, freeze=True):
    epoch = 0
    optimizer = tf.keras.optimizers.Adam(learning_rate=step_size)
    loss_function = tf.keras.losses.BinaryCrossentropy()
    test_loss = []
    test_err = []
    prev_valid = -float('inf')
    while epoch < epochs:
        epoch += 1
        train_loss = []
        train_err = []
        valid_loss = []
        valid_err = []
        for x, y in tqdm(validation_dataset):
            output = model.forward(x)
            prob = tf.math.sigmoid(output)
            loss = tf.math.reduce_mean(loss_function(y, output))
            predict = prob > 0.5
            '''
            predict = tf.Variable(tf.zeros([batch_size, 18]))
            # sort by probability
            predict_type = tf.argsort(prob, axis=1, direction='DESCENDING', stable=False, name=None)
            for idx in range(batch_size):
                # write first type
                predict[idx, predict_type[idx, 0]].assign(1)
                # check second type
                if prob[idx, predict_type[idx, 1]] > 0.5:
                    predict[idx, predict_type[idx, 1]].assign(1)
            '''
            error = tf.reduce_mean(tf.reduce_sum(tf.math.abs(tf.cast(predict, tf.float32) - tf.cast(tf.reshape(y, [-1, 18]), tf.float32)), axis=1))
            valid_loss.append(loss.numpy())
            valid_err.append(round(error.numpy(), 2))

        for x, y in tqdm(train_dataset):
            with tf.GradientTape() as tape:
                output = model.forward(x)
                prob = tf.math.sigmoid(output)
                loss = tf.math.reduce_mean(loss_function(y, output))
                predict = prob > 0.5
                '''
                predict = tf.Variable(tf.zeros([batch_size, 18]))
                predict_type = tf.argsort(prob, axis=1, direction='DESCENDING', stable=False, name=None)
                for idx in range(batch_size):
                    # write first type
                    predict[idx, predict_type[idx, 0]].assign(1)
                    # check second type
                    if prob[idx, predict_type[idx, 1]] > 0.5:
                        predict[idx, predict_type[idx, 1]].assign(1)
                '''
                error = tf.reduce_mean(tf.reduce_sum(tf.math.abs(tf.cast(predict, tf.float32) -
                                                                 tf.cast(tf.reshape(y, [-1, 18]), tf.float32)), axis=1))
                train_loss.append(loss.numpy())
                train_err.append(round(error.numpy(), 2))
            gradients = tape.gradient(loss, model.trainable_variables)
            # only apply
            if freeze:
                n = len(gradients)
                optimizer.apply_gradients(zip(gradients[n-2:], model.trainable_variables[n-2:]))
            else:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print('Epoch', epoch, 'Training loss: %.6f, Training Error: %.4f, '
                              'Validation loss: %.6f, Validation Error: %.4f'
              % (sum(train_loss) / len(train_loss), sum(train_err) / len(train_err),
                 sum(valid_loss) / len(valid_loss), sum(valid_err) / len(valid_err)))
    for x, y in tqdm(test_dataset):
        output = model.forward(x)
        prob = tf.math.sigmoid(output)
        loss = tf.math.reduce_mean(loss_function(y, output))
        predict = prob > 0.5
        '''
        predict = tf.Variable(tf.zeros([batch_size, 18]))
        predict_type = tf.argsort(prob, axis=1, direction='DESCENDING', stable=False, name=None)
        for idx in range(batch_size):
            # write first type
            predict[idx, predict_type[idx, 0]].assign(1)
            # check second type
            if prob[idx, predict_type[idx, 1]] > 0.5:
                predict[idx, predict_type[idx, 1]].assign(1)
        '''
        error = tf.reduce_mean(
            tf.reduce_sum(tf.math.abs(tf.cast(predict, tf.float32) - tf.cast(tf.reshape(y, [-1, 18]), tf.float32)),
                          axis=1))
        test_loss.append(loss.numpy())
        test_err.append(round(error.numpy(), 2))
    print('Test loss: %.4f, Test Error: %.4f' % (sum(test_loss) / len(test_loss), sum(test_err) / len(test_err)))

if __name__ == "__main__":
    train(pretrained_resnet18, train_dataset, validation_dataset, test_dataset, 8e-4, 100, freeze=False)