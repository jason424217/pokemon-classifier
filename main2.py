#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we complete main part of final project main 2
"""
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tqdm import tqdm
from data_preprocess2 import train_dataset, validation_dataset, test_dataset, batch_size
# from model2 import pretrained_resnet18
from model2 import tf_model

# hyper-parameter
step_size = 1e-4
training_epochs = 50

def train(model, train_dataset, validation_dataset, test_dataset, step_size, epochs, freeze=True):
    epoch = 0
    optimizer = tf.keras.optimizers.Adam(learning_rate=step_size)
    test_loss = []
    test_acc = []
    prev_valid = -float('inf')
    while epoch < epochs:
        epoch += 1
        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []
        for x, y in tqdm(validation_dataset):
            output = model.forward(x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, output))
            prob = tf.nn.softmax(output)
            acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y, prob))
            valid_loss.append(loss.numpy())
            valid_acc.append(round(acc.numpy(), 2))
        # early stop
        '''
        if sum(valid_acc) / len(valid_acc) < prev_valid:
            # break
            continue
        else:
            prev_valid = sum(valid_acc) / len(valid_acc)
        '''
        for x, y in tqdm(train_dataset):
            with tf.GradientTape() as tape:
                output = model.forward(x)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, output))
                # predict = prob > 0.5
                # loss = tf.math.reduce_mean(loss_function(y, prob))
                train_loss.append(loss.numpy())
                prob = tf.nn.softmax(output)
                acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y, prob))
                train_acc.append(round(acc.numpy(), 2))
            gradients = tape.gradient(loss, model.trainable_variables)
            # only apply
            if freeze:
                n = len(gradients)
                optimizer.apply_gradients(zip(gradients[n-2:], model.trainable_variables[n-2:]))
            else:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print('Epoch', epoch, 'Training loss: %.6f, Training Accuracy: %.4f, '
                              'Validation loss: %.6f, Validation Accuracy: %.4f'
              % (sum(train_loss) / len(train_loss), sum(train_acc) / len(train_acc),
                 sum(valid_loss) / len(valid_loss),  sum(valid_acc) / len(valid_acc)))
    for x, y in tqdm(test_dataset):
        output = model.forward(x)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, output))
        prob = tf.nn.softmax(output)
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y, prob))
        test_loss.append(loss.numpy())
        test_acc.append(round(acc.numpy(), 2))
    print('Test loss: %.4f, Test Accuracy: %.4f' % (sum(test_loss) / len(test_loss), sum(test_acc) / len(test_acc)))

if __name__ == "__main__":
    train(tf_model, train_dataset, validation_dataset, test_dataset, 1e-4, 500, freeze=True)