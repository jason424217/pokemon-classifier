#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we complete model2 on final project data
"""
import torchvision.models as models
import tensorflow as tf

class input_block(tf.keras.Model):
    def __init__(self, seed=1):
        super(input_block, self).__init__()
        # use random seed to make the initialization repeat
        tf.random.set_seed(seed)
        # define convolutional layers
        self._name = ''
        self.c1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='VALID', activation=None,
                                         use_bias=False, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.relu = tf.keras.layers.Activation('relu', name='relu')
        self.maxpool = tf.keras.layers.MaxPool2D(3, strides=2, padding='VALID', name='maxpool')

    def forward(self, input):
        # batch_size x frame x 224 x 224 x 3
        x = self.c1(input)
        # batch_size x frame x 112 x 112 x 64
        x = self.bn1(x)
        x = self.relu(x)
        output = self.maxpool(x)
        # batch_size x frame x 56 x 56 x 64
        return output

class basic_block(tf.keras.Model):
    def __init__(self, planes, stride=1, seed=1, name='layer1'):
        super(basic_block, self).__init__()
        # use random seed to make the initialization repeat
        tf.random.set_seed(seed)
        # define convolutional layers
        self._name = name
        self.stride = stride
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=self.stride, padding='SAME', activation=None,
                                            use_bias=False, name=name + '.conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name=name + '.bn1')
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, padding='SAME', activation=None,
                                            use_bias=False, name=name + '.conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(name=name + '.bn2')
        self.relu = tf.keras.layers.Activation('relu', name=name + '.relu')
        if self.stride != 1:
            self.short_cut = tf.keras.layers.Conv2D(planes, kernel_size=1, strides=self.stride, padding='SAME',
                                                    activation=None, use_bias=False, name=name + '.downsample.0')
            self.bn_short = tf.keras.layers.BatchNormalization(name=name + '.downsample.1')

    def forward(self, input):
        # batch_size x frame x H x W x planes
        x_1 = self.conv1(input)
        x_1 = self.bn1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.conv2(x_1)
        x_1 = self.bn2(x_1)
        if self.stride != 1:
            # shot cut
            x_2 = self.short_cut(input)
            x_2 = self.bn_short(x_2)
            # batch_size x frame x H/2 x W/2 x planes
            output = self.relu(x_1 + x_2)
        else:
            output = self.relu(x_1 + input)
            # batch_size x frame x H x W x planes
        return output


class ResNet18(tf.keras.Model):
    def __init__(self):
        super(ResNet18, self).__init__()
        # use random seed to make the initialization repeat
        # CNN 1-2
        self.input_part = input_block()
        # CNN 3-6
        self.block_11 = basic_block(planes=64, seed=1, name='layer1.0')
        self.block_12 = basic_block(planes=64, seed=2, name='layer1.1')
        # CNN 6-10
        self.block_21 = basic_block(planes=128, stride=2, seed=1, name='layer2.0')
        self.block_22 = basic_block(planes=128, seed=2, name='layer2.1')
        # CNN 10-14
        self.block_31 = basic_block(planes=256, stride=2, seed=1, name='layer3.0')
        self.block_32 = basic_block(planes=256, seed=2, name='layer3.1')
        # CNN 14-18
        self.block_41 = basic_block(planes=512, stride=2, seed=1, name='layer4.0')
        self.block_42 = basic_block(planes=512, seed=2, name='layer4.1')
        # Avg pooling
        self.Avg = tf.keras.layers.AvgPool2D(7, padding='VALID', name='avgpool')
        # self.fc = tf.keras.layers.Dense(1000, name='fc')
        # modified last fully connected layer
        self.fc = tf.keras.layers.Dense(18, name='fc')

    def forward(self, input):
        """
        here we define the forward function
        :param input: the input data
        :return: output tensor
        """
        # For each layer, a bias will also be initialized and add to the output after matrix multiply.
        # reshape
        x = tf.reshape(input, [-1, 224, 224, 3])
        # batch size x frames x 224 x 224 x 3
        x = self.input_part.forward(x)
        # batch size x frames x 56 x 56 x 64
        x = self.block_11.forward(x)
        x = self.block_12.forward(x)
        # batch size x frames x 56 x 56 x 64
        x = self.block_21.forward(x)
        x = self.block_22.forward(x)
        # batch size x frames x 28 x 28 x 128
        x = self.block_31.forward(x)
        x = self.block_32.forward(x)
        # batch size x frames x 14 x 14 x 256
        x = self.block_41.forward(x)
        x = self.block_42.forward(x)
        # batch size x frames x 7 x 7 x 512
        x = self.Avg(x)
        # batch size x frames x 1 x 1 x 512
        x = tf.reshape(x, [-1, 512])
        x = self.fc(x)
        return x
        # return tf.math.sigmoid(x)

def load_model(tf_model, torch_model):
    """
    In this function, we load the torch model to tensorflow models
    :param tf_model:
    :param torch_model:
    :return:
    """
    torch_layer_names = []
    # list all names of the torch model
    for name, module in torch_model.named_modules():
        torch_layer_names.append(name)

    tf_layer_names = {}
    for layers in tf_model.layers:
        if type(layers) == input_block or type(layers) == basic_block:
            for layer in layers.layers:
                tf_layer_names[layer.name] = layer
        else:
            tf_layer_names[layers.name] = layers

    # loading model
    print('Loading pretrained resnet-18 model from pytorch')
    for layer in tf_layer_names:
        if 'conv' in layer:
            tf_conv = tf_layer_names[layer]
            weights = torch_model.state_dict()[layer + '.weight'].numpy()
            weights_list = [weights.transpose((2, 3, 1, 0))]
            if len(tf_conv.weights) == 2:
                a = torch_model.state_dict()
                bias = torch_model.state_dict()[layer + '.bias'].numpy()
                weights_list.append(bias)
            tf_conv.set_weights(weights_list)
        elif 'bn' in layer:
            tf_bn = tf_layer_names[layer]
            gamma = torch_model.state_dict()[layer + '.weight'].numpy()
            beta = torch_model.state_dict()[layer + '.bias'].numpy()
            mean = torch_model.state_dict()[layer + '.running_mean'].numpy()
            var = torch_model.state_dict()[layer + '.running_var'].numpy()
            bn_list = [gamma, beta, mean, var]
            tf_bn.set_weights(bn_list)
        elif 'downsample.0' in layer:
            tf_downsample = tf_layer_names[layer]
            weights = torch_model.state_dict()[layer + '.weight'].numpy()
            weights_list = [weights.transpose((2, 3, 1, 0))]
            if len(tf_downsample.weights) == 2:
                bias = torch_model.state_dict()[layer + '.bias'].numpy()
                weights_list.append(bias)
            tf_downsample.set_weights(weights_list)
        elif 'downsample.1' in layer:
            tf_downsample = tf_layer_names[layer]
            gamma = torch_model.state_dict()[layer + '.weight'].numpy()
            beta = torch_model.state_dict()[layer + '.bias'].numpy()
            mean = torch_model.state_dict()[layer + '.running_mean'].numpy()
            var = torch_model.state_dict()[layer + '.running_var'].numpy()
            bn_list = [gamma, beta, mean, var]  # [gamma, beta, mean, var]
            tf_downsample.set_weights(bn_list)
        else:
            print('No parameters found for {}'.format(layer))
    # a = torch_model.state_dict()
    return tf_model


resnet_torch = models.resnet18(pretrained=True)
tf_model = ResNet18()
inputs = tf.keras.Input((None, None, 3))
tf_model.forward(inputs)
pretrained_resnet18 = load_model(tf_model, resnet_torch)