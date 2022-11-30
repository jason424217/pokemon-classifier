#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we complete the data process part1 of final project
"""
import numpy as np
import os
import pathlib
import pandas as pd
import tensorflow as tf
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

All_types = {'Normal': 0, 'Fire': 1, 'Water': 2, 'Grass': 3, 'Flying': 4, 'Fighting': 5, 'Poison': 6, 'Electric': 7,
             'Ground': 8, 'Rock': 9, 'Psychic': 10, 'Ice': 11, 'Bug': 12, 'Ghost': 13, 'Steel': 14, 'Dragon': 15,
             'Dark': 16, 'Fairy': 17}
df = pd.read_csv('pokemon.csv')
pokemon_name_lst = list(df.Name)
pokemon_type1_lst = list(df.Type1)
pokemon_type2_lst = list(df.Type2)

pokemon_dict = {}
for idx, name in enumerate(pokemon_name_lst):
    # assign the first type
    pokemon_dict[name] = pokemon_type1_lst[idx]


# list all image paths
data_path = 'PokemonData/'
data_root = pathlib.Path(data_path)
# list all image names
all_image_paths = [str(path) for path in list(data_root.glob('*/*'))]

from pathlib import Path
import imghdr
data_dir = "PokemonData/"
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir).rglob("*/*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")

# Count the number of images
image_count = len(all_image_paths)


# get the label names
pokemon_names = os.listdir(data_root)

# assign label index
label_to_index = dict((name, All_types[pokemon_dict[name.lower()]]) for name in pokemon_names)
# create a list of the label of all images
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

# shuffle
temp = list(zip(all_image_paths, all_image_labels))
random.seed(1)
random.shuffle(temp)
all_image_paths, all_image_labels = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
all_image_paths, all_image_labels = list(all_image_paths), list(all_image_labels)

def preprocess_image(img):
    img_final = tf.image.decode_jpeg(img, channels=3)
    img_final = tf.image.resize(img_final, [32, 32])
    img_final = img_final / 255.0
    img_final = tf.image.per_image_standardization(img_final)
    img_final = tf.reshape(img_final, [32*32*3])
    return img_final

# load image and preprocess
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    return preprocess_image(img)

# 5000 image as training data
train_image_path = all_image_paths[:5000]
train_image_labels = all_image_labels[:5000]
# 1000 image as validation dataset
validation_image_path = all_image_paths[5000:6000]
validation_image_labels = all_image_labels[5000:6000]
# 796 image as test dataset
test_image_path = all_image_paths[6000:]
test_image_labels = all_image_labels[6000:]

"""
Construct image datasets
"""
# path dataset
train_path_ds = tf.data.Dataset.from_tensor_slices(train_image_path)
validation_path_ds = tf.data.Dataset.from_tensor_slices(validation_image_path)
test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_path)

# image & label datasets
train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
train_label_ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.one_hot(tf.cast(train_image_labels, tf.int64), 18), (-1, 18)))

validation_image_ds = validation_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
validation_label_ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.one_hot(tf.cast(validation_image_labels, tf.int64), 18), (-1, 18)))

test_image_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
test_label_ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.one_hot(tf.cast(test_image_labels, tf.int64), 18), (-1, 18)))

# pack them together and get a (image, label) dataset
train_dataset = tf.data.Dataset.zip((train_image_ds, train_label_ds))
validation_dataset = tf.data.Dataset.zip((validation_image_ds, validation_label_ds))
test_dataset = tf.data.Dataset.zip((test_image_ds, test_label_ds))

# take a batch size
train_dataset = train_dataset.batch(100)
validation_dataset = validation_dataset.batch(100)
test_dataset = test_dataset.batch(796)
