#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we complete the data process part2 of final project
"""
import numpy as np
import os
import pathlib
import pandas as pd
import tensorflow as tf
import random

batch_size = 100

AUTOTUNE = tf.data.experimental.AUTOTUNE

All_types = {}
df = pd.read_csv('pokemon.csv')
pokemon_name_lst = list(df.Name)
pokemon_type1_lst = list(df.Type1)
pokemon_type2_lst = list(df.Type2)

num = 0
for idx in range(len(pokemon_name_lst)):
    if type(pokemon_type2_lst[idx]) == str:
        if pokemon_type1_lst[idx] + '+' + pokemon_type2_lst[idx] not in All_types and pokemon_type2_lst[idx] + '+' + pokemon_type1_lst[idx] not in All_types:
            All_types[pokemon_type1_lst[idx] + '+' + pokemon_type2_lst[idx]] = num
            num += 1
    else:
        if pokemon_type1_lst[idx] not in All_types:
            All_types[pokemon_type1_lst[idx]] = num
            num += 1
All_types['Electric+Fire'] = len(All_types)
All_types['Grass+Dragon'] = len(All_types)
All_types['Dragon+Fairy'] = len(All_types)
# list all image paths
data_path = 'data_set/'
data_root = pathlib.Path(data_path)
# list all image names
all_image_paths = [str(path) for path in list(data_root.glob('*/*'))]
# Count the number of images
image_count = len(all_image_paths)
# build set to store Pokémon where do not need to change there type
# Pichu, Unknown, Deoxys, Burmy, Cherrim, Shellos, Gastrodon, Giratina, Deerling, Sawsbuck, Genesect, Kyurem, Landorus,
# Keldeo, Thundurus, Basculin, Tornadus, Pikachu, Lucario, Vivillon, Heracross, Florges, Furfrou, Gallade, Meowstic,
# Kyogre, Glalie, Flabébé, Xerneas, Latios, Medicham, Abomasnow, Blastoise, Floette, Pumpkaboo, Garchomp, Gourgeist,
# Pidgeot, Aegislash, Houndoom, Banette, Diancie, Latias, Aerodactyl, Swampert, Camerupt, Sharpedo, Metagross, Steelix,
# Manectric, Gengar, Blaziken, Gardevoir, Rayquaza， Groudon, Mawile, Scizor, Alakazam, Sableye, Beedrill, Venusaur,
# Tyranitar, Slowbro, Salamence, Absol, Kangaskhan
unchanged_type = {'172', '201', '386', '412', '421', '422', '423', '487', '585', '586', '649', '646', '645', '647',
                  '642', '550', '641', '25', '448', '666', '214', '671', '676', '475', '678', '382', '362', '669',
                  '716', '381', '308', '460', '9', '670', '710', '445', '711', '18', '681', '229', '354', '719', '380',
                  '142', '260', '323', '319', '376', '208', '310', '94', '257', '282', '384', '383', '303', '212', '65',
                  '302', '15', '3', '248', '80', '373', '359', '115'}

# get the label names
all_image_labels = []
for image_path in all_image_paths:
    # print(image_path)
    name = image_path.split('/')[-1]
    name = name.split('.')[0]
    if '-' not in name:
        idx = eval(name) - 1
        if type(pokemon_type2_lst[idx]) == str:
            if pokemon_type1_lst[idx] + '+' + pokemon_type2_lst[idx] in All_types:
                all_image_labels.append(All_types[pokemon_type1_lst[idx] + '+' + pokemon_type2_lst[idx]])
            else:
                all_image_labels.append(All_types[pokemon_type2_lst[idx] + '+' + pokemon_type1_lst[idx]])
        else:
            all_image_labels.append(All_types[pokemon_type1_lst[idx]])
    else:
        name = name.split('-')
        if name[0] in unchanged_type:
            idx = eval(name[0]) - 1
            if type(pokemon_type2_lst[idx]) == str:
                if pokemon_type1_lst[idx] + '+' + pokemon_type2_lst[idx] in All_types:
                    all_image_labels.append(All_types[pokemon_type1_lst[idx] + '+' + pokemon_type2_lst[idx]])
                else:
                    all_image_labels.append(All_types[pokemon_type2_lst[idx] + '+' + pokemon_type1_lst[idx]])
            else:
                all_image_labels.append(All_types[pokemon_type1_lst[idx]])
        # special cases, type may vary
        elif name[0] == '351':
            # Castform
            if name[1] == 'sunny':
                all_image_labels.append(All_types['Fire'])
            elif name[1] == 'rainy':
                all_image_labels.append(All_types['Water'])
            elif name[1] == 'snowy':
                all_image_labels.append(All_types['Ice'])
        elif name[0] == '413':
            # Wormadam
            if name[1] == 'plant':
                all_image_labels.append(All_types['Bug+Grass'])
            elif name[1] == 'sandy':
                all_image_labels.append(All_types['Bug+Ground'])
            elif name[1] == 'trash':
                all_image_labels.append(All_types['Bug+Steel'])
        elif name[0] == '479':
            # Rotom
            if name[1] == 'Mow':
                all_image_labels.append(All_types['Electric+Grass'])
            elif name[1] == 'heat':
                all_image_labels.append(All_types['Electric+Fire'])
            elif name[1] == 'wash':
                all_image_labels.append(All_types['Water+Electric'])
            elif name[1] == 'Frost':
                all_image_labels.append(All_types['Electric+Ice'])
            elif name[1] == 'Fan':
                all_image_labels.append(All_types['Electric+Flying'])
        elif name[0] == '492':
            # Shaymin
            if name[1] == 'sky':
                all_image_labels.append(All_types['Grass+Flying'])
            elif name[1] == 'land':
                all_image_labels.append(All_types['Grass'])
        elif name[0] == '493':
            # Arceus
            all_image_labels.append(All_types[name[1].capitalize()])
        elif name[0] == '555':
            if name[1] == 'standard':
                all_image_labels.append(All_types['Fire'])
            elif name[1] == 'zen':
                all_image_labels.append(All_types['Ice'])
        elif name[0] == '648':
            # Meloetta
            if name[1] == 'pirouette':
                all_image_labels.append(All_types['Normal+Fighting'])
            elif name[1] == 'aria':
                all_image_labels.append(All_types['Normal+Psychic'])
        elif name[0] == '6':
            # Charizard
            if name[2] == 'x':
                all_image_labels.append(All_types['Dragon+Fire'])
            else:
                all_image_labels.append(All_types['Fire+Flying'])
        elif name[0] == '531':
            # mega Audino
            all_image_labels.append(All_types['Normal+Fairy'])
        elif name[0] == '150':
            # Mewtwo
            if name[2] == 'x':
                all_image_labels.append(All_types['Fighting+Psychic'])
            else:
                all_image_labels.append(All_types['Psychic'])
        elif name[0] == '306':
            # mega Aggron
            all_image_labels.append(All_types['Steel'])
        elif name[0] == '428':
            # mega Lopunny
            all_image_labels.append(All_types['Normal+Fighting'])
        elif name[0] == '720':
            # Hoopa
            if name[1] == 'unbound':
                all_image_labels.append(All_types['Dark+Psychic'])
            elif name[1] == 'confined':
                all_image_labels.append(All_types['Psychic+Ghost'])
        elif name[0] == '181':
            # mega Ampharos
            all_image_labels.append(All_types['Dragon+Electric'])
        elif name[0] == '254':
            # mega Sceptile
            all_image_labels.append(All_types['Grass+Dragon'])
        elif name[0] == '383':
            # primal Groudon
            all_image_labels.append(All_types['Ground+Fire'])
        elif name[0] == '334':
            # mega Altaria
            all_image_labels.append(All_types['Dragon+Fairy'])
        elif name[0] == '130':
            # mega Gyarados
            all_image_labels.append(All_types['Water+Dark'])
        elif name[0] == '127':
            # mega Pinsir
            all_image_labels.append(All_types['Bug+Flying'])
        else:
            print(image_path, name)

# shuffle
temp = list(zip(all_image_paths, all_image_labels))
random.seed(1)
random.shuffle(temp)
all_image_paths, all_image_labels = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
all_image_paths, all_image_labels = list(all_image_paths), list(all_image_labels)

# load image and preprocess
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    '''
    try:
        img_final = tf.image.decode_png(img, channels=3)
    except tf.python.framework.errors_impl.InvalidArgumentError:
    '''
    img_final = tf.io.decode_image(img, expand_animations = False, channels=3)
    img_final = tf.image.resize(img_final, [224, 224])
    img_final = img_final / 255.0
    return tf.image.per_image_standardization(img_final)

# 16000 image as training data
train_image_path = all_image_paths[:16000]
train_image_labels = all_image_labels[:16000]
# 2000 image as validation dataset
validation_image_path = all_image_paths[16000:18000]
validation_image_labels = all_image_labels[16000:18000]
# 2000 image as test dataset
test_image_path = all_image_paths[18000:20000]
test_image_labels = all_image_labels[18000:20000]

num_class = len(All_types)

"""
Construct image datasets
"""
train_path_ds = tf.data.Dataset.from_tensor_slices(train_image_path)
validation_path_ds = tf.data.Dataset.from_tensor_slices(validation_image_path)
test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_path)

# image & label datasets
train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
train_label_ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.one_hot(tf.cast(train_image_labels, tf.int64), num_class), (-1, num_class)))

validation_image_ds = validation_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
validation_label_ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.one_hot(tf.cast(validation_image_labels, tf.int64), num_class), (-1, num_class)))

test_image_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
test_label_ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.one_hot(tf.cast(test_image_labels, tf.int64), num_class), (-1, num_class)))

# pack them together and get a (image, label) dataset
train_dataset = tf.data.Dataset.zip((train_image_ds, train_label_ds))
validation_dataset = tf.data.Dataset.zip((validation_image_ds, validation_label_ds))
test_dataset = tf.data.Dataset.zip((test_image_ds, test_label_ds))

# take a batch size
train_dataset = train_dataset.batch(100)
validation_dataset = validation_dataset.batch(100)
test_dataset = test_dataset.batch(100)

