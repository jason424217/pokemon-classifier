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
# list all image paths
data_path = 'data_set/'
data_root = pathlib.Path(data_path)
# list all image names
all_image_paths = [str(path) for path in list(data_root.glob('*/*'))]
# Count the number of images
image_count = len(all_image_paths)

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
            all_image_labels.append(pokemon_type1_lst[idx])
    else:
        name = name.split('-')
        if name[0] == '172':
            # pichu
            all_image_labels.append(All_types['Electric'])
        elif name[0] == '201' or name[0] == '386':
            # Unknown, Deoxys
            all_image_labels.append(All_types['Psychic'])
        elif name[0] == '351':
            # Castform
            if name[1] == 'sunny':
                all_image_labels.append(All_types['Fire'])
            elif name[1] == 'rainy':
                all_image_labels.append(All_types['Water'])
            elif name[1] == 'snowy':
                all_image_labels.append(All_types['Ice'])
        elif name[0] == '412':
            # Burmy
            all_image_labels.append(All_types['Bug'])
        elif name[0] == '413':
            # Wormadam
            if name[1] == 'plant':
                all_image_labels.append(All_types['Bug+Grass'])
            elif name[1] == 'sandy':
                all_image_labels.append(All_types['Bug+Ground'])
            elif name[1] == 'trash':
                all_image_labels.append(All_types['Bug+Steel'])
        elif name[0] == '421':
            # Cherrim
            all_image_labels.append(All_types['Grass'])
        elif name[0] == '422':
            # Shellos
            all_image_labels.append(All_types['Water'])
        elif name[0] == '423':
            # Gastrodon
            all_image_labels.append(All_types['Water+Ground'])
        elif name[0] == '487':
            # Giratina
            all_image_labels.append(All_types['Ghost+Dragon'])
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
        else:
            print(name)

print('done')