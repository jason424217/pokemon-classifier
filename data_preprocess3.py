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

df = pd.read_csv('/home/derafael/Documents/deep learning for computer vision/Final project/pokemon.csv')
pokemon_name_lst = list(df.Name)
pokemon_type1_lst = list(df.Type1)
pokemon_type2_lst = list(df.Type2)

# list all image paths
data_path = '/home/derafael/Documents/deep learning for computer vision/Final project/data_set/'
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
    tmp_type = [0] * 18
    if '-' not in name:
        idx = eval(name) - 1
        # first type
        tmp_type[All_types[pokemon_type1_lst[idx]]] = 1
        if type(pokemon_type2_lst[idx]) == str:
            # second type
            tmp_type[All_types[pokemon_type2_lst[idx]]] = 1
    else:
        name = name.split('-')
        if name[0] in unchanged_type:
            idx = eval(name[0]) - 1
            # first type
            tmp_type[All_types[pokemon_type1_lst[idx]]] = 1
            if type(pokemon_type2_lst[idx]) == str:
                # second type
                tmp_type[All_types[pokemon_type2_lst[idx]]] = 1
        # special cases, type may vary
        elif name[0] == '351':
            # Castform
            if name[1] == 'sunny':
                tmp_type[All_types['Fire']] = 1
            elif name[1] == 'rainy':
                tmp_type[All_types['Water']] = 1
            elif name[1] == 'snowy':
                tmp_type[All_types['Ice']] = 1
        elif name[0] == '413':
            # Wormadam
            tmp_type[All_types['Bug']] = 1
            if name[1] == 'plant':
                tmp_type[All_types['Grass']] = 1
            elif name[1] == 'sandy':
                tmp_type[All_types['Ground']] = 1
            elif name[1] == 'trash':
                tmp_type[All_types['Steel']] = 1
        elif name[0] == '479':
            # Rotom
            tmp_type[All_types['Electric']] = 1
            if name[1] == 'Mow':
                tmp_type[All_types['Grass']] = 1
            elif name[1] == 'heat':
                tmp_type[All_types['Fire']] = 1
            elif name[1] == 'wash':
                tmp_type[All_types['Water']] = 1
            elif name[1] == 'Frost':
                tmp_type[All_types['Ice']] = 1
            elif name[1] == 'Fan':
                tmp_type[All_types['Flying']] = 1
        elif name[0] == '492':
            # Shaymin
            tmp_type[All_types['Grass']] = 1
            if name[1] == 'sky':
                tmp_type[All_types['Flying']] = 1
        elif name[0] == '493':
            # Arceus
            tmp_type[All_types[name[1].capitalize()]] = 1
        elif name[0] == '555':
            if name[1] == 'standard':
                tmp_type[All_types['Fire']] = 1
            elif name[1] == 'zen':
                tmp_type[All_types['Ice']] = 1
        elif name[0] == '648':
            # Meloetta
            tmp_type[All_types['Normal']] = 1
            if name[1] == 'pirouette':
                tmp_type[All_types['Fighting']] = 1
            elif name[1] == 'aria':
                tmp_type[All_types['Psychic']] = 1
        elif name[0] == '6':
            # Charizard
            tmp_type[All_types['Fire']] = 1
            if name[2] == 'x':
                tmp_type[All_types['Dragon']] = 1
            else:
                tmp_type[All_types['Flying']] = 1
        elif name[0] == '531':
            # mega Audino
            tmp_type[All_types['Normal']] = 1
            tmp_type[All_types['Fairy']] = 1
        elif name[0] == '150':
            # Mewtwo
            tmp_type[All_types['Psychic']] = 1
            if name[2] == 'x':
                tmp_type[All_types['Fighting']] = 1
        elif name[0] == '306':
            # mega Aggron
            tmp_type[All_types['Steel']] = 1
        elif name[0] == '428':
            # mega Lopunny
            tmp_type[All_types['Normal']] = 1
            tmp_type[All_types['Fighting']] = 1
        elif name[0] == '720':
            # Hoopa
            tmp_type[All_types['Psychic']] = 1
            if name[1] == 'unbound':
                tmp_type[All_types['Dark']] = 1
            elif name[1] == 'confined':
                tmp_type[All_types['Ghost']] = 1
        elif name[0] == '181':
            # mega Ampharos
            tmp_type[All_types['Dragon']] = 1
            tmp_type[All_types['Electric']] = 1
        elif name[0] == '254':
            # mega Sceptile
            tmp_type[All_types['Dragon']] = 1
            tmp_type[All_types['Grass']] = 1
        elif name[0] == '383':
            # primal Groudon
            tmp_type[All_types['Ground']] = 1
            tmp_type[All_types['Fire']] = 1
        elif name[0] == '334':
            # mega Altaria
            tmp_type[All_types['Dragon']] = 1
            tmp_type[All_types['Fairy']] = 1
        elif name[0] == '130':
            # mega Gyarados
            tmp_type[All_types['Dark']] = 1
            tmp_type[All_types['Water']] = 1
        elif name[0] == '127':
            # mega Pinsir
            tmp_type[All_types['Bug']] = 1
            tmp_type[All_types['Flying']] = 1
        else:
            print(image_path, name)
    all_image_labels.append(tmp_type)

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
    img_final = tf.io.decode_image(img, expand_animations=False, channels=3)
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
train_label_ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.cast(train_image_labels, tf.int64), (-1, 18)))

validation_image_ds = validation_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
validation_label_ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.cast(validation_image_labels, tf.int64), (-1, 18)))

test_image_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
test_label_ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.cast(test_image_labels, tf.int64), (-1, 18)))

# pack them together and get a (image, label) dataset
train_dataset = tf.data.Dataset.zip((train_image_ds, train_label_ds))
validation_dataset = tf.data.Dataset.zip((validation_image_ds, validation_label_ds))
test_dataset = tf.data.Dataset.zip((test_image_ds, test_label_ds))

# take a batch size
batch_size = 100
train_dataset = train_dataset.batch(100)
validation_dataset = validation_dataset.batch(100)
test_dataset = test_dataset.batch(100)

print('done')