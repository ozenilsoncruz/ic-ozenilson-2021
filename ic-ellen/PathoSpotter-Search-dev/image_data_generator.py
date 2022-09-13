#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:37:33 2020

@author: Ellen
"""
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# ----------------Constantes---------------------------
path = 'KIMIA_Path_960'
representacoes = {}
train_dir = path + '/train'

# ---------------Tratamento de Imagens-----------------

# Usa data augmentation para adicionar imagens no diretório de treinamento
train = image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    # rescale=1./255,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input)

# tamanhos de batch
train_batch = 32
val_batch = 8
image_size = 224

# Coloca labels em todas as imagens do diretório de treino
train_generator = train.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batch,
    class_mode='categorical',
    shuffle=True)

# print(train_generator.color_mode) # Retornando RGB

inp = open('kimia-input.csv','wb')
out = open('kimia-output.csv','wb')
for x_val, y_val in train_generator:
    np.save(inp, x_val)
    np.save(out, y_val)
inp.close()
out.close()