#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:03:33 2019

@author: Ellen
"""

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow import keras
import os
import glob
import scipy
import tensorboard
import datetime

###############################################
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(2)
###############################################

# ----------------Constantes---------------------------
path = 'KIMIA_Path_960'
representacoes = {}
train_dir = path + '/train'
validation_dir = path + '/validation'

# ----------------Definição do Modelo------------------
#model = VGG16(weights='imagenet', include_top=False, pooling='max')
model = VGG16(weights='imagenet', include_top=False)

# Adiciona Pooling
nmodel = Sequential()
nmodel.add(model)  # Adiciona a VGG num modelo sequencial?
nmodel.add(GlobalMaxPooling2D())  # Faz max pooling
nmodel.add(Dense(1024, activation='relu'))  # Adiciona camada densa
nmodel.add(Dropout(0.5))  # Faz dropout de 50% das entradas (meio alto, não?)
nmodel.add(Dense(1024, activation='relu'))  # Adiciona camada densa
nmodel.add(Dropout(0.25))  # Faz dropout de 25% das entradas
nmodel.add(Dense(20, activation='softmax'))  # Adiciona a camada de saída

# "Bloqueia" o treino das camadas que não o último bloco convolucional
for layer in range(len(model.layers)-4):
    model.layers[layer].treinable = False

# Compila o modelo novo, com a VGG16
# categorical cross entropy pq eh de classificao
# RMSprob
# acuracia como medida
nmodel.compile(loss='categorical_crossentropy',
               optimizer=keras.optimizers.RMSprop(lr=1e-5),
               metrics=['acc'])

# Usa data augmentation para adicionar imagens no diretório de treinamento
train = image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    # rescale=1./255,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input)

# Somente passa a função de preprocessamento da VGG16, sem fazer data augmentation no diretório de teste
validation = image.ImageDataGenerator(preprocessing_function=preprocess_input)

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

# Coloca labels em todas as imagens do diretório de teste
validation_generator = validation.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=val_batch,
    class_mode='categorical',
)

# Faz com que o TensorBoard tenha acesso aos dados de treinamento para visualização
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)


# treina a nova rede
# Método depreciado: fit suporta geradores agora.
history = nmodel.fit(
    x=train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    verbose=0,
    callbacks=[tensorboard_callback]
)

# Salva os pesos da nova rede
nmodel.save('finetuning_redimensionamento_max.h5')

# salvar gráfico do modelo

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('plots/acurracy.png')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('plots/loss.png')
