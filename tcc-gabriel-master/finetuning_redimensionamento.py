#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:03:33 2019

@author: gabriel
"""

from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras
from keras.callbacks import TensorBoard
import numpy as np
import os
import glob
import scipy
import matplotlib.pyplot as plt
import tensorboard
import datetime

###############################################
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
###############################################

#----------------constantes---------------------------
path = '/home/gabriel/Downloads/KIMIA_Path_960/'
representacoes={}
train_dir = '/home/gabriel/KIMIA_Path_960/train/'
validation_dir = '/home/gabriel/KIMIA_Path_960/validation/'

#----------------definicao do modelo------------------
#model = VGG16(weights='imagenet', include_top=False, pooling='max')
model = VGG16(weights='imagenet', include_top=False)

#adiciona pooling

nmodel = Sequential()
nmodel.add(model)
nmodel.add(GlobalMaxPooling2D())
nmodel.add(Dense(1024, activation='relu'))
nmodel.add(Dropout(0.5))
nmodel.add(Dense(1024, activation='relu'))
nmodel.add(Dropout(0.25))
nmodel.add(Dense(20, activation='softmax'))

#cria modelo do finetuning
#nmodel = Model(inputs=model.input, outputs=predictions)

#camadas que nao deverao ser retreinadas do treinamento
#todas menos o último bloco convolucional
for layer in range(len(model.layers)-4):
    model.layers[layer].treinable=False
    
#compilar o modelo
#categorical cross entropy pq eh de classificaco
#RMSprob
#acuracia como medida    
nmodel.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])
#cria conjunto de treinamento com data augmentation
train = image.ImageDataGenerator(
    rotation_range=20,
      width_shift_range=0.2,
      #rescale=1./255,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      preprocessing_function=preprocess_input)

validation = image.ImageDataGenerator(preprocessing_function=preprocess_input)

#tamanhos de bacth
train_batch = 32
val_batch = 8
image_size = 224

train_generator = train.flow_from_directory(
        train_dir, 
        target_size=(image_size, image_size), 
        batch_size=train_batch, 
        class_mode='categorical',
        shuffle=True)

validation_generator = validation.flow_from_directory(
        validation_dir, 
        target_size=(image_size, image_size), 
        batch_size=val_batch, 
        class_mode='categorical',
        )
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)


#treina a nova rede
history = nmodel.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1,callbacks=[tensorboard_callback])

nmodel.save('finetuning_redimensionamento_max.h5')

#salvar gráfico do modelo
import matplotlib.pyplot as plt


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

