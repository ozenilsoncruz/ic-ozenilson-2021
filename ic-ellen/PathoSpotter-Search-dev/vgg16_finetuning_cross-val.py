#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:03:33 2019

@author: Ellen
"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# ----------------Tratamento de Dados------------------



# ----------------Definição do Modelo------------------

def create_network():
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
    
    return nmodel
   
'''
# --------------------Treinamento----------------------

clf = KerasClassifier(
    build_fn=create_network, epochs=50, batch_size=10)
resultados = cross_val_score(
    estimator=clf, X=train_generator, y=validation_generator, cv=10, scoring='accuracy')

# ---------------------Resultados----------------------

# Salva os pesos da nova rede
nmodel.save('finetuning_redimensionamento_max.h5')

# Salva a média de 
'''
