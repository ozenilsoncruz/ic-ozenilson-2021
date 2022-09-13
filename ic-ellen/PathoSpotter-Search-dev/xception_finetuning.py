#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Set 25 15:13:33 2020
Este código tem como objetivo realizar o transfer
learning e fine tuning da Xception para o contexto
de imagens de patologias renais.
@author: Ellen
"""
# ------------------------------Importações--------------------------------

import datetime # Auxilia nos logs
import matplotlib.pyplot as plt # Auxilia na visualização
import tensorflow as tf  # Tensorflow
from tensorflow.keras.applications.xception import Xception  # Xception
# Pré-processamento da Xception
from tensorflow.keras.applications.xception import preprocess_input
# Camadas: Densa, Pooling Médio
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model  # Classe Model
# Funções de pré-processamento de imagem
from tensorflow.keras.preprocessing import image
# Para melhor observar os logs, utiliza-se o TensorBoard
from tensorflow.keras.callbacks import TensorBoard

# --------------------------Constantes-------------------------------------

path = 'PathoSpotter' # PathoSpotter
train_dir = path + '/train'
validation_dir = path + '/validation'

# ------------------------Tratamento das Imagens---------------------------
# /!\ ATENÇÃO! /!\ Essa parte do código é baseada no fine tuning da VGG16
# realizado pelo antigo aluno de TCC Gabriel.

# Usa data augmentation para adicionar imagens no diretório de treinamento
# /!\ Detalhe: os dados originais nunca são usados
train_generator = image.ImageDataGenerator(
    rotation_range=20,
    #width_shift_range=0.2,
    # rescale=1.0/255,
    #height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input)

# Somente passa a função de preprocessamento da Xception
validation_generator = image.ImageDataGenerator(
    preprocessing_function=preprocess_input)

# Tamanhos de batch (intervalo de imagens a ser usado para treinar a rede)
train_batch = 32
val_batch = 8
image_size = 299  # ATENÇÃO! O input da Xception difere da VGG16

# Coloca labels em todas as imagens do diretório de treino
train_augmented = train_generator.flow_from_directory(
    train_dir,
    save_to_dir = 'data_aug/',
    save_format = 'jpeg',
    target_size=(image_size, image_size),
    batch_size=train_batch,
    class_mode='categorical',
    shuffle=True)

# Coloca labels em todas as imagens do diretório de teste
validation_augmented = validation_generator.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=val_batch,
    class_mode='categorical',
)

# --------------------------Transfer Learning----------------------------

# Cria o modelo base, pré-treinado da imagenet
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3), pooling='avg')
# Congelando o modelo
base_model.trainable = False
# Colocando o modelo em inferência
x = base_model(base_model.input, training=False)
# Adiciona uma camada fully-connected que será treinada
x = Dense(1024, activation='relu')(x)
# Um classificador denso com várias classes
predictions = Dense(6, activation='softmax')(x) # PathoSpotter
# Instanciando o novo modelo a ser treinado
model = Model(inputs=base_model.input, outputs=predictions)
# Mostrando o modelo
# model.summary()
# Compila o novo modelo
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['acc'])
# Faz com que o TensorBoard tenha acesso aos dados de treinamento para visualização
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)
# Treinamento da nova rede neural
history = model.fit(
    x=train_augmented,
    steps_per_epoch=train_augmented.samples/train_augmented.batch_size,
    epochs=50,
    validation_data=validation_augmented,
    validation_steps=validation_augmented.samples/validation_augmented.batch_size,
    verbose=0,
    callbacks=[tensorboard_callback]
)

# ------------------------------Fine Tuning--------------------------------
# Descongela o Modelo Base
base_model.trainable = True

# Congela as camadas da Xception que não o último bloco convolucional
for layer in range(len(base_model.layers)-8):
    base_model.layers[layer].trainable = False
    
# Compila o novo modelo
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['acc'])

# Faz com que o TensorBoard tenha acesso aos dados de treinamento para visualização
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)

# Treinamento da nova rede neural
history = model.fit(
    x=train_augmented,
    steps_per_epoch=train_augmented.samples/train_augmented.batch_size,
    epochs=30,
    validation_data=validation_augmented,
    validation_steps=validation_augmented.samples/validation_augmented.batch_size,
    verbose=0,
    callbacks=[tensorboard_callback]
)

# -------------------------Salvando os Resultados--------------------------
# Salva os pesos da nova rede
model.save('finetuning_xception-newPS.h5') # PathoSpotter

# Salvar gráfico do modelo

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('finetuning_acurracy.png')
plt.clf() # Limpa a figura

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('finetuning_loss.png')
plt.clf() # Limpa a figura
