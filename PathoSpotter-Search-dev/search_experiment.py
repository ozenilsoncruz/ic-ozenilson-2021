#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Set 27 14:19:33 2020
Este código tem como objetivo reproduzir o experimento
de busca por uma imagem num dataset, originalmente idealizado
pelo aluno de TCC Gabriel.
@author: Ellen
"""
# ------------------ Bibliotecas Gerais -------------------
import numpy as np  # Numpy
import keras  # Keras
# Lida com processamento de imagens
from keras.preprocessing import image
# Classe Model do Keras
from keras.models import Model
# Lida com o SO e caminhos e diretórios
import os
# Calcula a distância entre os vetores
from scipy.spatial.distance import cosine, euclidean
# Copia elementos
import copy
import gc # Garbage Collector
import pandas as pd
# ---------------- Bibliotecas de Modelo ------------------
# Normalização da Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import Xception  # Xception

# ----------------- Constantes de Modelo ------------------
# Defina aqui a altura e largura da imagem a ser carregada
img_width = img_height = 299
# Defina aqui o caminho para a imagem a ser carregada
#search_img_path = 'Datasets/PS-SEARCH-TESTE/PS-SEARCH-TESTE (86).jpg'
# Defina aqui o caminho para o modelo a ser carregado
model_path = 'models/data_aug-model_2.h5'
# Defina aqui a camada de saída do seu modelo
#output_layer = 'dense'
# Defina aqui o caminho para o diretório a ser carregada
directory_path = 'Datasets/PS-SEARCH-TESTE'
# Defina a distância, entre 'cosine' e 'euclidean'
distance = 'cosine'
# Defina a quantidade k de resultados que você deseja ver
k = 10

# ------------------- Carrega o Modelo --------------------
# Carrega os pesos do modelo para uma variável auxiliar
load_aux = keras.models.load_model(model_path)
#load_aux.get_layer('xception').summary()
# Instancia um modelo com os pesos da auxiliar e camada de saída indicada
model = Model(inputs=load_aux.get_layer('xception').inputs,
              outputs=load_aux.get_layer('xception').get_layer('global_average_pooling2d').output)
#model = Model(inputs=load_aux.inputs,
#              outputs=load_aux.get_layer(output_layer).output)
# Mostra como a rede está organizada na memória
#model.summary()
'''
# -------------------- Recebe a Imagem --------------------
# Carrega a imagem do caminho com o tamanho especificado
imagem = image.load_img(path=search_img_path,
                        target_size=(img_height, img_width))
# Transforma a imagem em array
input_arr = image.img_to_array(imagem)
# Coloca mais uma dimensão no array
input_arr = np.expand_dims(input_arr, axis=0)

# ----------------- Pré-Processa a Imagem -----------------
# Normaliza o array para a rede conforme sua necessidade
pre_processed_array = preprocess_input(input_arr)
# Prevê o array com a rede instanciada
predicted_array = np.array(model.predict(
    pre_processed_array)).flatten().tolist()
'''
# --------------- Pré-Processa o Diretório ----------------
# Cria um vetor que vai receber as imagens do diretório após processadas
pre_processed_images = []
#count_val = 0
# Percorre todos os subdiretórios no caminho especificado atrás de imagens
# r=root, d=directories, f = files
for r, d, f in os.walk(directory_path):
    # Para cada arquivo encontrado
    for file in f:
        # Pega o nome arquivo
        file_ = file.lower()
        # Se ele conter uma das extensões suportadas, executa
        if (('.jpg' in file_) or ('.png' in file_) or ('.jpeg' in file_) or ('.tif' in file_)):
            #count_val += 1
            # Cria um objeto vazio
            obj = {}
            # Guarda o caminho da imagem no objeto
            img_path = os.path.join(r, file)
            obj['path'] = img_path
            # Guarda o nome da imagem no objeto
            obj['name'] = file
            # Carrega a imagem no tamanho do arquivo especificado
            img_data = image.load_img(img_path, target_size=(img_height, img_width))
            # Transforma em array
            x = image.img_to_array(img_data)
            # Coloca mais uma dimensão no array
            x = np.expand_dims(x, axis=0)
            # Normaliza o array para a rede conforme a necessidade
            x = preprocess_input(x)
            # Prevê o array e transforma num vetor de uma dimensão
            img_representation = np.array(model.predict(x)).flatten().tolist()
            # Salva o vetor no objeto
            obj['representation'] = img_representation
            # Salva o objeto na lista
            pre_processed_images.append(obj)
            del(file_)
            del(obj)
            del(x)
            del(img_data)
        gc.collect()
#print(count_val)
'''
# -------------- Salva o Pré-Processamento ----------------
df = pd.DataFrame(pre_processed_images)
df.to_pickle("./dataset.pkl")
'''
# --------- Calcula a Distância Entre as Imagens ----------
# Cria a heap de imagens menos distantes vazia
heap = []

for img1 in pre_processed_images:
    img1_representation = img1['representation']
    img1_name = img1['name']
    # Percorre a lista de objetos processados anteriormente
    for img2 in pre_processed_images:
        # Pega o vetor do objeto
        img2_representation = img2['representation']
        # Pega o objeto, copiando para evitar problemas com Numpy
        representation2 = copy.deepcopy(img2)

        # Verifica o método de cálculo da distância e calcula
        if(distance == 'cosine'):
            d = cosine(img1_representation, img2_representation)
        else:
            d = euclidean(img1_representation, img2_representation)

        # Define a similaridade ao outro vetor no objeto
        representation2['similarity'] = d

        # Povoa a lista inicialmente com as K primeiras imagens
        if(len(heap) < k):
            heap.append(representation2)
            # Ordena por distância, da menor pra maior
            heap.sort(key=lambda x: x['similarity'], reverse=False)
        # Ordena a lista dinamicamente quando já tem mais que K elementos
        elif(d <= heap[-1]['similarity']):
            del heap[-1]  # Remove o último elemento
            heap.append(representation2)  # Insere a representacao "melhor"
            # Ordena por distância, da menor pra maior
            heap.sort(key=lambda x: x['similarity'], reverse=False)

    # ----------- Mostra as Imagens Mais Similares ------------
    # Conta quantos elementos tem no caminho o nome da classe desejada
    #occ = sum('sclerosis' in obj['path'] for obj in heap)
    # Informa no cmd para agilizar o processo
    #print('Ocorrências da mesma classe: ',occ)
    # Abre o arquivo para colocar os resultados
    file_path = "tests/Teste-CV-Xception-"+ str(img1_name) + ".txt"
    f = open(file_path, "w")
    # Escreve qual a imagem buscada no arquivo
    head = "Imagem buscada: " + str(img1_name)
    f.write(head)
    f.write('\n')
    # Escreve o número de ocorrências da mesma classe no arquivo
    #f.write('Ocorrências da mesma classe: '+str(occ))
    #f.write('\n')
    # Escreve alguns atribudos das K imagens mais similares no arquivo
    for image_representation in heap:
        to_write = 'Imagem: ' + \
            str(image_representation['path']) + '| Similaridade: ' + \
            str(image_representation['similarity'])
        f.write(to_write)
        f.write('\n')
    f.close()
    gc.collect()
