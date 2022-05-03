#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 17:08:33 2020
Este código tem como objetivo realizar um clustering nos
dados obtidos no arquivo search_experiment.py.
@author: Ellen
"""
# ------------------ Bibliotecas Gerais -------------------
import pandas as pd # Pandas, para o Dataframe
import numpy as np # Numpy, para uma boa impressão de dados
from fcmeans import FCM # Fuzzy C-Means Clustering

# Lê o dataframe salvo pelo search_experiment.py
unpickled_df = pd.read_pickle("./dataset.pkl")
# Guarda o dataframe na memória, como vetores numpy
dataset = unpickled_df.to_numpy()
# Cria uma classe com o número de clusters a se criar
fcm = FCM(n_clusters=7)
# Atribui os vetores a seus clusters
fcm.fit(dataset)
# Guarda os centros calculados
fcm_centers = fcm.centers
# Guarda as labels atribuídas
fcm_labels  = fcm.u.argmax(axis=1)

# ---------------- Salvando os Resultados -----------------
f = open("Data-Clustering_7-Clusters.txt", "w")
f.write("Centers:")
f.write('\n')
f.write(np.array_str(fcm_centers))
f.write('\n')
f.write("Labels:")
f.write('\n')
f.write(str(fcm_labels))
f.close()