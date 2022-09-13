#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:40:19 2020

@author: gabriel
"""

from sklearn.manifold import TSNE
from DataBase import DataBase
import os
import collections
import pandas as pd
import sys
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# For reproducability of the results
np.random.seed(42)

path = sys.argv[1]
db = DataBase()
print('inicializando banco de dados..')
#carrega as representacoes
name_representations = ['pspotter', 'inception', 'vgg16', 'vgg16ft']
db.load_representations(path, name_representations)
separados = {}
for representacao in name_representations:
    classes = {}
    for imagem in db.images:
        c = imagem.path.split(os.path.sep)[-2]
        if c not in classes:
            classes[c] = []
        classes[c].append(imagem.representations[representacao])
    separados[representacao] = classes
    
for representacao in name_representations:
    completo = []
    for key in list(separados[representacao].keys()):
        completo.extends(separados[representacao][key])
    completo_normalizado = StandardScaler().fit_transform(completo)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(completo_normalizado)
    

    

    
