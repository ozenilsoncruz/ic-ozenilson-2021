#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:00:18 2020

@author: gabriel
Modo de uso:
python3 recortar_dataset.py /home/gabriel/Downloads/dataset

"""

import os
import sys
import random
import shutil

#captura path do diretorio
path = sys.argv[1]
tamanho = None 
if len(sys.argv) == 3:
    tamanho = int(sys.argv[2])
    
lista_dir = []
lista_dir_sufixo = []
tamanho_dir = {}
menor = None
sufix_novo_diretorio = '-recortado'

#deleta barra da path
if path[-1] == '/':
    path=path[:-1]
#lista arquivos
for root, dirs, files in os.walk(path, topdown=False):
    for name in dirs:
        diretorio = os.path.join(root, name)
        lista_dir.append(diretorio)
        lista_dir_sufixo.append(name)
        dpath, ddirs, dfiles = next(os.walk(diretorio))
        tamanho_dir[diretorio]=len(dfiles)
        
        if menor == None:
            menor = tamanho_dir[diretorio]
        elif menor > tamanho_dir[diretorio]:
            menor = tamanho_dir[diretorio]
#seta tamanho
if tamanho is None:
    tamanho = menor
    
for i, diretorio in enumerate(lista_dir):
    #novo do nome diretorio
    novo_diretorio = os.path.join(path+sufix_novo_diretorio, lista_dir_sufixo[i])
    print('copiando {} arquivos de {} para {}'.format(tamanho, diretorio, novo_diretorio))
    
    #cria o novo diretorio
    os.makedirs(novo_diretorio)
    
    #lista os diretorios antigos
    dpath, ddirs, dfiles = next(os.walk(diretorio))
    
    #embaralha lista
    random.shuffle(dfiles)
    
    #seleciona os primeiros itens da lista para copiar
    para_copiar = dfiles[0:tamanho]
    
    #copia os primeiros $tamanhos arquivos para o diretorio
    for arquivo in para_copiar:
        shutil.copy(os.path.join(diretorio, arquivo), novo_diretorio)
        
        
        
        
