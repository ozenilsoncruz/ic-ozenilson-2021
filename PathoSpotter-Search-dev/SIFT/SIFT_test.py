#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 2 14:15 2021
Este código tem como objetivo reproduzir o experimento
de busca por uma imagem num dataset utilizando o SIFT
como descritor de imagem.
Este código foi baseado em um exemplo do StackOverflow.
Exemplo: https://stackoverflow.com/a/50218701/6783733
@author: Ellen
"""
# ---------------------- Bibliotecas ----------------------
import cv2  # OpenCV que lida com o SIFT
import os  # Lida com o SO e caminhos e diretórios
import gc  # Garbage Collector

# ---------------- Constantes Ajustáveis ------------------
# Defina aqui o caminho para o diretório a ser carregado
directory_path = 'C:/Users/ellen/Documents/LACAD/Datasets/PS-Amiloidosis/validation/Amiloidosis'
# Número máximo de imagens mais semelhantes
k = 10

# Inicia o extrator Scale Invariant Feature Transform (SIFT)
sift = cv2.xfeatures2d.SIFT_create()

# --------------- Computando Descritores ------------------
# Vetor que guardará todas as imagens
processed_images = []
# Percorre todos os subdiretórios no caminho especificado atrás de imagens
# r=root, d=directories, f=files.
for r, d, f in os.walk(directory_path):
    # Para cada arquivo encontrado
    for file in f:
        # Pega o nome arquivo
        file_ = file.lower()
        # Se ele conter uma das extensões suportadas, executa
        if (('.jpg' in file_) or ('.png' in file_) or ('.jpeg' in file_) or ('.tif' in file_)):
            # Cria um objeto vazio
            obj = {}
            # Computa o caminho da imagem e guarda no objeto
            img_path = os.path.join(r, file)
            obj['path'] = img_path
            # Guarda o nome da imagem no objeto
            obj['name'] = file
            # Carrega a imagem: caminho, modo colorido
            img = cv2.imread(img_path, 1)
            # Detecta os pontos chave e computa o descritor pelo SIFT
            kp, des = sift.detectAndCompute(img, None)
            # Guarda os descritores da imagem no objeto
            obj['desc'] = des
            # Guarda os pontos chaves da imagem no objeto
            obj['kp'] = kp
            # Guarda este objeto em imagens processadas
            processed_images.append(obj)
            # Remove as variáveis auxiliares da memória
            del(file_)
            del(obj)
            del(img)
            del(kp)
            del(des)
        # Chama o Garbage Collector
        gc.collect()

# --------- Calcula a Distância Entre as Imagens ----------
# Inicializa a heap de imagens mais similares
heap = []
# Inicializa um objeto da classe brute-force descriptor matcher
bf = cv2.BFMatcher()

# Para cada uma daquelas imagens processadas
for img1 in processed_images:
    # Guardo as informações de arquivo
    img1_desc = img1['desc']
    img1_name = img1['name']
    # Percorre a lista de objetos processados anteriormente
    for img2 in processed_images:
        # Pega os descritores e pontos chave
        img2_desc = img2['desc']
        img2_kp = img2['kp']
        # Encontra as k melhores correspondências para cada descritor
        matches = bf.knnMatch(img1_desc, img2_desc, k=2)
        # Lista das boas correspondências
        good = []
        # Para cada par de descritores
        for m, n in matches:
            # Verifica se a distância entre os dois é
            if m.distance < 0.75*n.distance:
                good.append([m])
        # Verfica o número de boas distâncias
        a = len(good)
        # Calcula a porcentagem de boas distâncias entre todas elas
        percent = (a*100)/len(img2_kp)
        # Copia o objeto
        representation2 = img2
        # Define a similaridade ao outro vetor no objeto
        representation2['similarity'] = percent

        # Povoa a lista inicialmente com as K primeiras imagens
        if(len(heap) < k):
            heap.append(representation2)
            # Ordena por porcentagem de similaridade, da maior pra menor
            heap.sort(key=lambda x: x['similarity'], reverse=True)
        # Ordena a lista dinamicamente quando já tem mais que K elementos
        # e a porcentagem de similaridade do elemento é maior que o último da heap
        elif(percent >= heap[-1]['similarity']):
            del heap[-1]  # Remove o último elemento
            heap.append(representation2)  # Insere a representacao "melhor"
            # Ordena por porcentagem de similaridade, da maior pra menor
            heap.sort(key=lambda x: x['similarity'], reverse=True)

    # ----------- Mostra as Imagens Mais Similares ------------
    # Abre um arquivo para colocar os resultados
    file_path = "tests/Teste-SIFT-" + str(img1_name) + ".txt"
    f = open(file_path, "w")
    # Escreve qual a imagem buscada no arquivo
    head = "Imagem buscada: " + str(img1_name)
    f.write(head)
    f.write('\n')
    # Escreve o número de ocorrências da mesma classe no arquivo
    # Escreve alguns atribudos das K imagens mais similares no arquivo
    for image_representation in heap:
        to_write = 'Imagem: ' + \
            str(image_representation['path']) + ' | Similaridade: ' + \
            str(image_representation['similarity'])
        f.write(to_write)
        f.write('\n')
    f.close()
    gc.collect()
