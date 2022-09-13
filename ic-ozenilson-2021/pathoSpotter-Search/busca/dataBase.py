#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:23:10 2019
@author: gabriel


Modo de uso:
from DataBase import DataBase

path = '/home/gabriel/bd1.json'
# nome das represesentacoes no json
name_representations = ['pspotter', 'inception', 'vgg16']
db = DataBase()
db.load_representations(path, name_representations)
img = '/home/gabriel/Dropbox/UEFS/2019.1/TCC/codigos/principal/img/S19.tif'
heap = db.search(img, model=db.PSPOTTER, k=20, distance='cosine')
"""
import copy
import json
import os
import time

import cv2
import numpy as np
from keras.applications import vgg16, inception_v3, resnet_v2, xception
from keras.models import Model, load_model
from keras.preprocessing import image
from scipy.spatial.distance import braycurtis, canberra, cosine, euclidean

from imageRepresentation import ImageRepresentation



class DataBase:
    # constantes
    VGG16 = 0
    INCEPTION = 1
    XCEPTION = 2
    RESNET50 = 3
    REPRESENTATONS = ['vgg16', 'inception', 'xception', 'resNet50']
    
    def __init__(self):
        self.__vgg16__ = None
        self.__inception__ = None
        self.__xception__ = None
        self.__resNet50__ = None
        self.images = None # banco de dados
        
    def iniciar_modelos(self, path):
        # modelos
        self.__vgg16__ = load_model(path+'vgg16/vgg16.h5')
        self.__vgg16__ = Model(inputs=self.__vgg16__.inputs, 
                               outputs=self.__vgg16__.get_layer('global_max_pooling2d').output)

        self.__inception__ = load_model(path+'inception/inception.h5')
        self.__inception__ = Model(inputs=self.__inception__.inputs,
                             outputs=self.__inception__.get_layer('global_max_pooling2d').output)

        self.__xception__ = load_model(path+'xception/xception.h5')
        self.__xception__ = Model(inputs=self.__xception__.inputs,
                            outputs=self.__xception__.get_layer('global_max_pooling2d').output)
        
        self.__resNet50__ = load_model(path+'resNet50/resNet50.h5')
        self.__resNet50__ = Model(inputs=self.__resNet50__.inputs,
                            outputs=self.__resNet50__.get_layer('global_max_pooling2d').output)
        
    def __len__(self):
        return len(self.images)

    def __image_preparation__(self, img_path, model):
        '''
        Metodo responsavel pelo pre processamento da imagem: leitura, redimensionamento, normalizacao
        '''
        if model == self.VGG16:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            if model == self.VGG16:
                x = vgg16.preprocess_input(x)
            #else: 
                #x = resnet_v2.preprocess_input(x)
        
        elif model == self.INCEPTION or model == self.XCEPTION or model == self.RESNET50:
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            if model == self.INCEPTION:
                x = inception_v3.preprocess_input(x)
            elif model == self.XCEPTION:
                x = xception.preprocess_input(x)
            else: 
                x = resnet_v2.preprocess_input(x)
        
        return x

    def __get_image_representation(self, img_data, model):
        '''
        Retorna a imagem aplicada a uma rede neural
        '''
        model = self.__get_model__(model)
        predict = np.array(model.predict(img_data)).flatten().tolist()
        #print("O vetor é: \n", predict)
        return predict

    def __get_model__(self, num_model):
        '''
        Retorna um modelo de rede neural
        '''
        # retorna o modelo vgg16
        if num_model == self.VGG16:
            return self.__vgg16__

        # retorna o modelo inception
        elif num_model == self.INCEPTION:
            return self.__inception__

        # retorna o modelo xception
        elif num_model == self.XCEPTION:
            return self.__xception__
        
        # retorna o modelo resNet50
        elif num_model == self.RESNET50:
            return self.__resNet50__

    def __img_resize__(self, img, target):
        '''
        Metodo utilizado para redimensionar imagens, sem manter aspecto.
        Metodo recebe o modelo a que se destina o redmensionamento.
        '''
        if target == self.INCEPTION or target == self.XCEPTION:
            dim = (299, 299)
        else: #(target == self.VGG16 or target == self.RESNET50):
            dim = (224, 224)

        # redimensiona a imagem com o modo bilienar
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        # muda o tipo do vetor para melhorar o processemanto
        resized = resized.astype('float32')
        # normaliza a imagem com valores entre 0 e 1
        resized = resized / 255.0

        return resized
    
    def salvar_json(self, path, arq):
        with open(path+'db33.json', 'w') as f:
            json.dump(arq, f)
            f.close()
        print('json salvo em {}'.format(path+'db33.json'))

    def load_representationsJson(self, json_path, names_representations):
        '''
        Metodo responsavel por carregar representacoes de um arquivo json.

        json_path: path para arquivo json contendo as representacoes

        names_representations: lista com nome nome de representacoes presentes no arquivo json.
        '''

        images = list()
        start = time.time()
        
        with open(json_path, 'r') as arquivo:
            bd = json.load(arquivo)
        for img in bd:
            rep = ImageRepresentation(name=img['name'], image_path=img['path'])
            for s in names_representations:
                rep.add_representation(name_representation=s, representation=img[img['name']][s])
            images.append(rep)
        self.images = images
        
        end = time.time()
        return "Carregamento do arquivo finalizado em "+str(end - start)+" segundos" 
    
    def load_representationsDB(self, db):
        """Metodo responsavel por carregar um banco de imagens organizado em pastase salva em um json.

        Args:
            db (str): caminho do banco de imagens

        Returns:
            list: lista das representacoes de imagens
        """
        images = list()
        arq_json = list()
        for pasta, subpasta, arquivo in os.walk(db):
            for arq in arquivo:
                image_path = os.path.join(pasta, arq)
                rep = ImageRepresentation(name=arq, image_path=image_path)
                for s in self.REPRESENTATONS:
                    img_data = self.__image_preparation__(image_path, self.REPRESENTATONS.index(s))
                    img_representation = self.__get_image_representation(img_data, self.REPRESENTATONS.index(s))
                    rep.add_representation(name_representation=s, representation=img_representation)
                arq_json.append(rep.to_dict()) #armazena as informações de cada imagem
                images.append(rep)
                
        self.salvar_json(db, arq_json) #salva em um arquivo json 
        self.images = images 
        return "Carregamento das imagens finalizado"
    
    def search(self, img, model=2, k=10, distance='cosine'):
        '''
        Metodo responsavel por realizar uma busca.

        img_path: caminho para imagem objeto da busca ou representacao da img
        model: numero do modelo a ser usado, padrao eh o pspotter k,
        realizando uma busca utilizand o ps-k search
        k: quantidades de imagens e serem buscadas
        retorno: k imagens mais similares à buscada
        '''

        heap = []
        if type(img) == str:
            img_data = self.__image_preparation__(img, model)
            img_representation = self.__get_image_representation(img_data, model) 
        else:
            img_representation = img.representations[self.REPRESENTATONS[model]]

        #start = time.time() 
        for representation in self.images:
            img2_representation = representation.representations[self.REPRESENTATONS[model]]
            #representation = copy.deepcopy(img2)
            # calcula a distancia por metricas diferentes
            if(distance == 'braycurtis'):
                d = braycurtis(img_representation, img2_representation)
            elif('cosine' in distance):
                d = cosine(img_representation, img2_representation)
            elif(distance == 'euclidean'):
                d = euclidean(img_representation, img2_representation)
            
            
            # seta similaridade no objeto de representacao
            representation.similarity = d
            # povoa a lista inicialmente
            if (len(heap) < k):
                heap.append(representation)
                heap.sort(key=lambda x: x.similarity, reverse=False)
            elif (d <= heap[-1].similarity):
                # remove o ultimo substituindo-o com a representacao atual
                if distance == 'cosine' or distance == 'euclidean' or distance == 'braycurtis':
                    del heap[-1]
                    heap.append(representation)
                    heap.sort(key=lambda x: x.similarity, reverse=False)
                else:
                    # só fazer o processo anterior se a distancia for menor tanto cosine quanto a braycurtis
                    if('braycurtis' in distance):
                        d = braycurtis(img_representation, img2_representation)
                    elif('euclidean' in distance):
                        d = euclidean(img_representation, img2_representation)   
                        
                        
                    representation.similarity = d
                    if (d <= heap[-1].similarity):
                        del heap[-1]
                        heap.append(representation)
                        heap.sort(key=lambda x: x.similarity, reverse=False)
         
                    
        #end = time.time()
        #print(f"Tempo de busca para distancia {distance}:", end - start)
        # retorna uma lista de tamanho k, onde a primeira eh a mais similar e a k-esima a menos similar
        return heap
