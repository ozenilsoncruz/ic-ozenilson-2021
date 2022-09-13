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

import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtGui import QGuiApplication
from scipy.spatial.distance import cosine, euclidean
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from ImageRepresentation import ImageRepresentation


class DataBase:
    # constantes
    VGG = 0
    INCEPTION = 1
    PSPOTTER = 2
    VGGFT = 3
    REPRESENTATONS = ['vgg16', 'inception', 'pspotter', 'vgg16ft']

    # patho-spotter
    pspotter_model = 'pspotter.h5'
    # vgg16
    vgg16ft_model = 'vgg16ft.h5'

    # banco de dados
    images = None

    # modelos
    __vgg__ = VGG16(weights='imagenet', include_top=False, pooling='max')
    __inception__ = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                             pooling='max')
    __aux__ = tf.keras.models.load_model(pspotter_model)
    __pspotter__ = Model(inputs=__aux__.inputs,
                         outputs=__aux__.get_layer('flatten_1').output)
    __aux1__ = tf.keras.models.load_model(vgg16ft_model)
    __vggft__ = Model(inputs=__aux1__.inputs, outputs=__aux1__.get_layer(
        'global_max_pooling2d_1').output)
    processStatus = 0

    # metodos python

    def __len__(self):
        return len(self.images)

    # --------------------------------------------------------

    def __image_preparation__(self, img_path, model):
        '''
        Metodo responsavel pelo pre processamento da imagem: leitura, redimensionamento, normalizacao
        '''

        if model == self.VGG or model == self.VGGFT:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.vgg16.preprocess_input(x)

            return x
        elif model == self.PSPOTTER:
            # carrega a imagem
            # str = "r"
            img = cv2.imread(img_path)
            # redimensiona a imagem
            img = self.__img_resize__(img, model)
            # poe num vetor de uma unica dimensao
            img_data = np.expand_dims(img, axis=0)
            # retorna a imagem ja pronta para predicao
            return img_data

        elif model == self.INCEPTION:
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.inception_resnet_v2.preprocess_input(x)
            return x

    def __get_image_representation(self, img_data, model):
        '''
        Retorna a imagem aplicada a uma rede neural
        '''
        model = self.__get_model__(model)
        print("O vetor é: ")
        print(np.array(model.predict(img_data)).flatten().tolist())
        print("Final.")
        return np.array(model.predict(img_data)).flatten().tolist()

    def __get_model__(self, num_model):
        '''
        Retorna o um modelo de rede neural
        '''
        # retorna o modelo vgg
        if num_model == self.VGG:
            return self.__vgg__

        # retorna o modelo inception
        elif num_model == self.INCEPTION:
            return self.__inception__

        # retorna o modelo pspotter
        elif num_model == self.PSPOTTER:

            return self.__pspotter__
        # retorna o vgg com finetuning
        elif num_model == self.VGGFT:
            return self.__vggft__

    def __img_resize__(self, img, target):
        '''
        Metodo utulizado para redimensionar imagens, sem manter aspecto.
        Metodo recebe o modelo a que se destina o redmensionamento.
        '''
        if target == self.INCEPTION:
            dim = (299, 299)
        elif (target == self.VGG or target == self.PSPOTTER):
            dim = (224, 224)

        # redimensiona a imagem com o modo bilienar
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        # muda o tipo do vetor para melhorar o processemanto
        resized = resized.astype('float32')

        # normaliza a imagem com valores entre 0 e 1
        resized = resized / 255.0

        return resized

    def load_representations(self, json_path, names_representations):
        '''
        Metodo responsavel por carregar representacoes de um arquivo json.

        json: path para arquivo json contendo as representacoes

        names_representations: lista com nome nome de representacoes presentes
        no arquivo json.
        '''

        images = list()

        with open(json_path) as arquivo:
            bd = json.load(arquivo)

        for img in bd:
            rep = None
            rep = ImageRepresentation(name=img['name'], image_path=img['path'])
            for s in names_representations:
                rep.add_representation(
                    name_representation=s, representation=img[s])
            images.append(rep)
        self.images = images
        return images

    def search(self, img_path, model=2, k=10, distance='cosine'):
        '''
        Metodo responsavel por realizar uma busca.

        img_path: caminho para imagem objeto da busca
        model: numero do modelo a ser usado, padrao eh o pspotter k,
        realizando uma busca utilizand o ps-k search
        k: quantidades de imagens e serem buscadas
        retorno: k imagens mais similares à buscada
        '''

        heap = []
        img_data = self.__image_preparation__(img_path, model)
        img_representation = self.__get_image_representation(img_data, model)

        for img2 in self.images:
            img2_representation = img2.representations[self.REPRESENTATONS[model]]
            representation = copy.deepcopy(img2)

            # calcula a distancia
            if (distance == 'cosine'):
                d = cosine(img_representation, img2_representation)
            else:
                d = euclidean(img_representation, img2_representation)

            # seta similaridade no objeto de representacao
            representation.similarity = d
            # povoa a lista inicialmente
            if (len(heap) < k):
                heap.append(representation)
                heap.sort(key=lambda x: x.similarity, reverse=False)
            elif (d <= heap[-1].similarity):
                # remove o ultimo substituindo-o com a representacao atual
                del heap[-1]
                # remove o ultimo substituindo-o com a representacao atual
                heap.append(representation)
                heap.sort(key=lambda x: x.similarity, reverse=False)
        # retorna uma lista de tamanho k, onde a primeira eh a mais similar e a k-esima a menos similar
        return heap

    # metodo ta errado, tem que testar e fazer direito
    def generate_representations(self, bar, path_in, path_out):
        QGuiApplication.processEvents()
        images = []
        path = path_in
        count = 0
        # if os.path.isfile(path_out):
        #    old_images = dict(self.images)
        # else:
        with open(path_out, 'w') as file_out:
            file_out.write("")
            file_out.close()
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            # for file in tqdm_gui(f):
            bar.setMaximum(int(len(f)))
            for file in f:
                file_ = file.lower()
                if (('.jpg' in file_) or ('.png' in file_) or ('.jpeg' in file_) or ('.tif' in file_)):
                    obj = {}
                    img_path = os.path.join(r, file)
                    obj['path'] = img_path
                    obj['name'] = file

                    for i, representation in enumerate(self.REPRESENTATONS):
                        img_data = self.__image_preparation__(img_path, model=i)
                        img_representation = self.__get_image_representation(img_data, model=i)
                        obj[representation] = img_representation
                    images.append(obj)
                    count += 1
                    bar.setValue(count)
                    QGuiApplication.processEvents()

        with open(path_out, 'w') as bd_file:
            ''''
            if self.images is None:
                print('Self.images é nulo. Path_out: %s' %path_out)
                dumps = json.dumps(images)
            else:
                print('Concatenando. Path_out: %s' % path_out)
                for i in images:
                    old_images.append(i)
                dumps = json.dumps(old_images)
            '''''
            dumps = json.dumps(images)
            bd_file.write(dumps)
            bd_file.close()
