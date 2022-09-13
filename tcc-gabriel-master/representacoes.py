# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:48:07 2019

@author: gabriel
Modo de usar:
python3 representacoes.py /home/gabriel/Downloads/dataset-recortado /home/gabriel/reprsentacao_7.json

"""

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
import keras
import numpy as np
import os
import glob
import json
import cv2
import sys

#constantes 
#------------
VGG = 0
INCEPTION = 1
PSPOTTER = 2
number_checkpoint = 100
#------------

#diretorios
#------------
main_path='/home/gabriel/'
pspotter_model = 'pspotter.h5'
vgg16ft_path = 'vgg16ft.h5'
nome_log = '.log'
path_checklist = main_path+'.checklist.txt'
#------------
#estrutura para checklist


#------------modelos--------------
vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max')
inception = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', pooling='max')

aux = keras.models.load_model(pspotter_model)
pspotter = Model(inputs = aux.inputs, outputs=aux.get_layer('flatten_1').output)

aux2 = keras.models.load_model(vgg16ft_path)
vgg16ft = Model(inputs= aux2.inputs, outputs = aux2.get_layer('global_max_pooling2d_1').output)
 

#---------------------------------
def img_resize(img, target):
    
    '''
    Metodo utulizado para redimensionar imagens, sem manter aspecto.
    Metodo recebe o modelo a que se destina o redmensionamento.
    '''
    
    if target == INCEPTION:
        dim=(299, 299)
    elif target == VGG16 or target == PSPOTTER:
        dim=(224, 224)

    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    resized = resized.astype('float32')
    resized = resized/255.0
    
    return resized
#----------------constantes---------------------------
path = sys.argv[1]
path_bd = sys.argv[2]

checklist=None
if os.path.isfile(path_checklist):
    file_checklist = open(path_checklist, 'r')
    checklist = int(file_checklist.read())
    file_checklist.close()
    file_bd = open(path_bd, 'r')
    bd = json.loads(file_bd.readline())
    file_bd.close()
    
else:
    checklist = 0
    representacoes={}
    bd=list()
    media_similaridade={}



##itera pelas imagens
#os.chdir(path)
##mudar o tipo de imagem
#files = np.array(glob.glob("*.tif"))
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
            files.append(os.path.join(r, file))
            
files = np.sort(files)
files = files[(checklist*number_checkpoint):] #iniciar a partir de onde parou
for i, image_name in enumerate(files):
    size = len(bd)
    if i==0:
        if size == 0:
            print('iniciando o programa, nenhum checkpoint encontrado')
        else:
            aux = len(files)
            name = bd[-1]['path'].split('/')[-1]
            print('iniciando o programa com checkpoint, ultima imagem salva: %s'%(name))
            print('faltando %d imagens para computar..'%(aux))
            print('iniciando com imagem %s'%(image_name))
        
    
    img_path = image_name
    
    #pre processamento para inception
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.inception_resnet_v2.preprocess_input(x)
    
    inception_feature = inception.predict(x)
    inception_feature = np.array(inception_feature).flatten()
    
   
    #pre processamento vgg16
    img=image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)
    
    #parte específica para cada modelo de vgg
    vgg16_feature = vgg16.predict(x)
    vgg16_feature = np.array(vgg16_feature).flatten()
    
    vgg16ft_feature = vgg16ft.predict(x)
    vgg16ft_feature = np.array(vgg16ft_feature).flatten()
    
    #pre processamento para a patho spotter
    img = cv2.imread(img_path)
    img = img_resize(img, VGG16)
    img_data = np.expand_dims(img, axis=0)
    pspotter_feature = pspotter.predict(img_data)
    pspotter_feature = np.array(pspotter_feature).flatten()
    
    
    obj = {}
    obj['path']=img_path
    obj['name']=image_name
    obj['vgg16']=vgg16_feature.tolist()
    obj['pspotter'] = pspotter_feature.tolist()
    obj['inception'] = inception_feature.tolist()
    obj['vgg16ft'] = vgg16ft_feature.tolist()
    
    bd.append(obj)
    size = len(bd)
    
    #checkpoint
    if size%number_checkpoint == 0 and size !=0:
        ch = int(size/number_checkpoint)
        print('checkpoint nº %d, bd com  %d imagens, ultima imagem: %s' % (ch, size, image_name))
        print(len(bd))
        bd_file = open(path_bd, 'w')
        dumps = json.dumps(bd)
        bd_file.write(dumps)
        bd_file.close()
        file_checklist = open(path_checklist, 'w')
        file_checklist.write(str(ch))
        file_checklist.close()

bd_file = open(path_bd, 'w')
dumps = json.dumps(bd)
bd_file.write(dumps)
bd_file.close()
if os.path.isfile(path_checklist):
    os.remove(path_checklist)        
        
    



