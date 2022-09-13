# -*- coding: utf-8 -*-
"""
Created on Wed 11 nov 2019

@author: gabriel antonio carneiro
script de avaliação dos resultados buscados utilizando como base precision e recall 
"""
from DataBase import DataBase
import os
import numpy as np
import glob
import collections
import pandas as pd
import time
import sys


#diretorios de interesse
path = sys.argv[1]
imgs_for_search_path = sys.argv[2]
resultado = sys.argv[3]

quantity_image_per_class = None

#recebe quantidade de elementos por classe 
if len(sys.argv) > 4:
    quantity_image_per_class = int(sys.argv[4])

#nome das represesentacoes no json
name_representations = ['pspotter', 'inception', 'vgg16', 'vgg16ft']

#quantidade de imagens de cada classe presentes no banco de dados

#numero de imagens na pasta - 1, por conta que tem que retirar a propria imagem que está sendo buscada
quantities = {'e':868, 'g':758, 'h':506, 'n':464} #ja que exclui a propria imagem, ficam 47 no bd

def quantidade_de_classes(path):
    

def img_class(path, mode = 'letter'):
    '''
    Retorna a classe da imagem para imagens que possuem no nome a classe representadada pela
    primeira letra ou pela pasta. 
    
    ex: 'imagens/k32.jpg'
    se o modo for letra, 'k' é retornado
    se o modo for pasta, 'imagem' é retornado
    '''
    
    if mode == 'letter':
        return path.split(os.path.sep)[-1][0]
    elif mode == 'folder':
        return path.split(os.path.sep)[-2]
    
    

def img_representation_to_dict(rep, distance):
    representation = {}
    representation['name'] = rep.name
    representation['metric'] = distance
    return representation
    


def analysis_query_result(result, metric, model, img_obj_class):
    '''
    Metodo responsavel por processar cada um dos resultados buscados, retornando
    um json com informacoes relevantes entre a imagem objeto e os resultados da busca
    '''
    k = len(result)
    process = {}
    #pega a path de cada imagem que esta no resultado
    process['imgs_path'] = [x.path for x in result]
    #pega a similaridade de cada uma das imagens
    process['imgs_distance'] = [x.similarity for x in result]
    #pega a classe de cada uma das imagens
    process['imgs_classes'] = [img_class(x.path) for x in result]
    #quantidade de itens recuperados que correspondem a classe da imagem
    process['true_positives'] = collections.Counter(process['imgs_classes'])[img_obj_class]
    #calculo do precision
    process['precision'] = process['true_positives']/k
    #calculo revogacao
    process['recall'] = process['true_positives']/quantities[img_obj_class]
    #f1 escore
    
    process['f1'] = 2*(process['precision']*process['recall'])/(process['precision']+process['recall'])
    process['metric'] = metric
    process['model'] = model
    
    return process

def short_analysis_query_result(result, metric, model, img_obj_class, method_get_class = 'letter'):
    '''
    Metodo responsavel por processar cada um dos resultados buscados, retornando
    um json com informacoes relevantes entre a imagem objeto e os resultados da busca
    '''
    k = len(result)
    process = {}
    process['k'] = k
    imgs_class = [img_class(x.path, method_get_class) for x in result]
    #quantidade de itens recuperados que correspondem a classe da imagem
    process['true_positive'] = collections.Counter(imgs_class)[img_obj_class]
    #calculo do precision
    process['precision'] = process['true_positive']/k
    #calculo revogacao
    process['recall'] = process['true_positive']/quantities[img_obj_class]
    #f1 escore
    #process['f1'] = 2*(process['precision']*process['recall'])/(process['precision']+process['recall'])
    process['metric'] = metric
    process['model'] = model
    
    return process

            
                
#inicializa classe responsável pelos dados
db = DataBase()
#carrega as representacoes
db.load_representations(path, name_representations)




#itera pelas imagens
#acho que nao precisa disso
#os.chdir(imgs_for_search_path)
#filtra por extensao


files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(imgs_for_search_path):
    for file in f:
            files.append(os.path.join(r, file))
            
files = np.sort(files)

'''
coisas que eu tenho que saber com o resultado da busca:
   - precision
   - recall 
   - f1 escore 
   - quantidades de elementos da mesma classe que a buscada
   - manter o k padrao
'''
columns_df = ['img', 'class','k', 'ps_cos_precision', 'ps_cos_recall', 'ps_cos_tp', 'vgg_cos_precision', 'vgg_cos_recall', 'vgg_cos_tp', 'inc_cos_precision', 'inc_cos_recall', 'inc_cos_tp','ps_euc_precision', 'ps_euc_recall', 'ps_euc_tp', 'vgg_euc_precision', 'vgg_euc_recall', 'vgg_euc_tp', 'inc_euc_precision', 'inc_euc_recall', 'inc_euc_tp', 'vggft_cos_precision', 'vggft_cos_recall', 'vggft_cos_tp', 'vggft_euc_precision', 'vggft_euc_recall', 'vggft_euc_tp']
df = pd.DataFrame(columns=columns_df)
results = []
k = 20

times = []
for file in f:
    start = time.time()
    result_ps_cosine = db.search(file, model=db.PSPOTTER, k=20, distance='cosine')
    end = time.time()
    
    times.append(end - start)
    

def evaluate (files, k, df, class_method = 'letter'):
    if class_method == 'letter':
        for file in files:
        
            analysis = list()
            image_object_class = img_class(file)
            aux = {}
            aux['img'] = file
            aux['class'] = image_object_class
            aux['k'] = k
            
            result_ps_cosine = db.search(file, model=db.PSPOTTER, k=k+1, distance='cosine')
            del result_ps_cosine[0] #deleta a própria imagem
            an = short_analysis_query_result(result_ps_cosine, 'cosine', 'pspotter', image_object_class)
            aux['ps_cos_precision'] = an['precision']
            aux['ps_cos_recall'] = an['recall']
            aux['ps_cos_tp'] = an['true_positive']
            
            analysis.append(an)
        
            result_ps_euclidian = db.search(file, model=db.PSPOTTER, k=k+1, distance='euclidian')
            del result_ps_euclidian [0]
            an = short_analysis_query_result(result_ps_euclidian, 'euclidian', 'pspotter', image_object_class)
            aux['ps_euc_precision'] = an['precision']
            aux['ps_euc_recall'] = an['recall']
            aux['ps_euc_tp'] = an['true_positive']
            
            analysis.append(an)
            
            result_vgg_cosine = db.search(file, model=db.VGG, k=k+1, distance='cosine')
            del result_vgg_cosine[0]
            an = short_analysis_query_result(result_vgg_cosine, 'cosine', 'vgg', image_object_class)
            aux['vgg_cos_precision'] = an['precision']
            aux['vgg_cos_recall'] = an['recall']
            aux['vgg_cos_tp'] = an['true_positive']
            
            analysis.append(an)
            
            result_vgg_euclidian = db.search(file, model=db.VGG, k=k+1, distance='euclidian')
            del result_vgg_euclidian[0]
            an = short_analysis_query_result(result_vgg_euclidian, 'euclidian', 'vgg', image_object_class)
            aux['vgg_euc_precision'] = an['precision']
            aux['vgg_euc_recall'] = an['recall']
            aux['vgg_euc_tp'] = an['true_positive']
            analysis.append(an)
            
            result_inc_cosine = db.search(file, model=db.INCEPTION, k=k+1, distance='cosine')
            del result_inc_cosine[0]
            an = short_analysis_query_result(result_inc_cosine, 'cosine', 'inception', image_object_class)
            aux['inc_cos_precision'] = an['precision']
            aux['inc_cos_recall'] = an['recall']
            aux['inc_cos_tp'] = an['true_positive']
            analysis.append(an).split(os.path.sep)[-2]
            
            result_inc_euclidian = db.search(file, model=db.INCEPTION, k=k+1, distance='euclidian')
            del result_inc_euclidian[0]
            an = short_analysis_query_result(result_inc_euclidian, 'euclidian', 'inception', image_object_class)
            aux['inc_euc_precision'] = an['precision']
            aux['inc_euc_recall'] = an['recall']
            aux['inc_euc_tp'] = an['true_positive']
            analysis.append(an)
            
            result_vgg_cosine = db.search(file, model=db.VGGFT, k=k+1, distance='cosine')
            del result_vgg_cosine[0]
            an = short_analysis_query_result(result_vgg_cosine, 'cosine', 'vggft', image_object_class)
            aux['vggft_cos_precision'] = an['precision']
            aux['vggft_cos_recall'] = an['recall']
            aux['vggft_cos_tp'] = an['true_positive']
            
            analysis.append(an)
            
            result_vgg_euclidian = db.search(file, model=db.VGGFT, k=k+1, distance='euclidian')
            del result_vgg_euclidian[0]
            an = short_analysis_query_result(result_vgg_euclidian, 'euclidian', 'vggft', image_object_class)
            aux['vggft_euc_precision'] = an['precision']
            aux['vggft_euc_recall'] = an['recall']
            aux['vggft_euc_tp'] = an['true_positive']
            analysis.append(an)
           
                
            result = {}
            result['image_object'] = file
            result['class_image_object'] = img_class(file)
            result['analysis'] = analysis
            results.append(result)
            df = df.append(aux, ignore_index = True)
    else:
        for file in files:
            analysis = list()
            image_object_class = img_class(file, 'folder')
            aux = {}
            aux['img'] = file
            aux['class'] = image_object_class
            aux['k'] = k
            
            result_ps_cosine = db.search(file, model=db.PSPOTTER, k=k+1, distance='cosine')
            del result_ps_cosine[0] #deleta a própria imagem
            an = short_analysis_query_result(result_ps_cosine, 'cosine', 'pspotter', image_object_class, 'folder')
            aux['ps_cos_precision'] = an['precision']
            aux['ps_cos_recall'] = an['recall']
            aux['ps_cos_tp'] = an['true_positive']
            
            analysis.append(an)
        
            result_ps_euclidian = db.search(file, model=db.PSPOTTER, k=k+1, distance='euclidian')
            del result_ps_euclidian [0]
            an = short_analysis_query_result(result_ps_euclidian, 'euclidian', 'pspotter', image_object_class, 'folder')
            aux['ps_euc_precision'] = an['precision']
            aux['ps_euc_recall'] = an['recall']
            aux['ps_euc_tp'] = an['true_positive']
            
            analysis.append(an)
            
            result_vgg_cosine = db.search(file, model=db.VGG, k=k+1, distance='cosine')
            del result_vgg_cosine[0]
            an = short_analysis_query_result(result_vgg_cosine, 'cosine', 'vgg', image_object_class, 'folder')
            aux['vgg_cos_precision'] = an['precision']
            aux['vgg_cos_recall'] = an['recall']
            aux['vgg_cos_tp'] = an['true_positive']
            
            analysis.append(an)
            
            result_vgg_euclidian = db.search(file, model=db.VGG, k=k+1, distance='euclidian')
            del result_vgg_euclidian[0]
            an = short_analysis_query_result(result_vgg_euclidian, 'euclidian', 'vgg', image_object_class, 'folder')
            aux['vgg_euc_precision'] = an['precision']
            aux['vgg_euc_recall'] = an['recall']
            aux['vgg_euc_tp'] = an['true_positive']
            analysis.append(an)
            
            result_inc_cosine = db.search(file, model=db.INCEPTION, k=k+1, distance='cosine')
            del result_inc_cosine[0]
            an = short_analysis_query_result(result_inc_cosine, 'cosine', 'inception', image_object_class, 'folder')
            aux['inc_cos_precision'] = an['precision']
            aux['inc_cos_recall'] = an['recall']
            aux['inc_cos_tp'] = an['true_positive']
            analysis.append(an)
            
            result_inc_euclidian = db.search(file, model=db.INCEPTION, k=k+1, distance='euclidian')
            del result_inc_euclidian[0]
            an = short_analysis_query_result(result_inc_euclidian, 'euclidian', 'inception', image_object_class, 'folder')
            aux['inc_euc_precision'] = an['precision']
            aux['inc_euc_recall'] = an['recall']
            aux['inc_euc_tp'] = an['true_positive']
            analysis.append(an)
            
            result_vgg_cosine = db.search(file, model=db.VGGFT, k=k+1, distance='cosine')
            del result_vgg_cosine[0]
            an = short_analysis_query_result(result_vgg_cosine, 'cosine', 'vggft', image_object_class, 'folder')
            aux['vggft_cos_precision'] = an['precision']
            aux['vggft_cos_recall'] = an['recall']
            aux['vggft_cos_tp'] = an['true_positive']
            
            analysis.append(an)
            
            result_vgg_euclidian = db.search(file, model=db.VGGFT, k=k+1, distance='euclidian')
            del result_vgg_euclidian[0]
            an = short_analysis_query_result(result_vgg_euclidian, 'euclidian', 'vggft', image_object_class, 'folder')
            aux['vggft_euc_precision'] = an['precision']
            aux['vggft_euc_recall'] = an['recall']
            aux['vggft_euc_tp'] = an['true_positive']
            analysis.append(an)
           
                
            result = {}
            result['image_object'] = file
            result['class_image_object'] = img_class(file)
            result['analysis'] = analysis
            results.append(result)
            df = df.append(aux, ignore_index = True)
            
    
import json
from os.path import expanduser
home = expanduser("~")
with open(home+'/results.json', 'w') as f:
    json.dump(results, f)
    
df.to_csv(home+'/results.csv', index=False, sep=';')

import string
letters = string.ascii_uppercase
letters = list(letters)

#auxiliar que imprime médias
for i in range(20):
    df_aux = df[df['class'] == letters[i]]
    
    print("classe: %s" %(letters[i]))
    print("precision média pspotter: %f" %(np.mean(df_aux['ps_cos_precision'])))
    print("tp média pspotter: \t%f" %(np.mean(df_aux['ps_cos_tp'])))
    
    print("precision média vgg: \t%f" %(np.mean(df_aux['vgg_cos_precision'])))
    print("tp média vgg \t\t%f" %(np.mean(df_aux['vgg_cos_tp'])))
    
    print("precision média inc: \t%f" %(np.mean(df_aux['inc_cos_precision'])))
    print("tp média inc \t\t%f" %(np.mean(df_aux['inc_cos_tp'])))
    
    print("=======================================")