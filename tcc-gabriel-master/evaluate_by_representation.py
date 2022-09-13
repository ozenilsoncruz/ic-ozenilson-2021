# -*- coding: utf-8 -*-
"""
Created on Wed 11 nov 2019

@author: gabriel antonio carneiro
script de avaliação dos resultados buscados utilizando como base precision e recall 
Modo de uso:
    python3 evaluate_by_representation.py /home/gabriel/representacao_database_misto.json /home/gabriel/resultado_multilesao /home/gabriel/Downloads/dataset-recortado-465

"""
from DataBase import DataBase
import os
import collections
import pandas as pd
import sys
import json

#diretorios de interesse
path = sys.argv[1]
resultado_path = sys.argv[2]
img_path = sys.argv[3]
quantity_image_per_class = None

#recebe quantidade de elementos por classe 
if len(sys.argv) > 4:
    quantity_image_per_class = int(sys.argv[4])

#nome das represesentacoes no json
name_representations = ['pspotter', 'inception', 'vgg16', 'vgg16ft']


def img_class(path):
    '''
    Retorna a classe da imagem para imagens que possuem no nome a classe representadada pela
    primeira letra ou pela pasta. 
    
    ex: 'imagens/k32.jpg'
    se o modo for letra, 'k' é retornado
    se o modo for pasta, 'imagem' é retornado
    '''
    #retura barra do final
    if(path[-1]=='/'):
        path = path[:-1]
        
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
    imgs_class = [img_class(x.path) for x in result]
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


def evaluate (image, k, df):
    print('processando imagem {}'.format(image.path))
    #lista para guardar resultado
    analysis = list()
    
    #obtem classe da imagem
    image_object_class = img_class(image.path)
    
    aux = {}
    aux['img'] = image.path
    aux['class'] = image_object_class
    aux['k'] = k
    
    result_ps_cosine = db.search_by_representation(image.representations['pspotter'], 
                                                          model=db.PSPOTTER, k=k+1, distance='cosine')
    del result_ps_cosine[0] #deleta a própria imagem
    #metodo que analisa os resultados
    an = short_analysis_query_result(result_ps_cosine, 'cosine', 'pspotter', image_object_class)
    
    aux['ps_cos_precision'] = an['precision']
    aux['ps_cos_recall'] = an['recall']
    aux['ps_cos_tp'] = an['true_positive']
    
    analysis.append(an)
    
    

    result_ps_euclidian = db.search_by_representation(image.representations['pspotter'], model=db.PSPOTTER, k=k+1, distance='euclidian')
    del result_ps_euclidian [0]
    an = short_analysis_query_result(result_ps_euclidian, 'euclidian', 'pspotter', image_object_class)
    aux['ps_euc_precision'] = an['precision']
    aux['ps_euc_recall'] = an['recall']
    aux['ps_euc_tp'] = an['true_positive']
    
    analysis.append(an)
    
    result_vgg_cosine = db.search_by_representation(image.representations['vgg16'], model=db.VGG, k=k+1, distance='cosine')
    del result_vgg_cosine[0]
    an = short_analysis_query_result(result_vgg_cosine, 'cosine', 'vgg', image_object_class)
    aux['vgg_cos_precision'] = an['precision']
    aux['vgg_cos_recall'] = an['recall']
    aux['vgg_cos_tp'] = an['true_positive']
    
    analysis.append(an)
    
    result_vgg_euclidian = db.search_by_representation(image.representations['vgg16'], model=db.VGG, k=k+1, distance='euclidian')
    del result_vgg_euclidian[0]
    an = short_analysis_query_result(result_vgg_euclidian, 'euclidian', 'vgg', image_object_class)
    aux['vgg_euc_precision'] = an['precision']
    aux['vgg_euc_recall'] = an['recall']
    aux['vgg_euc_tp'] = an['true_positive']
    analysis.append(an)
    
    result_inc_cosine = db.search_by_representation(image.representations['inception'], model=db.INCEPTION, k=k+1, distance='cosine')
    del result_inc_cosine[0]
    an = short_analysis_query_result(result_inc_cosine, 'cosine', 'inception', image_object_class)
    aux['inc_cos_precision'] = an['precision']
    aux['inc_cos_recall'] = an['recall']
    aux['inc_cos_tp'] = an['true_positive']
    analysis.append(an)
    
    result_inc_euclidian = db.search_by_representation(image.representations['inception'], model=db.INCEPTION, k=k+1, distance='euclidian')
    del result_inc_euclidian[0]
    an = short_analysis_query_result(result_inc_euclidian, 'euclidian', 'inception', image_object_class)
    aux['inc_euc_precision'] = an['precision']
    aux['inc_euc_recall'] = an['recall']
    aux['inc_euc_tp'] = an['true_positive']
    analysis.append(an)
    
    result_vggft_cosine = db.search_by_representation(image.representations['vgg16ft'], model=db.VGGFT, k=k+1, distance='cosine')
    del result_vggft_cosine[0]
    an = short_analysis_query_result(result_vggft_cosine, 'cosine', 'vggft', image_object_class)
    aux['vggft_cos_precision'] = an['precision']
    aux['vggft_cos_recall'] = an['recall']
    aux['vggft_cos_tp'] = an['true_positive']
    
    analysis.append(an)
    
    result_vggft_euclidian = db.search_by_representation(image.representations['vgg16ft'], model=db.VGGFT, k=k+1, distance='euclidian')
    del result_vggft_euclidian[0]
    an = short_analysis_query_result(result_vggft_euclidian, 'euclidian', 'vggft', image_object_class)
    aux['vggft_euc_precision'] = an['precision']
    aux['vggft_euc_recall'] = an['recall']
    aux['vggft_euc_tp'] = an['true_positive']
    analysis.append(an)
   
        
    result = {}
    result['image_object'] = image.path
    result['class_image_object'] = img_class(image.path)
    result['analysis'] = analysis
    results.append(result)
    df = df.append(aux, ignore_index = True)
    
#===============================
    for j in range(1,3):
        k = k-5 #pra fazer pra k = 10, 15
        aux = {}
        aux['img'] = image.path
        aux['class'] = image_object_class
        aux['k'] = k
        
        result_ps_cosine = result_ps_cosine[:k]
        an = short_analysis_query_result(result_ps_cosine, 'cosine', 'pspotter', image_object_class)
        
        aux['ps_cos_precision'] = an['precision']
        aux['ps_cos_recall'] = an['recall']
        aux['ps_cos_tp'] = an['true_positive']
        
        analysis.append(an)
        
        
    
        result_ps_euclidian = result_ps_euclidian[:k]
        an = short_analysis_query_result(result_ps_euclidian, 'euclidian', 'pspotter', image_object_class)
        aux['ps_euc_precision'] = an['precision']
        aux['ps_euc_recall'] = an['recall']
        aux['ps_euc_tp'] = an['true_positive']
        
        analysis.append(an)
        
        result_vgg_cosine = result_vgg_cosine[:k] 
        an = short_analysis_query_result(result_vgg_cosine, 'cosine', 'vgg', image_object_class)
        aux['vgg_cos_precision'] = an['precision']
        aux['vgg_cos_recall'] = an['recall']
        aux['vgg_cos_tp'] = an['true_positive']
        
        analysis.append(an)
        
        result_vgg_euclidian = result_vgg_euclidian[:k]
        an = short_analysis_query_result(result_vgg_euclidian, 'euclidian', 'vgg', image_object_class)
        aux['vgg_euc_precision'] = an['precision']
        aux['vgg_euc_recall'] = an['recall']
        aux['vgg_euc_tp'] = an['true_positive']
        analysis.append(an)
        
        result_inc_cosine = result_inc_cosine[:k]
        an = short_analysis_query_result(result_inc_cosine, 'cosine', 'inception', image_object_class)
        aux['inc_cos_precision'] = an['precision']
        aux['inc_cos_recall'] = an['recall']
        aux['inc_cos_tp'] = an['true_positive']
        analysis.append(an)
        
        result_inc_euclidian = result_inc_euclidian[:k]
        an = short_analysis_query_result(result_inc_euclidian, 'euclidian', 'inception', image_object_class)
        aux['inc_euc_precision'] = an['precision']
        aux['inc_euc_recall'] = an['recall']
        aux['inc_euc_tp'] = an['true_positive']
        analysis.append(an)
        
        result_vggft_cosine = result_vggft_cosine[:k]
        an = short_analysis_query_result(result_vggft_cosine, 'cosine', 'vggft', image_object_class)
        aux['vggft_cos_precision'] = an['precision']
        aux['vggft_cos_recall'] = an['recall']
        aux['vggft_cos_tp'] = an['true_positive']
        
        analysis.append(an)
        
        result_vggft_euclidian = result_vggft_euclidian[:k]
        an = short_analysis_query_result(result_vggft_euclidian, 'euclidian', 'vggft', image_object_class)
        aux['vggft_euc_precision'] = an['precision']
        aux['vggft_euc_recall'] = an['recall']
        aux['vggft_euc_tp'] = an['true_positive']
        analysis.append(an)
       
            
        result = {}
        result['image_object'] = image.path
        result['class_image_object'] = img_class(image.path)
        result['analysis'] = analysis
        results.append(result)
        df = df.append(aux, ignore_index = True)
    return df
    

def set_class_quantities(path):
    '''
    Metodo que deifne a quantidade de itens de cada classe
    '''
    quantities= {}
    
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            diretorio = diretorio = os.path.join(root, name)
            dpath, ddirs, dfiles = next(os.walk(diretorio))
            quantities[name] = len(dfiles)
            
    return quantities
        

#==========================================================#
                
#inicializa classe responsável pelos dados
db = DataBase()
print('inicializando banco de dados..')
#carrega as representacoes
db.load_representations(path, name_representations)
#numero de imagens na pasta - 1, por conta que tem que retirar a propria imagem que está sendo buscada
quantities = set_class_quantities(img_path) #ja que exclui a propria imagem, ficam 47 no bd
#define as coulnas do df
columns_df = ['img', 'class','k', 'ps_cos_precision', 'ps_cos_recall', 'ps_cos_tp', 'vgg_cos_precision', 'vgg_cos_recall', 'vgg_cos_tp', 'inc_cos_precision', 'inc_cos_recall', 'inc_cos_tp','ps_euc_precision', 'ps_euc_recall', 'ps_euc_tp', 'vgg_euc_precision', 'vgg_euc_recall', 'vgg_euc_tp', 'inc_euc_precision', 'inc_euc_recall', 'inc_euc_tp', 'vggft_cos_precision', 'vggft_cos_recall', 'vggft_cos_tp', 'vggft_euc_precision', 'vggft_euc_recall', 'vggft_euc_tp']
df = pd.DataFrame(columns=columns_df)
results = []
k = 20


    
images = db.images
print('iniciando o processameto...')
for image in images:
    df = evaluate(image, k, df)

print('processamento encerrado...')

with open(resultado_path, 'w') as f:
    json.dump(results, f)
    f.close()
print('json salvo em {}'.format(resultado_path))    
df.to_csv(resultado_path+'.csv', index=False, sep=';')
print('csv salvo em {}.csv'.format(resultado_path))
print('terminado!')