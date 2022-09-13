import collections, json, os, time

import numpy as np
import pandas as pd

from dataBase import DataBase



def img_class(path):
    '''
    Retorna a classe da imagem pelo caminho. 
    '''
    
    caminho = path.split(os.path.sep)
    for c in caminho:
        if c in classes:
            return c
    return ''


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
    process['recall'] = process['true_positives']/(quantities[img_obj_class]-1)
    #f1 escore
    
    process['f1'] = 2*(process['precision']*process['recall'])/(process['precision']+process['recall'])
    process['metric'] = metric
    process['model'] = model
    
    return process


def short_analysis_query_result(result, metric, model, img_obj_class):
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
    process['recall'] = process['true_positive']/(quantities[img_obj_class]-1)
    #f1 escore
    #process['f1'] = 2*(process['precision']*process['recall'])/(process['precision']+process['recall'])
    process['metric'] = metric
    process['model'] = model
    
    return process


def set_class_quantities(path):
    '''
    Metodo que define a quantidade de itens de cada classe considerando as subpastas
    '''
    quantities= {}
    
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            diretorio = os.path.join(root, name)
            dpath, ddirs, dfiles = next(os.walk(diretorio))
            sub = root.replace(path, '').split('/')
            if len(sub) > 1:
                if quantities.get(sub[1]):
                    quantities[sub[1]] += len(dfiles)
                else:
                    quantities[sub[1]] = len(dfiles)
            else:
                if quantities.get(name):
                    quantities[name] += len(dfiles)
                else:
                    quantities[name] = len(dfiles)
            
    return quantities


def evaluate(file, k, df, name_representation, distance_metrics):
    """Realiza as buscas e computa os resultados"""
    analysis = list()
    
    path = file
    if type(file) != str:
        path = file.path
        
    image_object_class = img_class(path)
    aux = {}
    aux['img'] = path
    aux['class'] = image_object_class
    aux['k'] = k

    for name in name_representation:
        for distance in distance_metrics:
            search_result = db.search(file, model=db.REPRESENTATONS.index(name), k=k+1, distance=distance)
            del search_result[0] #deleta a própria imagem      
            msg = [img_class(x.path) for x in search_result]
            #print(collections.Counter(msg)[image_object_class])
            an = short_analysis_query_result(search_result, distance, name, image_object_class)
            aux[name+'_'+distance+'_'+'precision'] = an['precision']
            aux[name+'_'+distance+'_'+'tp'] = an['true_positive']

            analysis.append(an)

    #ediciona os resultados na lista de resultados
    result = {}
    result['image_object'] = path
    result['class_image_object'] = image_object_class
    result['analysis'] = analysis
    results.append(result)

    return pd.DataFrame([aux])


def salvar_resultados(path, df, x):
    """Salva os resultados em arquivos no formato json e csv"""
    with open(path+f'/resultados/json/resultado{x}.json', 'w') as f:
        json.dump(results, f)
        f.close()
    print('json salvo em {}'.format(path+'/resultados/json/'))

    df.to_csv(path+f'/resultados/csv/resultado{x}.csv', index=False, sep=';')
    print('csv salvo em {}.csv'.format(path+'/resultados/csv/'))


def exibir_medias(df):
    """Imprime as medias dos resultados obtidos nas buscas"""
    for classe in classes: 
        df_aux = df[df['class'] == classe]
        print("Classe: %s" %(classe))
        print("==============================================")
        for distance in distance_metrics:
            print('Metrica de distancia: ', distance)
            print('----------------------------------------------')
            for name in name_representation:
                print('Arquitetura: ', name)
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print(f"Precision média: {np.mean(df_aux[name+'_'+distance+'_'+'precision'])*100:.2f}%")
                print(f"Tp média: {np.mean(df_aux[name+'_'+distance+'_'+'tp']):.2f}")
                #print(f"Recall média: {np.mean(df_aux[name+'_'+distance+'_'+'recall']):.2f}")
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('----------------------------------------------')
        print("==============================================\n\n")


def images_search(path, qtd_imgs):
    """Busca uma determinada quantidade de imagens"""
    images_path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            if qtd_imgs == 0:
                break
            images_path.append(os.path.join(root, f))
            qtd_imgs -= 1
    return images_path


def search_test(db, k, name_representation, distance_metrics):
    """Faz a busca entre de k imagens para todas as imagens do banco de dados"""
    df = pd.DataFrame()
    
    for image in db.images:
        df = pd.concat([df, evaluate(file=image, k=k, df=df, 
                        name_representation=name_representation, 
                        distance_metrics=distance_metrics)])
    return df



#diretorios de interesse
path_atual = os.getcwd()
path_models = path_atual.replace('/'+path_atual.split('/')[-1], '')+'/modelos/save_models/'
path_db = path_atual.replace('/'+path_atual.split('/')[-1], '').replace('/'+path_atual.split('/')[-2], '')+'/db/db2'

#nome das CNNs
name_representation = ['vgg16']

#metricas de distancia
distance_metrics = ['braycurtis', 'cosine']

#numero de imagens, por conta que tem que retirar a propria imagem que está sendo buscada
quantities = set_class_quantities(path_db)

#classes
classes = list(quantities.keys())

#variaveis auxiliares
results = []
k = 20

#inicializa classe responsável pelos dados
db = DataBase()
db.iniciar_modelos(path_models)

#carrega as imagens de uma pasta
db.load_representationsDB(path_db)

start = time.time() 
print('Iniciando busca...')
db.load_representationsJson(path_db+'.json', name_representation)
df1 = search_test(db, k, name_representation, distance_metrics)   
print(f"Busca encerrada em:  {(time.time() - start):.2f} segundos")

exibir_medias(df1)
salvar_resultados(path_atual, df1, 'kimia_test')

print('Processo Finalizado!')
