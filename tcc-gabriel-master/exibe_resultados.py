#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:39:36 2020

@author: gabriel
"""

import pandas as pd
import numpy as np 
import sys


arquivo_csv = sys.argv[1]
resultado_csv = sys.argv[2]

df = pd.read_csv(arquivo_csv, sep=';')

valores_k = np.unique(df.k.values)
classes = np.unique(df['class'].values)
distancias = {'cos':'cosseno', 'euc':'euclidiana'}
modelos = ['ps', 'vgg', 'inc', 'vggft']
medidas = ['tp', 'precision', 'recall']
columns_df = ['class','k', 'ps_cos_precision', 'ps_cos_recall', 'ps_cos_tp', 'vgg_cos_precision', 'vgg_cos_recall', 'vgg_cos_tp', 'inc_cos_precision', 'inc_cos_recall', 'inc_cos_tp','ps_euc_precision', 'ps_euc_recall', 'ps_euc_tp', 'vgg_euc_precision', 'vgg_euc_recall', 'vgg_euc_tp', 'inc_euc_precision', 'inc_euc_recall', 'inc_euc_tp', 'vggft_cos_precision', 'vggft_cos_recall', 'vggft_cos_tp', 'vggft_euc_precision', 'vggft_euc_recall', 'vggft_euc_tp']
df_result = pd.DataFrame(columns=columns_df)


for classe in classes:
    
    df_classe = df[df['class'] == classe]
    print('=> CLASSE: {}'.format(classe))
    
    for k in valores_k:
        dicionario = {}
        df_classe_k = df_classe[df_classe['k'] == k]
        dicionario['k'] = k
        dicionario['class'] = classe
        print('\t=>Resultados para K = {}:'.format(k))
        for distancia in list(distancias.keys()):
            print('\t\t=> Distância {}:'.format(distancias[distancia]))
            for modelo in modelos:
                print('\t\t\t=> Modelo:{}'.format(modelo))
                for medida in medidas:
                    valor = np.average(df_classe_k['{}_{}_{}'.format(modelo, distancia, medida)].values)
                    #valor = round(valor, 2)
                    print('\t\t\t\t=>média de {}: {}'.format(medida, valor))
                    
                    dicionario['{}_{}_{}'.format(modelo, distancia, medida)] = valor
                    valor = np.std(df_classe_k['{}_{}_{}'.format(modelo, distancia, medida)].values)
                    
                    print('\t\t\t\t=>desvio padrao de {}: {}'.format(medida, valor))
        df_result = df_result.append(dicionario, ignore_index=True)




print('\n\n\n=> Média total')
for k in valores_k:
    dicionario = {}
    dicionario['class'] = 'total'
    df_k = df[df['k'] == k]
    dicionario['k'] = k
    print('\t=> Para K = {}'.format(k))
    for distancia in list(distancias.keys()):
        print('\t\t=> Distância {}:'.format(distancias[distancia]))
        for modelo in modelos:
            print('\t\t\t=> Modelo:{}'.format(modelo))
            for medida in medidas:
                valor = np.average(df_k['{}_{}_{}'.format(modelo, distancia, medida)].values)
                #valor = round(valor, 2)
                dicionario['{}_{}_{}'.format(modelo, distancia, medida)] = valor
                print('\t\t\t\t=>média de {}: {}'.format(medida, valor))
                
                valor = np.std(df_k['{}_{}_{}'.format(modelo, distancia, medida)].values)
            
                print('\t\t\t\t=>desvio padrao de {}: {}'.format(medida, valor))

    df_result = df_result.append(dicionario, ignore_index=True)
    

#tratamento para porcentagem
for distancia in list(distancias.keys()):
    for modelo in modelos:
        for medida in ['precision', 'recall']:
            df_result['{}_{}_{}'.format(modelo, distancia, medida)] = round(df_result['{}_{}_{}'.format(modelo, distancia, medida)]*100,2)
    
df_result.to_csv(resultado_csv+'.csv', index=False, sep=';')
print('csv salvo em {}.csv'.format(resultado_csv))