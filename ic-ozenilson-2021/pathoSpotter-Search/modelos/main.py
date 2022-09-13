import os
import time
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import vgg16, vgg19, inception_v3, resnet_v2, xception
from keras.callbacks import (EarlyStopping, ModelCheckpoint)
from keras.layers import (Dense, GlobalAveragePooling2D, GlobalMaxPool2D, Input)
from keras.models import Model
from keras.optimizers import adam_v2
from keras.utils import np_utils
from PIL import Image




print("Tensorflow: ",tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def set_class_quantzities(path):
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


def model_x(model):
    model.trainable = False
    
    x = model.output
    x = GlobalMaxPool2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(24, activation='softmax')(x)
    
    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=adam_v2.Adam(learning_rate=0.00001, decay=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.AUC()])
    
    return model


def load_qtdImages(path, classes, dim, qtd_imgs):
    X = []
    Y = []
    processed_image_count = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            diretorio = os.path.join(root, name)
            dpath, ddirs, dfiles = next(os.walk(diretorio))
            for i in range(qtd_imgs):
                if len(dfiles) <= i:
                    break
                file_path = os.path.join(diretorio, dfiles[i])
                suffix = file_path[len(path):].lstrip(os.sep)
                label = suffix.split(os.sep)[0]

                img = Image.open(file_path)
                img = img.resize(dim)
                img = img.convert('RGB')
                img = np.asarray(img)

                img = img/255.

                X.append(img)
                Y.append(classes.index(label))
                processed_image_count += 1

    print (f"Imagens processadas: {processed_image_count}") 
    
    X = np.array(X, dtype='float32')
    Y = np.array(Y) #Atrinui o número de classes a Y
    
    return X, Y


def load_dataset(base_dir, classes, dim):
    X = []
    Y = []
    processed_image_count = 0
    for root, subdirs, files in os.walk(base_dir):
        for d in subdirs:
            classes.append(d)
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir) # verifica nesse caso se o caminho passado contém a string base_dir(caminho da base de dados) no início
            suffix = file_path[len(base_dir):].lstrip(os.sep)
            label = suffix.split(os.sep)[0]
            
            img = Image.open(file_path)
            img = img.resize(dim)
            img = img.convert('RGB')
            img = np.asarray(img)

            height, width, chan = img.shape # Altura largura e canal de cores
            assert chan == 3 # Caso seja falso a execução é interrompida
            
            img = img/255.
            
            X.append(img)
            Y.append(classes.index(label))
            processed_image_count += 1

    print (f"Imagens processadas: {processed_image_count}")
    
    X = np.array(X, dtype='float32')
    Y = np.array(Y) #Atrinui o número de classes a Y
    
    return X, Y


path_atual = os.getcwd()
path_models = path_atual.replace('/'+path_atual.split('/')[-1], '')+'/modelos/save_models/'
path_db = path_atual.replace('/'+path_atual.split('/')[-1], '').replace('/'+path_atual.split('/')[-2], '')+'/dataset/kimia/'

quantidade = set_class_quantzities(path_db)
classes = list(quantidade.keys())

K = 5
metrics = pd.DataFrame(index=['accuracy', 'loss', 'val_accuracy', 'val_loss', 'precision', 'recall', 'time'])

#inicia as redes do ImageNet
inception_model = inception_v3.InceptionV3(weights='imagenet', 
                                           include_top=False, 
                                           input_tensor=Input(shape=(299, 299, 3)), 
                                           input_shape=(299, 299, 3))
resnet50_model = resnet_v2.ResNet50V2(weights='imagenet', 
                                      include_top=False, 
                                      input_tensor=Input(shape=(299, 299, 3)), 
                                      input_shape=(299, 299, 3))
xception_model = xception.Xception(weights='imagenet', 
                                   include_top=False, 
                                   input_tensor=Input(shape=(299, 299, 3)), 
                                   input_shape=(299, 299,3))
vgg16_model = vgg16.VGG16(weights='imagenet', 
                          include_top=False, 
                          input_tensor=Input(shape=( 224, 224, 3)), 
                          input_shape=( 224, 224,3))

modelos = {'inception': inception_model, 'resnet': resnet50_model, 
           'xception': xception_model, 'vgg16': vgg16_model}


#Carregamento da Base de dados
print("Carregando dataset...")
start = time.time()

#dataset com dimensao 299x299
x_train_299, y_train_299 = load_dataset(path_db+'train', classes, (299, 299))
x_val_299, y_val_299 = load_dataset(path_db+'test', classes, (299, 299))

#dataset com dimensao 224x224
x_train_224, y_train_224 = load_dataset(path_db+'train', classes, (224, 224))
x_val_224, y_val_224 = load_dataset(path_db+'test', classes, (224, 224))

print(f"Dataset carregado em: {time.time()-start:.2f} segundos")

for name, model in modelos.items():
    print(f'Iniciando treinamento do modelo {name}')
    
    #Generate batches from indices
    if 'vgg' not in name:
        x_train, y_train = x_train_299, y_train_299
        x_val, y_val = x_val_299, y_val_299
    else:
        x_train, y_train = x_train_224, y_train_224
        x_val, y_val = x_val_224, y_val_224
        
    y_train = np_utils.to_categorical(y_train, 24)
    y_val = np_utils.to_categorical(y_val, 24)

    
    #Definições de parametros de avaliação
    checkpoint = ModelCheckpoint(path_models+name+'/'+name+'.h5', monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=100, verbose=1, mode='auto', baseline=None)
    callbacks_list = [checkpoint, earlyStopping]
    
    t0 = time.time()
    
    model = model_x(model)
    history = model.fit(x = x_train, 
                        y = y_train,
                        batch_size=64,
                        validation_data=(x_val, y_val),
                        epochs=200,
                        callbacks=callbacks_list)
                        
    ttt = time.time() - t0
    
    accuracy = np.array(history.history['categorical_accuracy'])
    loss = np.array(history.history['loss'])
    val_accuracy = np.array(history.history['val_categorical_accuracy'])
    val_loss = np.array(history.history['val_loss'])
   
    accuracy = sorted(accuracy, key = float)
    val_accuracy = sorted(val_accuracy, key = float)
    
    metrics.loc['time', name] = ttt
    metrics.loc['accuracy', name] = accuracy[-1]
    metrics.loc['loss', name] = loss[-1]
    metrics.loc['val_accuracy', name] = val_accuracy[-1]
    metrics.loc['val_loss', name] = val_loss[-1]
    
    print("Tempo de Treinamento: %.3f" % (ttt))
    print('Ac: ', accuracy[-1])
    print('Acval: ', val_accuracy[-1])
    
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('Acurácia')
    plt.xlabel('Épocas')
    plt.legend(['Treinamento', 'Teste'], loc='upper left')
    plt.savefig(path_models+name+'/'+name+"_accuracy.png")
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Épocas')
    plt.legend(['Treinamento', 'Teste'], loc='upper left')
    plt.savefig(path_models+name+'/'+name+"_loss.png")
    plt.clf()
    
    #salvar dataframe
    metrics.to_csv(path_models+name+'/dados_train.csv', index=False, sep=';')
    
    
print('Finalizidado!')