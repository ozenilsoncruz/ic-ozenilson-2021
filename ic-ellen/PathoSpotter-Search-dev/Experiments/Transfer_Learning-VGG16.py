import numpy as np
import pandas as pd
import datetime  # Auxilia nos logs
import json
import os
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Camadas: Densa, Pooling Médio
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model  # Classe Model
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
# Para melhor observar os logs, utiliza-se o TensorBoard
from tensorflow.keras.callbacks import TensorBoard

'''
            LEMBRETES
    Cheque se a base de dados onde o cross validation irá operar é a Kimia-Path 80-20.
    Cheque se o arquivo 'training_labels.csv' existe na mesma pasta que o script.
'''

# ------------------------Dados de Configuração--------------------------

# Onde se deve buscar as imagens para o treino
train_image_dir = 'KIMIA_Path_960/train/'
test_image_dir = 'KIMIA_Path_960/validation/'
num_epochs = 1  # 200  # Número de Épocas

# --------------------------Criação de Modelo----------------------------


def create_new_model():
    # Cria o modelo base, pré-treinado da imagenet
    base_model = VGG16(weights='imagenet', include_top=False,
                             input_shape=(299, 299, 3), pooling='avg')
    # Congelando o modelo
    base_model.trainable = False
    # Colocando o modelo em inferência
    x = base_model(base_model.input, training=False)
    # Um classificador denso com várias classes
    predictions = Dense(20, activation='softmax')(x)  # PathoSpotter
    # Instanciando o novo modelo a ser treinado
    model = Model(inputs=base_model.input, outputs=predictions)
    # Mostrando o modelo
    # model.summary()
    return model

# ------------------------Preparação dos K-Folds-------------------------


# Leitura dos dados de treino de um CSV com os caminhos
train_data = pd.read_csv('training_labels-80_20.csv')
test_data = pd.read_csv('test_labels-80_20.csv')
# Pega as classes do CSV
Y = train_data[['label']]

# Instancio um "Stratified" K-Fold, que garante sempre o mesmo percentual de amostras de cada classe
# Número de folds: 5, shuffle as imagens de ordem com seed 7
skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

# --------------------------Data Augmentation----------------------------

# Instancio um Gerador que gera lotes de dados de imagem de tensor com data augmentation em tempo real
# Faixa de graus para rotações aleatórias: 20º; às vezes, vira a imagem horizontalmente; também
# pré-processa a imagem de acordo com os resquisitos da Xception
idg = ImageDataGenerator(rotation_range=20,
                         horizontal_flip=True,
                         fill_mode='nearest',
                         preprocessing_function=preprocess_input)

# --------------------------Cross Validation-----------------------------

# Função auxiliar que retorna o nome do modelo de acordo com seu fold


def get_model_name(k):
    return 'model_'+str(k)+'.h5'


fold_var = 1  # Contador do fold

# As variáveis train_index e val_index são matrizes com os índices
# que a função cross_validation.StratifiedKFold montou
for train_index, val_index in skf.split(np.zeros(len(Y)), Y):
    # Criar modelos em um loop faz com que o estado global consuma uma quantidade cada
    # vez maior de memória ao longo do tempo. Chamar esse método libera o estado global.
    tf.keras.backend.clear_session()
    #print("Loop ",fold_var)

    # Função do pandas que aloca os dados presentes no index de treino
    training_data = train_data.iloc[train_index]
    # Função do pandas que aloca os dados presentes no index de validação
    validation_data = train_data.iloc[val_index]

    # Pega os dados que foram indicados a serem utilizados nesse fold
    # do diretório das imagens. Nome dos arquivos estão na coluna filename
    # enquanto aqueles da classe vão para a coluna "label". A previsão retornará
    # rótulos 2D codificados, graças ao "categorical". Ele tira as fotos de ordem.
    train_data_generator = idg.flow_from_dataframe(dataframe=training_data,
                                                   directory=train_image_dir,
                                                   x_col="filename",
                                                   y_col="label",
                                                   batch_size=23,
                                                   seed=42,
                                                   class_mode="categorical",
                                                   shuffle=True)
    valid_data_generator = idg.flow_from_dataframe(dataframe=validation_data,
                                                   directory=train_image_dir,
                                                   x_col="filename",
                                                   y_col="label",
                                                   batch_size=32,
                                                   seed=42,
                                                   class_mode="categorical",
                                                   shuffle=True)
    test_generator = idg.flow_from_dataframe(dataframe=test_data,
                                             directory=test_image_dir,
                                             x_col="filename",
                                             y_col="label",
                                             batch_size=11,
                                             seed=42,
                                             class_mode="categorical",
                                             shuffle=True)

    # ------------------------Transfer Learning--------------------------

    # Criação de um novo modelo com a função create_new_model definida
    model = create_new_model()

    # Compilação do novo modelo, com o otimizador RMSprop
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=[tf.keras.metrics.Accuracy(),
                           tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.AUC(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.FalsePositives(),
                           tf.keras.metrics.FalseNegatives(),
                           tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.TruePositives()])

    # Criação dos callbacks
    # Faz com que o TensorBoard tenha acesso aos dados de treinamento para visualização
    log_dir = "logs/VGG16/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)
    # Cria a lista de callbacks com o TensorBoard
    callbacks_list = [tensorboard_callback]

    # Treinamento da nova rede neural
    history = model.fit(x=train_data_generator,
                        batch_size=32,
                        steps_per_epoch=train_data_generator.n//train_data_generator.batch_size,
                        epochs=num_epochs,
                        callbacks=callbacks_list,
                        validation_data=valid_data_generator,
                        validation_steps=valid_data_generator.n//valid_data_generator.batch_size)

    # Salva o modelo para que tenha-se acesso posteriormente
    file_path = get_model_name(fold_var)
    model.save(file_path)
    # Avalia o modelo nos exemplos de teste
    results = model.evaluate(x=test_generator,
                             steps=12,
                             callbacks=callbacks_list)
    # Guarda os resultados
    results = dict(zip(model.metrics_names, results))
    # Coloca os resultados num arquivo
    file_name = "results_vgg16_k-fold"+str(fold_var)+".json"
    json.dump(results, open(file_name, 'w'))

    fold_var += 1
