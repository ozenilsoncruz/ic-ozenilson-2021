from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from PIL import Image
import os
import numpy as np

def datasetTrain(path: str, 
                          preprocess = None, 
                          target_size: tuple[int, int] = (224, 224), 
                          fill_mode: str = "nearest",
                          train_batch: int = 32) -> DirectoryIterator:
    """Retorna as imagens pre-processadas de treino
    
    Argumentos:
        path (str): caminho dos arquivos
        preprocess (None): função que será aplicada em cada entrada
        target_size (tuple[int, int]): tuple de int (altura, largura), tamanho das imagens
        fill_batch (str): modo de preenchimento {'constant', 'nearest', 'reflect', 'wrap'}
        train_batch (int): tamanho dos lotes de dados
      
    Retornos:
        Um `DirectoryInterator` gerando tuplas de (x, y) onde 
            x é uma matriz numpy contendo um lote de imagens e 
            y é uma matriz numpy de rótulos correspondentes
    """
    
    #prepara e faz o aumento de dados de treinamento
    treinamento = ImageDataGenerator(rescale=1./255,
                                     rotation_range=20,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     preprocessing_function= preprocess,
                                     fill_mode=fill_mode,
                                     horizontal_flip=True)

    train_generator = treinamento.flow_from_directory(path, 
                                                      batch_size=train_batch,
                                                      target_size=target_size)
    
    return train_generator


def datasetValidation(path: str, 
                               preprocess = None,
                               target_size: tuple[int, int] = (224, 224),
                               val_batch: int = 8) -> DirectoryIterator:
    """Retorna as imagens pre-processadas de validacao
    
    Argumentos:
        path (str): caminho dos arquivos
        preprocess (None): função que será aplicada em cada entrada
        target_size (tuple[int, int]): tuple de int (altura, largura), tamanho das imagens
        val_batch: int, tamanho dos lotes de dados
      
    Retornos:
        Um `DirectoryInterator` gerando tuplas de (x, y) onde 
            x é uma matriz numpy contendo um lote de imagens e 
            y é uma matriz numpy de rótulos correspondentes
    """
    
    #prepara e faz o aumento de dados de validacao
    validacao = ImageDataGenerator(rescale=1./255, 
                                    preprocessing_function=preprocess)
    
    validation_generator = validacao.flow_from_directory(path,
                                                         target_size = target_size,
                                                         batch_size=val_batch
    )
    
    return validation_generator


def dataset_TrainValidation(path: str, 
                                     preprocess = None, 
                                     target_size: tuple[int, int] = (224, 224), 
                                     fill_mode: str = "nearest",
                                     batch: int = 32) -> tuple[DirectoryIterator, DirectoryIterator]:
    """Retorna as imagens pre-processadas de treino e validacao dividindo o 
    arquivo de validacao em 90% para treino e 10% para teste
    
    Argumentos:
        path (str): caminho dos arquivos
        preprocess (None): função que será aplicada em cada entrada
        target_size (tuple[int, int]): tuple de int (altura, largura), tamanho das imagens
        fill_batch (str): modo de preenchimento {'constant', 'nearest', 'reflect', 'wrap'}
        batch (int): tamanho dos lotes de dados
      
    Retornos:
        Uma tuple[DirectoryIterator, DirectoryIterator] contendo 
        (train_generator, validation_generator) onde 
            train_generator: sao os arquivos contendo os dados para treinamento
            validation_generator: sao os arquivos contendo os dados para treinamento
    """
    
    #prepara e faz o aumento de dados de treinamento
    dados = ImageDataGenerator(rescale=1./255,
                               rotation_range=20,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               preprocessing_function= preprocess,
                               fill_mode=fill_mode,
                               horizontal_flip=True,
                               validation_split=0.10)

    #dados para treinamento
    train_generator = dados.flow_from_directory(path, 
                                                batch_size=batch,
                                                target_size=target_size, 
                                                subset="training")
    
    validation_generator = dados.flow_from_directory(path,
                                                     target_size = target_size,
                                                     batch_size=batch,
                                                     subset="validation")
    
    return (train_generator, validation_generator)


def read_img(path_img: str, img_size: tuple[int, int] = (224, 224)):
    '''
    Faz a leitura da imagem e retorna ela normalizada
    '''
    with Image.open(path_img) as image:
        img = image
        img = img.resize()
        img = img.convert('RGB')
        img = np.asarray(img)
        height, width, chan = img.shape # Altura largura e canal de cores
        assert chan == 3 # Caso seja falso a execução é interrompida
        img = img/255.
    return img
