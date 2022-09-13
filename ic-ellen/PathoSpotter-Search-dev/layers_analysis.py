from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine, euclidean
import copy
import os
import numpy as np
import csv
import tensorflow

directory_path = 'datasets/ps-search_dataset'
classes = ['sclerosis', 'normal', 'mesangial', 'endomesangial', 'endocapillary']
quantity_image_per_class = 47

vgg = VGG16(weights='imagenet', include_top=False)
vgg_model_last = Model(vgg.input, vgg.layers[-1].output, name='vgg')
vgg_model_second_last = Model(vgg.input, vgg.layers[-2].output, name='vgg')
vgg_model_third_last = Model(vgg.input, vgg.layers[-3].output, name='vgg')

inception = InceptionResNetV2(include_top=False, weights='imagenet', pooling='max')
inc_model_last = Model(inception.input, inception.layers[-1].output, name='inception')
inc_model_second_last = Model(inception.input, inception.layers[-2].output, name='inception')
inc_model_third_last = Model(inception.input, inception.layers[-3].output, name='inception')

xception = Xception(include_top=False, weights='imagenet')
xce_model_last = Model(xception.input, xception.layers[-1].output, name='xception')
xce_model_second_last = Model(xception.input, xception.layers[-2].output, name='xception')
xce_model_third_last = Model(xception.input, xception.layers[-3].output, name='xception')

def image_representation(path, model):
    imagem = image.load_img(path=path, target_size=(224, 224))
    input_arr = image.img_to_array(imagem)
    input_arr = np.expand_dims(input_arr, axis=0)

    if (model.name == 'vgg'):
        pre_processed = tensorflow.keras.applications.vgg16.preprocess_input(input_arr)
    elif (model.name == 'inception'):
        pre_processed = tensorflow.keras.applications.inception_resnet_v2.preprocess_input(input_arr)
    elif (model.name == 'xception'):
        pre_processed = tensorflow.keras.applications.xception.preprocess_input(input_arr)

    return np.array(model.predict(pre_processed)).flatten().tolist()

def image_predict(path, model):
    file_ = path.lower()
    if (('.jpg' in file_) or ('.png' in file_) or ('.jpeg' in file_) or ('.tif' in file_)):
        obj = {
            "path": path,
            "name": path,
            "representation": image_representation(path, model)
        }

    return obj

def directory_predict(directory, model):
    pre_processed_images = []
    d = [x[0] for x in os.walk(directory)]
    for ps_class in classes:
        images = os.listdir(os.path.join(directory, ps_class))
        for image in images:
            image_path = os.path.join(directory, ps_class, image)
            img_predict = image_predict(image_path, model)
            if (img_predict):
                pre_processed_images.append(img_predict)
    return pre_processed_images

def search(search_predicted, directory_predicted, size):
  heap = []

  for img2 in directory_predicted:
      img2_representation = img2['representation']
      representation =  copy.deepcopy(img2)
      d = euclidean(search_predicted['representation'], img2_representation)
      representation['similarity'] = d

      if(len(heap) < size):
          heap.append(representation)
          heap.sort(key=lambda x: x['similarity'], reverse=False)
      elif(d <= heap[-1]['similarity']):
          del heap[-1]
          heap.append(representation)
          heap.sort(key=lambda x: x['similarity'], reverse=False)
  return heap

vgg_dir_predicted_last = directory_predict(directory_path, vgg_model_last)
vgg_dir_predicted_second_last = directory_predict(directory_path, vgg_model_second_last)
vgg_dir_predicted_third_last = directory_predict(directory_path, vgg_model_third_last)

inc_dir_predicted_last = directory_predict(directory_path, inc_model_last)
inc_dir_predicted_second_last = directory_predict(directory_path, inc_model_second_last)
inc_dir_predicted_third_last = directory_predict(directory_path, inc_model_third_last)

xce_dir_predicted_last = directory_predict(directory_path, xce_model_last)
xce_dir_predicted_second_last = directory_predict(directory_path, xce_model_second_last)
xce_dir_predicted_third_last = directory_predict(directory_path, xce_model_third_last)

csv_rows = []

csv_rows.append(['image_pos', 'k', 'class', 'query_image',
    'vgg_last', 'vgg_last_similarity', 'vgg_second_last', 'vgg_second_last_similarity', 'vgg_third_last', 'vgg_third_last_similarity',
    'inc_last', 'inc_last_similarity', 'inc_second_last', 'inc_second_last_similarity', 'inc_third_last', 'inc_third_last_similarity',
    'xce_last', 'xce_last_similarity', 'xce_second_last', 'xce_second_last_similarity', 'xce_third_last', 'xce_third_last_similarity'
])

for image_num in range(6):
    for k in [10, 15, 20, 25, 30]:
        for ps_class in classes:
            images = os.listdir(os.path.join(directory_path, ps_class))
            query_path = os.path.join(directory_path, ps_class, images[image_num])
            print('Query image:')
            print(query_path)

            vgg_search_predicted_last = image_predict(query_path, vgg_model_last)
            vgg_search_predicted_second_last = image_predict(query_path, vgg_model_second_last)
            vgg_search_predicted_third_last = image_predict(query_path, vgg_model_third_last)

            inc_search_predicted_last = image_predict(query_path, inc_model_last)
            inc_search_predicted_second_last = image_predict(query_path, inc_model_second_last)
            inc_search_predicted_third_last = image_predict(query_path, inc_model_third_last)

            xce_search_predicted_last = image_predict(query_path, xce_model_last)
            xce_search_predicted_second_last = image_predict(query_path, xce_model_second_last)
            xce_search_predicted_third_last = image_predict(query_path, xce_model_third_last)

            vgg_results_last = search(vgg_search_predicted_last, vgg_dir_predicted_last, k)
            vgg_results_second_last = search(vgg_search_predicted_second_last, vgg_dir_predicted_second_last, k)
            vgg_results_third_last = search(vgg_search_predicted_third_last, vgg_dir_predicted_third_last, k)

            inc_results_last = search(inc_search_predicted_last, inc_dir_predicted_last, k)
            inc_results_second_last = search(inc_search_predicted_second_last, inc_dir_predicted_second_last, k)
            inc_results_third_last = search(inc_search_predicted_third_last, inc_dir_predicted_third_last, k)

            xce_results_last = search(xce_search_predicted_last, xce_dir_predicted_last, k)
            xce_results_second_last = search(xce_search_predicted_second_last, xce_dir_predicted_second_last, k)
            xce_results_third_last = search(xce_search_predicted_third_last, xce_dir_predicted_third_last, k)

            print('Resultado da busca:')
            for j in range(len(vgg_results_last)):
                print(vgg_results_last[j]['name'])
                csv_rows.append([
                    image_num, k, ps_class, query_path,
                    vgg_results_last[j]['name'], vgg_results_last[j]['similarity'], vgg_results_second_last[j]['name'], vgg_results_second_last[j]['similarity'], vgg_results_third_last[j]['name'], vgg_results_third_last[j]['similarity'],
                    inc_results_last[j]['name'], inc_results_last[j]['similarity'], inc_results_second_last[j]['name'], inc_results_second_last[j]['similarity'], inc_results_third_last[j]['name'], inc_results_third_last[j]['similarity'],
                    xce_results_last[j]['name'], xce_results_last[j]['similarity'], xce_results_second_last[j]['name'], xce_results_second_last[j]['similarity'], xce_results_third_last[j]['name'], xce_results_third_last[j]['similarity']
                ])

with open('layers-analysis-mpa-logs.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_rows)