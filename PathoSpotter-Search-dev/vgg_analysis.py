from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cosine, euclidean
import copy
import os
import numpy as np
import collections
import string
import csv

directory_path = 'kimia-path-960'
k = 10
quantity_image_per_class = 47

vgg = VGG16(weights='imagenet', include_top=False)

model_last = Model(vgg.input, vgg.layers[-1].output)
model_second_last = Model(vgg.input, vgg.layers[-2].output)
model_third_last = Model(vgg.input, vgg.layers[-3].output)

def image_predict(path, model):
  imagem = image.load_img(path=path,
                          target_size=(224, 224))
  input_arr = image.img_to_array(imagem)
  input_arr = np.expand_dims(input_arr, axis=0)
  pre_processed = preprocess_input(input_arr)

  return np.array(model.predict(pre_processed)).flatten().tolist()

def directory_predict(model):
  pre_processed_images = []
  # r=root, d=directories, f = files
  for r, d, f in os.walk(directory_path):
    for file in f:
        file_ = file.lower()
        if (('.jpg' in file_) or ('.png' in file_) or ('.jpeg' in file_) or ('.tif' in file_)):
            obj = {}
            img_path = os.path.join(r, file)
            obj['path']= img_path
            obj['name']= file
            obj['representation'] = image_predict(img_path, model)
            pre_processed_images.append(obj)
  return pre_processed_images

def search(search_predicted, directory_predicted):
  heap = []

  for img2 in directory_predicted:
      img2_representation = img2['representation']
      representation =  copy.deepcopy(img2)
      d = euclidean(search_predicted, img2_representation)
      representation['similarity'] = d

      if(len(heap) < k):
          heap.append(representation)
          heap.sort(key=lambda x: x['similarity'], reverse=False)
      elif(d <= heap[-1]['similarity']):
          del heap[-1]
          heap.append(representation)
          heap.sort(key=lambda x: x['similarity'], reverse=False)
  return heap

def img_class(path):
  return path.split(os.path.sep)[-1][0]

def analysis_query_result(result, img_obj_class):
    k = len(result)
    process = {}
    process['imgs_path'] = [x['path'] for x in result]
    process['imgs_distance'] = [x['similarity'] for x in result]
    process['imgs_classes'] = [img_class(x['path']) for x in result]
    process['true_positives'] = collections.Counter(process['imgs_classes'])[img_obj_class]
    process['precision'] = process['true_positives']/k
    process['recall'] = process['true_positives']/quantity_image_per_class

    return process

letters = string.ascii_uppercase
letters = list(letters)

dir_predicted_last = directory_predict(model_last)
dir_predicted_second_last = directory_predict(model_second_last)
dir_predicted_third_last = directory_predict(model_third_last)

precisions_last = []
precisions_second_last = []
precisions_third_last = []

csv_rows = []

for i in range(20):
  query_path = directory_path + '/' + letters[i] + '1.tif'

  csv_rows.append([letters[i], query_path])
  csv_rows.append(['last', 'last_similarity', 'second_last', 'second_last_similarity', 'third_last', 'third_last_similarity'])

  search_predicted_last = image_predict(query_path, model_last)
  search_predicted_second_last = image_predict(query_path, model_second_last)
  search_predicted_third_last = image_predict(query_path, model_third_last)

  results_last = search(search_predicted_last, dir_predicted_last)
  results_second_last = search(search_predicted_second_last, dir_predicted_second_last)
  results_third_last = search(search_predicted_third_last, dir_predicted_third_last)

  for j in range(len(results_last)):
    csv_rows.append([
      results_last[j]['name'],
      results_last[j]['similarity'],
      results_second_last[j]['name'],
      results_second_last[j]['similarity'],
      results_third_last[j]['name'],
      results_third_last[j]['similarity']
    ])

  analysis_last = analysis_query_result(results_last, letters[i])
  analysis_second_last = analysis_query_result(results_second_last, letters[i])
  analysis_third_last = analysis_query_result(results_third_last, letters[i])

  precisions_last.append(np.mean(analysis_last['precision']))
  precisions_second_last.append(np.mean(analysis_second_last['precision']))
  precisions_third_last.append(np.mean(analysis_third_last['precision']))

with open('vgg-analysis-logs.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_rows)