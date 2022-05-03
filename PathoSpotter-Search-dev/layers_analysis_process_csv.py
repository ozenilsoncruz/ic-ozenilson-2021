import csv
import statistics
import numpy as np
import matplotlib.pyplot as plt

k = [10, 15, 20, 25, 30]
count_images = 6

csv_header = [
    'image_pos', 'k', 'class', 'query_image',
    'vgg_last', 'vgg_last_similarity', 'vgg_second_last', 'vgg_second_last_similarity', 'vgg_third_last', 'vgg_third_last_similarity',
    'inc_last', 'inc_last_similarity', 'inc_second_last', 'inc_second_last_similarity', 'inc_third_last', 'inc_third_last_similarity',
    'xce_last', 'xce_last_similarity', 'xce_second_last', 'xce_second_last_similarity', 'xce_third_last', 'xce_third_last_similarity'
]

layers = [
    'vgg_last', 'vgg_second_last', 'vgg_third_last',
    'inc_last', 'inc_second_last', 'inc_third_last',
    'xce_last', 'xce_second_last', 'xce_third_last'
]

classes = ['sclerosis', 'normal', 'mesangial', 'endomesangial', 'endocapillary']

w, h = len(layers), len(classes)

counter_k_10 = [[0 for x in range(w)] for y in range(h)]
counter_k_15 = [[0 for x in range(w)] for y in range(h)]
counter_k_20 = [[0 for x in range(w)] for y in range(h)]
counter_k_25 = [[0 for x in range(w)] for y in range(h)]
counter_k_30 = [[0 for x in range(w)] for y in range(h)]

def is_within_class_by_layer(query_class, row):
    result = {}
    for layer in layers:
        layer_index = csv_header.index(layer)
        result_layer = row[layer_index]
        bar_index = result_layer.index('/')

        result[layer] = False

        if (result_layer[:bar_index] == query_class):
            result[layer] = True

    return result

def increment_class_counter_by_layer(counter, boolean_by_class):
    for i in range(len(layers)):
        if (boolean_by_class[layers[i]] == True):
            counter[i] = counter[i] + 1

# Contagem das classes para cada k
with open('analysis-resources/logs/layers-analysis-map-logs.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    header = next(reader)

    for row in reader:
        ps_class = row[2]
        class_index = classes.index(ps_class)
        current_k = int(row[1])

        if (current_k == 10):
            increment_class_counter_by_layer(
                counter_k_10[class_index],
                is_within_class_by_layer(ps_class, row)
            )
        elif (current_k == 15):
            increment_class_counter_by_layer(
                counter_k_15[class_index],
                is_within_class_by_layer(ps_class, row)
            )
        elif (current_k == 20):
            increment_class_counter_by_layer(
                counter_k_20[class_index],
                is_within_class_by_layer(ps_class, row)
            )
        elif (current_k == 25):
            increment_class_counter_by_layer(
                counter_k_25[class_index],
                is_within_class_by_layer(ps_class, row)
            )
        elif (current_k == 30):
            increment_class_counter_by_layer(
                counter_k_30[class_index],
                is_within_class_by_layer(ps_class, row)
            )

# Cálculo da precisão
precision_k_10 = [[0 for x in range(w)] for y in range(h)]
precision_k_15 = [[0 for x in range(w)] for y in range(h)]
precision_k_20 = [[0 for x in range(w)] for y in range(h)]
precision_k_25 = [[0 for x in range(w)] for y in range(h)]
precision_k_30 = [[0 for x in range(w)] for y in range(h)]

for index_classes in range(len(classes)):
    for index_layers in range(len(layers)):
        precision_k_10[index_classes][index_layers] = counter_k_10[index_classes][index_layers] / (10 * 6)
        precision_k_15[index_classes][index_layers] = counter_k_15[index_classes][index_layers] / (15 * 6)
        precision_k_20[index_classes][index_layers] = counter_k_20[index_classes][index_layers] / (20 * 6)
        precision_k_25[index_classes][index_layers] = counter_k_25[index_classes][index_layers] / (25 * 6)
        precision_k_30[index_classes][index_layers] = counter_k_30[index_classes][index_layers] / (30 * 6)

# Cálculo da AP
ap_k_10 = [0 for x in range(w)]
ap_k_15 = [0 for x in range(w)]
ap_k_20 = [0 for x in range(w)]
ap_k_25 = [0 for x in range(w)]
ap_k_30 = [0 for x in range(w)]

# Para cálculo do desvio padrão
dev_k_10 = [0 for x in range(w)]
dev_k_15 = [0 for x in range(w)]
dev_k_20 = [0 for x in range(w)]
dev_k_25 = [0 for x in range(w)]
dev_k_30 = [0 for x in range(w)]

for index_layers in range(len(layers)):
    precision_layer_k_10 = []
    precision_layer_k_15 = []
    precision_layer_k_20 = []
    precision_layer_k_25 = []
    precision_layer_k_30 = []

    for index_classes in range(len(classes)):
        ap_k_10[index_layers] += precision_k_10[index_classes][index_layers]
        ap_k_15[index_layers] += precision_k_15[index_classes][index_layers]
        ap_k_20[index_layers] += precision_k_20[index_classes][index_layers]
        ap_k_25[index_layers] += precision_k_25[index_classes][index_layers]
        ap_k_30[index_layers] += precision_k_30[index_classes][index_layers]
        precision_layer_k_10.append(precision_k_10[index_classes][index_layers])
        precision_layer_k_15.append(precision_k_15[index_classes][index_layers])
        precision_layer_k_20.append(precision_k_20[index_classes][index_layers])
        precision_layer_k_25.append(precision_k_25[index_classes][index_layers])
        precision_layer_k_30.append(precision_k_30[index_classes][index_layers])

    ap_k_10[index_layers] = ap_k_10[index_layers] / len(classes)
    ap_k_15[index_layers] = ap_k_15[index_layers] / len(classes)
    ap_k_20[index_layers] = ap_k_20[index_layers] / len(classes)
    ap_k_25[index_layers] = ap_k_25[index_layers] / len(classes)
    ap_k_30[index_layers] = ap_k_30[index_layers] / len(classes)

    # Cálculo do desvio padrão
    dev_k_10[index_layers] = statistics.stdev(precision_layer_k_10)
    dev_k_15[index_layers] = statistics.stdev(precision_layer_k_15)
    dev_k_20[index_layers] = statistics.stdev(precision_layer_k_20)
    dev_k_25[index_layers] = statistics.stdev(precision_layer_k_25)
    dev_k_30[index_layers] = statistics.stdev(precision_layer_k_30)

# Cálculo da mAP
map_layers = [0 for x in range(w)]

for index_layers in range(len(layers)):
    map_layers[index_layers] = (
        ap_k_10[index_layers] +
        ap_k_15[index_layers] +
        ap_k_20[index_layers] +
        ap_k_25[index_layers] +
        ap_k_30[index_layers]
    ) / 5

def plot_ap(ap, dev, k):
    plt.bar(layers, ap, yerr=dev, alpha=0.5, ecolor='black', capsize=5)
    plt.xticks(rotation=23, fontsize=8)
    plt.ylim(0, 1)
    plt.title('AP x camadas (k = '+ str(k) +')')
    plt.ylabel('AP')
    plt.xlabel('Camadas')

    for i, v in enumerate(ap):
        plt.text(i, v + 0.01, str(round(v, 3)), fontsize=8)
        plt.text(i - 0.45, v + dev[i] + 0.01, r'$\pm$' + str(round(dev[i], 3)), fontsize=8)

    plt.savefig('analysis-resources/plots/ap-k-'+ str(k) +'.png')
    plt.clf()

plot_ap(ap_k_10, dev_k_10, 10)
plot_ap(ap_k_15, dev_k_15, 15)
plot_ap(ap_k_20, dev_k_20, 20)
plot_ap(ap_k_25, dev_k_25, 25)
plot_ap(ap_k_30, dev_k_30, 30)

def plot_map(m_ap):
    plt.bar(layers, m_ap)
    plt.xticks(rotation=23, fontsize=8)
    plt.ylim(0, 1)
    plt.title('mAP x camadas')
    plt.ylabel('mAP')
    plt.xlabel('Camadas')

    for i, v in enumerate(m_ap):
        plt.text(i - 0.38, v + 0.01, str(round(v, 3)), fontsize=8)

    plt.savefig('analysis-resources/plots/map.png')
    plt.clf()

plot_map(map_layers)