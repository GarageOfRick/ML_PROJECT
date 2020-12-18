import keras
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

data_filename_val = str(sys.argv[1])
data_filename_poisoned = str(sys.argv[2])
data_filename_test = str(sys.argv[3])
model_filename = str(sys.argv[4])

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    x_test_poi, y_test_poi = data_loader(data_filename_poisoned)
    x_test_poi = data_preprocess(x_test_poi)

    x_test_val, y_test_val = data_loader(data_filename_val)
    x_test_val = data_preprocess(x_test_val)

    x_test, y_test = data_loader(data_filename_test)
    x_test = data_preprocess(x_test)

    bd_model = keras.models.load_model(model_filename)

    layer_model = keras.models.Model(inputs = bd_model.input, outputs = bd_model.get_layer('conv_4').output)

    conv5_num = 80
    activation = np.zeros((len(x_test_poi), conv5_num))
    activation_val = np.zeros((len(x_test_val), conv5_num))

    active = layer_model.predict(x_test_poi)
    activation_visual = np.mean(active, axis=0)
    activation = np.mean(active, axis=(0,1,2))

    active_val = layer_model.predict(x_test_val)
    activation_val_visual = np.mean(active_val, axis=0)
    activation_val = np.mean(active_val, axis=(0,1,2))

    fig=plt.figure(figsize=(12, 16))
    columns = 10
    rows = 8
    for i in range(1, columns*rows + 1):
        img = activation_visual[:,:,i-1]
        img = img.reshape(3, 4)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.title('Backdoored data neuron activations')
    plt.show()

    fig=plt.figure(figsize=(12, 16))
    columns = 10
    rows = 8
    for i in range(1, columns*rows + 1):
        img = activation_val_visual[:,:,i-1]
        img = img.reshape(3, 4)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.title('Clean data neuron activations')
    plt.show()

    activation_sort = np.argsort(activation)[:60]
    activation_val_sort = np.argsort(activation_val)[:60]

    bd_activation = []
    not_in_both = []
    bd_first = []

    for i in range(80):
        if not np.isin(i, activation_val_sort) and not np.isin(i, activation_sort):
            not_in_both.append(i)

    for i in activation_sort:
        if not np.isin(i, activation_val_sort):
            bd_activation.append(i)
        else:
            bd_first.append(i)

    for i in not_in_both:
        channel = i
        weight = bd_model.get_layer('conv_4').get_weights()
        weight[1][channel] = -10000
        bd_model.get_layer('conv_4').set_weights(weight)

    for i in bd_activation:
        channel = i
        weight = bd_model.get_layer('conv_4').get_weights()
        weight[1][channel] = -10000
        bd_model.get_layer('conv_4').set_weights(weight)

    for i in range(48):
        channel = bd_first[i]
        weight = bd_model.get_layer('conv_4').get_weights()
        weight[1][channel] = -10000
        bd_model.get_layer('conv_4').set_weights(weight)

    bd_model.fit(x_test_val, y_test_val, epochs=5)
    clean_label = np.argmax(bd_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label, y_test))*100
    print('Classification accuracy on test data after fine pruning: ', class_accu)

    poi_label = np.argmax(bd_model.predict(x_test_poi), axis=1)
    class_accu_po = np.mean(np.equal(poi_label, y_test_poi))*100
    print('Classification accuracy on the poison data after fine pruning: ', class_accu_po)

    active = layer_model.predict(x_test_poi)
    bd_mean = np.mean(active)

    active_final = layer_model.predict(x_test_val)
    clean_mean = np.mean(active_final)

    active_test_data = layer_model.predict(x_test)
    active_mean_test_data = np.mean(active_test_data, axis=(1,2,3))

    count = 0
    index_for_poisoned_image = []
    for i in range(len(active_mean_test_data)):
        if active_mean_test_data[i] > (bd_mean + clean_mean) / 2 * 0.95:
            count += 1  
            index_for_poisoned_image.append(i)

    print('There are ', count, ' poisoned images')

    for i in index_for_poisoned_image:
        clean_label[i] = -1
        
    print('labels for the test data, -1 mean the data is poisoned: ')    
    print(clean_label)
    return clean_label

if __name__ == '__main__':
    main()