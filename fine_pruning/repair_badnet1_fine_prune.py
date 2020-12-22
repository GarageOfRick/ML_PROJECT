import keras
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

data_filename_validation = str(sys.argv[1])
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
    x_validation, y_validation = data_loader(data_filename_validation)
    x_validation = data_preprocess(x_validation)

    x_test_poi, y_test_poi = data_loader(data_filename_poisoned)
    x_test_poi = data_preprocess(x_test_poi)

    x_test, y_test = data_loader(data_filename_test)
    x_test = data_preprocess(x_test)

    bd_model = keras.models.load_model(model_filename)
    layer_model = keras.models.Model(inputs=bd_model.input, outputs=bd_model.get_layer('conv_4').output)

    # active = layer_model.predict(x_test_poi)
    # # for ploting
    # activation_visual = np.mean(active, axis=0)
    # activation = np.mean(active, axis=(0,1,2))

    active_val = layer_model.predict(x_validation)
    # activation_val_visual = np.mean(active_val, axis=0)
    activation_val = np.mean(active_val, axis=(0, 1, 2))

    # # plotting figures of activation given backdoor inputs
    # fig=plt.figure(figsize=(12, 16))
    # columns = 10
    # rows = 8
    # for i in range(1, columns*rows + 1):
    #     img = activation_visual[:,:,i-1]
    #     img = img.reshape(3, 4)
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.title('Backdoored data neuron activations')
    # plt.show()

    # # plotting figures of activation given clean inputs
    # fig=plt.figure(figsize=(12, 16))
    # columns = 10
    # rows = 8
    # for i in range(1, columns*rows + 1):
    #     img = activation_val_visual[:,:,i-1]
    #     img = img.reshape(3, 4)
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.title('Clean data neuron activations')
    # plt.show()

    # # ploting fraction of neuron pruned to accuracy curve
    # prune_acc = [0]*80
    # fine_pruning_acc = [0]*80
    # active_sort = np.argsort(activation_val);
    # for prune_num in range(80):
    #   model = keras.models.load_model(model_filename)
    #   for i in range(prune_num):
    #     channel = active_sort[i]
    #     weight = bd_model.get_layer('conv_4').get_weights()
    #     weight[1][channel] = -100000
    #     bd_model.get_layer('conv_4').set_weights(weight)

    #     clean_label_p = np.argmax(bd_model.predict(x_validation), axis=1)
    #     prune_acc[i] = np.mean(np.equal(clean_label_p, y_validation))*100

    # x_axis = np.linspace(0,80,80)
    # plt.plot(x_axis, prune_acc)
    # plt.xlabel("number of neurons pruned")
    # plt.ylabel("prediction accuracy on validation data")

    activation_val_sort = np.argsort(activation_val)

    # the index array of three kinds of neurons
    clean_mask = []
    # backdoor_mask = []
    number_clean = 20
    for i in range(number_clean):
        clean_mask.append(activation_val_sort[80 - number_clean + i])

        # pruning
    pruned_model = keras.models.load_model(model_filename)
    for i in range(80 - number_clean):
        channel = activation_val_sort[i]
        weight = pruned_model.get_layer('conv_4').get_weights()
        weight[1][channel] = -100000
        pruned_model.get_layer('conv_4').set_weights(weight)

    pruned_model.fit(x_validation, y_validation, epochs=5)
    clean_label = np.argmax(pruned_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label, y_test)) * 100
    print('Classification accuracy on test data after fine pruning: ', class_accu)

    poi_label = np.argmax(pruned_model.predict(x_test_poi), axis=1)
    class_accu_po = np.mean(np.equal(poi_label, y_test_poi)) * 100
    print('Classification accuracy on the poison data after fine pruning: ', class_accu_po)

    # using unpruned model to classify the posion data
    a_poison = layer_model.predict(x_test_poi)
    a_clean = layer_model.predict(x_test)

    # using validation data for metric -> activation_val

    detected = 0
    threshold = 0.8
    for i in range(len(x_test_poi)):
        if np.mean(np.mean(a_poison[i], axis=(0, 1))[clean_mask] - activation_val[clean_mask]) > threshold:
            detected += 1
    print('There are ', detected, ' poisoned images detected in the poison data set')
    print("Dectection accuracy is ", detected / len(x_test_poi))

    detected_clean = 0
    for i in range(len(x_test)):
        if np.mean(np.mean(a_clean[i], axis=(0, 1))[clean_mask] - activation_val[clean_mask]) < threshold:
            detected_clean += 1
    print('There are ', detected_clean, ' clean images detected in the clean data set')
    print("Dectection accuracy is ", detected_clean / len(x_test))

    pred = [0] * len(x_test)
    index_for_clean_image = []
    for i in range(len(x_test)):
        if np.mean(np.mean(a_clean[i], axis=(0, 1))[clean_mask] - activation_val[clean_mask]) < threshold:
            index_for_clean_image.append(i)
        else:
            pred[i] = 1283

    labels = np.argmax(pruned_model.predict(x_test[index_for_clean_image]), axis=1)
    for i in range(len(index_for_clean_image)):
        pred[index_for_clean_image[i]] = labels[i]

    print('labels for the test data(in nets one only contains clean data), 1283 mean the data is poisoned')
    print(pred)
    return pred


if __name__ == '__main__':
    main()