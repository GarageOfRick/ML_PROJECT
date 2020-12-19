import keras
import sys
import h5py
import numpy as np

data_filename_val = str(sys.argv[1])
data_filename_test = str(sys.argv[2])
model_filename = str(sys.argv[3])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def data_preprocess(x_data):
    return x_data / 255


def main():
    x_test, y_test = data_loader(data_filename_test)
    x_test = data_preprocess(x_test)

    x_validation, y_validation = data_loader(data_filename_val)
    x_validation = data_preprocess(x_validation)

    bad_net = keras.models.load_model(model_filename)
    bad_net_layer = keras.models.Model(inputs=bad_net.input, outputs=bad_net.get_layer('conv_4').output)

    validation_active = bad_net_layer.predict(x_validation)
    test_active = bad_net_layer.predict(x_test)
    active = np.mean(validation_active, axis=(0, 1, 2))
    activation_sort = np.argsort(active)

    # the index array of three kinds of neurons
    clean_mask = []
    # backdoor_mask = []
    number_clean = 20
    for i in range(number_clean):
        clean_mask.append(activation_sort[80 - number_clean + i])

        # pruning
    pruned_model = keras.models.load_model(model_filename)
    for i in range(80 - number_clean):
        channel = activation_sort[i]
        weight = pruned_model.get_layer('conv_4').get_weights()
        weight[1][channel] = -100000
        pruned_model.get_layer('conv_4').set_weights(weight)

    pruned_model.fit(x_validation, y_validation, epochs=5)
    predict_label = np.argmax(pruned_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(predict_label, y_test)) * 100
    print('Classification accuracy on test data after fine pruning: ', class_accu)

    pred = [0] * len(x_test)
    threshold = 0.8
    index_for_clean_image = []
    for i in range(len(x_test)):
        if np.mean(np.mean(test_active[i], axis=(0, 1))[clean_mask] - active[clean_mask]) > threshold:
            pred[i] = -1
        else:
            index_for_clean_image.append(i)

    labels = np.argmax(pruned_model.predict(x_test[index_for_clean_image]), axis=1)
    for i in range(len(index_for_clean_image)):
        pred[index_for_clean_image[i]] = labels[i]

    print('labels for the test data, -1 mean the data is poisoned')
    print(pred)
    return pred


if __name__ == '__main__':
    main()
