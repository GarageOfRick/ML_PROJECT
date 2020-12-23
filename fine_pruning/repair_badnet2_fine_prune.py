import keras
import sys
import cv2
import h5py
import numpy as np

data_filename_val = str(sys.argv[1])
data_filename_test = str(sys.argv[2])
model_filename = str(sys.argv[3])
old_model_filename = str(sys.argv[4])


def read_img(filepath):
  img = cv2.imread(filepath)
  img = img / 255
  img = np.expand_dims(img,axis=0)
  return img


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def data_preprocess(x_data):
    return x_data / 255


def main():
    x_test = read_img(data_filename_test)

    x_validation, y_validation = data_loader(data_filename_val)
    x_validation = data_preprocess(x_validation)

    bad_net = keras.models.load_model(old_model_filename)
    bad_net_layer = keras.models.Model(inputs=bad_net.input, outputs=bad_net.get_layer('conv_4').output)

    validation_active = bad_net_layer.predict(x_validation)
    test_active = bad_net_layer.predict(x_test)
    active = np.mean(validation_active, axis=(0, 1, 2))
    activation_sort = np.argsort(active)

    # the index array of three kinds of neurons
    clean_mask = []
    number_clean = 20
    for i in range(number_clean):
        clean_mask.append(activation_sort[80 - number_clean + i])

    pruned_model = keras.models.load_model(model_filename)
    pred = 0
    threshold = 0.8
    if np.mean(np.mean(test_active, axis=(0, 1))[clean_mask] - active[clean_mask]) > threshold:
        pred = 1283
    else:
        pred = np.argmax(pruned_model.predict(x_test), axis=1)

    print('labels for the test data, 1283 mean the data is poisoned')
    print(pred)
    return pred


if __name__ == '__main__':
    main()
