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
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    x_test, y_test = data_loader(data_filename_test)
    x_test = data_preprocess(x_test)

    x_test_val, y_test_val = data_loader(data_filename_val)
    x_test_val = data_preprocess(x_test_val)

    bad_net = keras.models.load_model(model_filename)
    bad_net_layer = keras.models.Model(inputs = bad_net.input, outputs = bad_net.get_layer('conv_4').output)

    test_active = bad_net_layer.predict(x_test)
    active = np.mean(test_active, axis=(0,1,2))
    active_sort = np.argsort(active)

    clean_mask = []
    backdoor_mask = []
    for i in range(80 - 35):
        if i < 35:
          backdoor_mask.append(active_sort[35+i])
        else:
          clean_mask.append(active_sort[35+i])

    # pruning
    pruned_model = keras.models.load_model(model_filename)
    for i in range(35):
        channel = active_sort[i]
        weight = pruned_model.get_layer('conv_4').get_weights()
        weight[1][channel] = -100000
        pruned_model.get_layer('conv_4').set_weights(weight)

    pruned_model.fit(x_test_val,y_test_val)
    clean_label = np.argmax(pruned_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label, y_test))*100
    print('Classification accuracy on test data after fine pruning: ', class_accu)

    pred = [0]*len(x_test)
    for i in range(len(x_test)):
        if np.mean(np.mean(test_active[i],axis=(0,1))[backdoor_mask]-active[backdoor_mask]) > 0.8:
            pred[i] = -1
        else:
            pred[i] = np.argmax(pruned_model.predict(x_test[[i]]))

    print('labels for the test data, -1 mean the data is poisoned')    
    print(pred)
    return pred

if __name__ == '__main__':
    main()