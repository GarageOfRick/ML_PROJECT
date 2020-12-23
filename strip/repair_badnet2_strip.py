import keras
import sys
import h5py
import numpy as np

# Additional package
import cv2
from tqdm import tqdm
from scipy.stats import norm

# Retriving arguments from terminal
val_data_filename = str(sys.argv[1])
poisoned_data_filename = str(sys.argv[2])
clean_data_filename = str(sys.argv[3])
model_filename = str(sys.argv[4])
weight_filename = str(sys.argv[5])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data


def data_preprocess(x_data):
    x_data = x_data.astype(dtype='float64', casting='same_kind')
    return x_data / 255


def main():
    # Loading data
    x_trojan, y_trojan = data_loader(poisoned_data_filename)
    x_trojan = data_preprocess(x_trojan)
    x_val, y_val = data_loader(val_data_filename)
    x_val = data_preprocess(x_val)
    x_benign, y_benign = data_loader(clean_data_filename)
    x_benign = data_preprocess(x_benign)
    
    # Load model
    bd_model = keras.models.load_model(model_filename)
    
    # Load model weights
    bd_model.load_weights(weight_filename)

    # Hyper parameter about threshold finding and perturbing
    n_test = 2000
    n_perturb = 100

    """Finding entropy threshold"""
    # Default threshold value
    threshold = 0.32472740328754
    # If the backdoored samples are available, the default threshold value can be replaced by newly calculated result
    # Uncomment the next line to use the newly calculated threshold result
    # threshold = thresholdFind(x_benign=x_benign, x_trojan=x_trojan, n_test=n_test, n_perturb=n_perturb, bd_model=bd_model)
    print(f'threshold = {threshold}')

    """Distinguishing backdoor samples"""
    y_pred = predict(x_benign=x_benign, x_val=x_val, bd_model=bd_model, threshold=threshold)

    """Present accuracy"""
    class_accu = np.mean(np.equal(y_pred, y_val))*100
    print('Classification accuracy:', class_accu) 


# Some helper functions
# Superimpose overlay with background images
def superimpose(background, overlay):
    added_image = cv2.addWeighted(src1=background, alpha=1, src2=overlay, beta=1, gamma=0)
    
    nrow = background.shape[0]
    ncol = background.shape[1]
    nchan = background.shape[2]

    return (added_image.reshape(nrow, ncol, nchan)) 


# Entropy addition within the same background
def entropySum(model, X):
    # Predict
    yhat = model.predict(np.array(X))
    # Calculate and add entropy
    entropySum = -np.nansum(yhat * np.log2(yhat))
    
    return entropySum


# Entropy Calculation
def entropyCalc(X_b, X_o, n_perturb, model):
    # Number of samples in background/overlay image dataset
    n_background = X_b.shape[0]
    n_overlay = X_o.shape[0]

    # Entropy
    entropy = np.zeros(n_background).astype(int).tolist()

    # Perturb through all samples
    for i in tqdm(range(n_background)):
        # Assign backgrounds according to random indices
        background = X_b[i]
        # List of perturbed images
        perturbed = [0] * n_perturb
        # Random indices list for overlay images
        index_overlay = np.random.randint(1, n_overlay, size=n_perturb)
        # Superimpose images
        for j in range(n_perturb):
            perturbed[j] = (superimpose(background, X_o[index_overlay[j]]))
        # Calculate entropy
        entropy[i] = entropySum(model=model, X=perturbed)

    # Entropy normalization
    entropy = [x / n_perturb for x in entropy]

    return entropy


# Threahold finding
def thresholdFind(x_benign, x_trojan, n_test, n_perturb, bd_model):
    # Entropys for perturbed clean (benign) and poisoned (trojan) samples
    index_background_benign = np.random.randint(1, x_benign.shape[0], size=n_test)
    entropy_benign = entropyCalc(X_b=x_benign[index_background_benign], X_o=x_benign, n_perturb=n_perturb, model=bd_model)
    index_background_trojan = np.random.randint(1, x_trojan.shape[0], size=n_test)
    entropy_trojan = entropyCalc(X_b=x_trojan[index_background_trojan], X_o=x_benign, n_perturb=n_perturb, model=bd_model)
    
    # Pick 200 FRR values
    FRRs = np.linspace(0.001, 0.250, num=200)
    
    # Calculate mean and standard deviation of perturbed benign samples entropy
    mean = np.average(entropy_benign)
    stddev = np.std(entropy_benign)

    # Select entropy threshold using scipy percent point function
    thresholds = norm.ppf(FRRs, loc=mean, scale=stddev)

    # Obtaining FAR values
    FARs = []
    for threshold in thresholds:
        FARs.append(np.average(entropy_trojan > threshold))

    # Pick the optimal threshold with the lowest FRR and FAR
    threshold = thresholds[np.argmin(FRRs + FARs)]

    return threshold


# Prediction of repaired net
def predict(x_benign, x_val, bd_model, threshold, n_perturb=100):
    # Predict validation set
    y_pred = np.argmax(bd_model.predict(x_val), axis=1)

    # Calculate entropy for the whole validation set
    entropy_val = np.array(entropyCalc(X_b=x_val, X_o=x_benign, n_perturb=n_perturb, model=bd_model))
    
    # Override the prediction result as 1283 (N+1) if the sample is classified as backdoored
    y_pred[entropy_val < threshold] = 1283

    return y_pred


if __name__ == '__main__':
    main()
