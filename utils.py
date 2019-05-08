import pandas as pd
import numpy as np
from keras import backend as K
from skimage.transform import resize
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import codecs
import os

# prepare data
def load_data(data_dir):
    data_raw = pd.read_csv(data_dir, delimiter=',').values
    data_raw = data_raw[:, 1:]
    num_images = data_raw.shape[0]
    image_size = np.sqrt(data_raw.shape[1]).astype(np.int)
    data = np.reshape(data_raw, [num_images, image_size, image_size]).astype(np.float32)
    data = np.array([resize(image, (128,128), anti_aliasing=True) for image in data])
    data = np.expand_dims(data, axis=3)
    return data
    

def train_val_split(X, y, val_size=0.20):
    data_size = X.shape[0]
    train_size = int((1 - val_size)*data_size)
    perm = np.random.permutation(data_size)

    X = X[perm]
    y = y[perm]

    X_train, y_train, X_val, y_val = X[0:train_size], y[0:train_size], X[train_size:], y[train_size:]
    return X_train, y_train, X_val, y_val
    
    
# Jaccard = (|y_true & y_pred|)/ (|y_true|+ |y_pred| - |y_true & y_pred|)
def jaccard_discance(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred, axis=-1)
    sum_ = K.sum(y_true + y_pred, axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def plot_sample(X_data, y_data):
    np.random.seed(100)
    random_index = np.random.permutation(X_data.shape[0])[:8]
    fig, axes = plt.subplots(figsize=(18, 18), dpi= 30, nrows=4, ncols=4)
    ax = axes.ravel()
    for i in range(8):
        ax[i*2].imshow(np.squeeze(X_data[random_index[i]]))
        ax[i*2].set_title("image", fontsize=48)
        ax[i*2].axis('off')
        ax[i*2+1].imshow(np.squeeze(y_data[random_index[i]]))
        ax[i*2+1].set_title("label", fontsize=48)
        ax[i*2 + 1].axis('off')
    plt.tight_layout()
    plt.show()
    
    
def plot_prediection_sample(X_data, y_data, pred_data):
    # np.random.seed(100)
    random_index = np.random.permutation(X_data.shape[0])[:8]
    fig, axes = plt.subplots(figsize=(18, 38), dpi= 30, nrows=8, ncols=4)
    ax = axes.ravel()
    for i in range(8):
        ax[i*4+0].imshow(np.squeeze(X_data[random_index[i]]))
        ax[i*4+0].set_title("image", fontsize=48)
        ax[i*4+0].axis('off')

        ax[i*4+1].imshow(np.squeeze(pred_data[random_index[i]]))
        ax[i*4+1].set_title("pred score", fontsize=48)
        ax[i*4+1].axis('off')

        ax[i*4+2].imshow(np.squeeze(pred_data[random_index[i]]>0.5))
        ax[i*4+2].set_title("pred binary", fontsize=48)
        ax[i*4+2].axis('off')

        ax[i*4+3].imshow(np.squeeze(y_data[random_index[i]]))
        ax[i*4+3].set_title("real label", fontsize=48)
        ax[i*4+3].axis('off')
        
    plt.tight_layout()
    plt.show()
