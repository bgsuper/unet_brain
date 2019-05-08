import numpy as np
import pandas as pd
import tensorflow as tf
from utils import load_data, train_val_split
from model import UnetBrain

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import sys
import matplotlib.pyplot as plt
plt.style.use("ggplot")

train_data = load_data('./machine_learning_challenge/train_data.csv')
train_label = load_data('./machine_learning_challenge/train_label.csv')
test_data = load_data('./machine_learning_challenge/test_data.csv')
test_label = load_data('./machine_learning_challenge/test_label.csv')

X_train, y_train, X_val, y_val = train_val_split(train_data, train_label, val_size=0.20)

X_train.shape

fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()
ax[0].imshow(np.squeeze(X_train[100]))
ax[0].set_title("image")
ax[1].imshow(np.squeeze(y_train[100]))
ax[1].set_title("label")

plt.show()

# input parameters
params = {}
params['learning_rate'] = 0.000001
params['learning_decay'] = 0.95
params['num_filters'] = 16
params['kernel_size'] = 3
params['pooling_size'] = 2
params['dropout'] = 0.1
params['batch_norm']= True
params['image_size'] = 128

unet_model = UnetBrain(**params).model
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0000001, verbose=1),
    ModelCheckpoint('model-UnetBrain.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = unet_model.fit(X_train, y_train, batch_size=32, epochs=150, callbacks=callbacks,\
                    validation_data=(X_val, y_val))

# plot training loss history
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_jaccard_discance"], label="jaccard_discance")
plt.plot( np.argmin(results.history["val_jaccard_discance"]), np.min(results.history["val_jaccard_discance"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("jaccard_discance")
plt.legend();
plt.show()

# use trained model on test data
# load the model parameters
unet_model.load_weights('model-UnetBrain.h5')
pred_test = unet_model.predict(test_data)

# example plot
fig, axes = plt.subplots(nrows=2, ncols=2)
ax = axes.ravel()
ax[0].imshow(np.squeeze(test_data[10]))
ax[0].set_title("image")
ax[1].imshow(np.squeeze(pred_test[10]))
ax[1].set_title("pred label")

ax[2].imshow(np.squeeze(pred_test[10]>0.5))
ax[2].set_title("pred label binary")

ax[3].imshow(np.squeeze(test_label[10]))
ax[3].set_title("real label")
plt.show()
