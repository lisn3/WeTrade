import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
import random
from sklearn.metrics import mean_squared_error
import os

def myloss_train(y_true, y_pred):
    phase='train'
    loss = - (0.505 * y_true * tf.log(y_pred+pow(10.0, -9)) + 0.495 * (1 - y_true) * tf.log(1 - y_pred+pow(10.0,-9)))
    return loss


study_type='XR_ELBOW'
test_data_raw = np.load('data/MURA_{}_valid.npy'.format(study_type))
test_labels = np.load('data/MURA_{}_valid_lab.npy'.format(study_type))
model_path = '/home/lsn/MURA-TL/mura_densenet/models/TL-whole.h5'
model=load_model(model_path, custom_objects={'myloss_train': myloss_train})
scores=model.evaluate(test_data_raw, test_labels)
print(scores)