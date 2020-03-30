from keras.applications.densenet import DenseNet169
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import tensorflow as tf
from math import pow

def calculate_w(train_label, test_label):
    # tai = total abnormal images, tni = total normal images
    #tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
    #tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
    data_cat=['train', 'valid']
    tai={}
    tni={}
    tai['train']=np.sum(np.argmax(train_label,axis=1)==1)
    tai['valid']=np.sum(np.argmax(test_label,axis=1)==1)
    tni['train']=len(train_label)-tai['train']
    tni['valid']=len(test_label)-tai['valid']
    global Wt1, Wt0
    Wt1 = {x: tni[x] / (tni[x] + tai[x]) for x in data_cat}
    Wt0 = {x: tai[x] / (tni[x] + tai[x]) for x in data_cat}

    print('total abnormal images:', tai)
    print('total normal images:', tni, '\n')
    print('Wt0 train:', Wt0['train'])
    #print('Wt0 valid:', Wt0['valid'])
    print('Wt1 train:', Wt1['train'])
    #print('Wt1 valid:', Wt1['valid'])
    return Wt1, Wt0

def myloss_train(y_true, y_pred):
    phase='train'
    loss = - (Wt1[phase] * y_true * tf.log(y_pred+pow(10.0, -9)) + Wt0[phase] * (1 - y_true) * tf.log(1 - y_pred+pow(10.0,-9)))
    return loss

def build_model(pre_trained):
    if pre_trained:
        base_model=DenseNet169(weights='imagenet', include_top=False, pooling=None,
                               input_shape=(224,224,3))
        for layers in base_model.layers[0:368]:
            layers.trainable = False
    else:
        base_model = DenseNet169(weights=None, include_top=False, pooling=None,
                                 input_shape=(224,224,3))

    model = Flatten()(base_model.output)
    model = Dense(512, activation='relu', name='fc1')(model)
    model = Dropout(0.5)(model)
    model = Dense(1, activation='sigmoid', name='prediction')(model)
    model_densenet_pretrain = Model(inputs=base_model.input, outputs=model, name='Densenet_pretrain')

    #calculate_w(y_train_data, y_test)

    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # 0.0001  fix4 #'categorical_crossentropy', #myloss_train,
    model_densenet_pretrain.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model_densenet_pretrain


def model_tl():
    return build_model(True)

def model_scratch():
    return build_model(False)





