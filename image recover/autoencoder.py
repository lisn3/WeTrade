import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import gzip
import argparse
import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Flatten,Dense
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet169
import os
from test import test_rec

"""
 When using VGG pre-trained parameters, better img pixel is 0~255. without normalization 
"""

def mymodel1(input_img, TRAINABLE=False):

    base_model = VGG16(weights='imagenet')

    for layer in base_model.layers:
        layer.trainable=TRAINABLE
    
#-------------------encoder---------------------------- 
#--------(pretrained & trainable if selected)----------

#    block1
    x=base_model.get_layer('block1_conv1')(input_img)
    x=base_model.get_layer('block1_conv2')(x)
    x=base_model.get_layer('block1_pool')(x)

#    block2
    x=base_model.get_layer('block2_conv1')(x)
    x=base_model.get_layer('block2_conv2')(x)
    x=base_model.get_layer('block2_pool')(x)

#    block3
    x=base_model.get_layer('block3_conv1')(x)
    x=base_model.get_layer('block3_conv2')(x)
    x=base_model.get_layer('block3_conv3')(x)    
    x=base_model.get_layer('block3_pool')(x)

#    block4
    x=base_model.get_layer('block4_conv1')(x)
    x=base_model.get_layer('block4_conv2')(x)
    x=base_model.get_layer('block4_conv3')(x)    
    x=base_model.get_layer('block4_pool')(x)

#    block5
    x=base_model.get_layer('block5_conv1')(x)
    x=base_model.get_layer('block5_conv2')(x)
    x=base_model.get_layer('block5_conv3')(x)
     
    
#--------latent space (trainable) ------------
    x=base_model.get_layer('block5_pool')(x)     
    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='latent')(x)
    x = UpSampling2D((2,2))(x)  
    
#--------------decoder (trainable)----------- 
        
  # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock5_conv3')(x)
    x = UpSampling2D((2,2))(x)

  # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dblock4_conv3')(x)
    x = UpSampling2D((2,2))(x)

  # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dblock3_conv3')(x)
    x = UpSampling2D((2,2))(x)     
     
  # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dblock2_conv3')(x)
    x = UpSampling2D((2,2))(x)        
 
  # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dblock1_conv1')(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same', name='dblock1_conv3')(x)
#    x = UpSampling2D((2,2))(x) 
    
    return x


input_image = Input(shape = (224, 224, 3))

autoencoder=Model(input_image, mymodel1(input_image))
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
autoencoder.compile(loss='mean_squared_error', optimizer = adam)
autoencoder.summary()

parser=argparse.ArgumentParser()
parser.add_argument('--train_data', default='XR_FOREARM')
parser.add_argument('--test_data', default='XR_ELBOW')
parser.add_argument('--n_rec', default=5, type=int)
args=parser.parse_args()

n_rec=args.n_rec
#path='/home/lsn/MURA-TL/mura_densenet/dataloader/NPY_img'
path='data/'
# a small toy dataset from imagenet
if args.train_data=='imagenet':
    x=np.load("data/train_imagenet.npy")
else:
    imgpath=os.path.join(path, "MURA_{}_train.npy".format(args.train_data))
    x=np.load(imgpath)

print("\n【Training on {} data.】\n".format(args.train_data))
x_train,x_val = train_test_split(x, test_size=0.2, random_state=123)

model_path='models/ae_vgg_{}.h5'.format(args.train_data)
if os.path.exists(model_path):
    print("\n【Restoring model】")
    autoencoder=load_model(model_path)

autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=16,epochs=1,verbose=1,validation_data=(x_val, x_val))

autoencoder.save(model_path)
print('\n【Saving model to {}】'.format(model_path))

print("\n\n【Start testing】\n")
print("\n【Testing on validation data.】")
if args.train_data=='imagenet':
    imgpath="data/train_imagenet.npy"
else:
    imgpath=os.path.join(path, "MURA_{}_valid.npy".format(args.train_data))
testdata=np.load(imgpath)
rec=test_rec(n_rec, model_path, testdata, args.train_data)

print("\n【Testing on target data.】")
imgpath=os.path.join(path, "MURA_{}_train.npy".format(args.test_data))
testdata=np.load(imgpath)
rec=test_rec(n_rec, model_path, testdata, args.test_data)

