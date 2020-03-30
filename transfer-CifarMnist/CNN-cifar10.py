import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import time, pickle
from keras.utils import to_categorical
from skimage import util
from tflearn.datasets import oxflower17

batch_size = 32
nb_classes = 10
epochs = 50
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

train_data_num=2000
test_data_num=10000

# The data, split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)

#X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train /= 255
X_test /= 255

test_ind = random.sample(range(0, len(X_test)), test_data_num)
X_test, y_test = X_test[test_ind], y_test[test_ind]

def process_x(X_data):
    X_data = X_data.astype('float32')
    X_data /= 255.0
    return X_data


def random_data(X_data, y_data, data_num):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]
    X_train=process_x(X_train)

    y_train_ohe = to_categorical(y_train, nb_classes)

    print(X_train.shape)
    return X_train, y_train_ohe


def mislabel(X_data, y_data, data_num, rate):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]
    mis_num=round(data_num*rate)
    for i in range(mis_num):
        a=np.random.randint(9)
        b=(y_train[i]+a)%10
        y_train[i]=b

    y_train_ohe = to_categorical(y_train, nb_classes)
    X_train=process_x(X_train)
    return X_train, y_train_ohe


def noisydata(X_data, y_data, data_num, percentage, name, mean=0, sigma=1):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]

    y_train_ohe = to_categorical(y_train, nb_classes)

    X_train=process_x(X_train)

    if name=='gaussian':
        for i in range(len(X_train)):
            img=X_train[i]
            Noiseimg=util.random_noise(img, mode='gaussian',var=sigma)
            X_train[i]=Noiseimg


    if name=='pepper':
        for i in range(len(X_train)):
            img=X_train[i]
            Noiseimg=util.random_noise(img, mode='pepper',amount=percentage)
            X_train[i]=Noiseimg

    #X_train=process_x(X_train)
    return X_train, y_train_ohe

def down_sample(X_data, y_data, data_num, new_w, new_h):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]
    
    w,h=X_train[0].shape[0], X_train[0].shape[1]
    #rew,reh= round(w*smallrate),round(h*smallrate)
    #plt.imshow(X_train[0], cmap='gray')
    #plt.show()
    X_1 = np.array([cv2.resize(i, (new_w, new_h)) for i in X_train])
    print("down sampling image shape: ",X_1.shape)
    #plt.imshow(X_1[0], cmap='gray')
    X_1= np.array([cv2.resize(i, (32 , 32)) for i in X_1])
    X_train=process_x(X_1)

    y_train_ohe = to_categorical(y_train, nb_classes)

    print("feed into model shape: ",X_train.shape)
    return X_train, y_train_ohe

def unmatch_data(X_data, y_data, unmatch_per, oriset='cifar10',name='oxflower17'):

    size1=round(train_data_num*(1-unmatch_per))
    X1, y1=random_data(X_data, y_data, size1)

    size2=round(train_data_num* unmatch_per)
    if name=='FashionMNIST':
        (X_, y_), (X_test1, y_test1) = fashion_mnist.load_data()
        X2, y2= random_data(X_, y_, size2)
    if name == 'oxflower17':
        X_, y_ = oxflower17.load_data(dirname="17flowers", one_hot=False)
        X_ = np.array([cv2.resize(i, (32 , 32)) for i in X_])
        y_ = np.array([random.randint(0, 9) for _ in range(y_.shape[0])])
        print("oxflower y shape: ", y_.shape)
        if size2 > y_.shape[0]:
        	rnd=size2//y_.shape[0]+1
        	X_new=X_
        	y_new=y_
        	for _ in range(rnd):
	            X_new=np.concatenate([X_new, X_])
	            y_new=np.concatenate([y_new, y_])
        X2, y2 = random_data(X_new, y_new, size2)

    print("Unmatched dataset shape: ",X2.shape)
    return np.concatenate([X1, X2]), np.concatenate([y1, y2])

def unbalance(X_data, y_data, missclass):
    new_ind = np.random.permutation(len(X_data))
    X_data = X_data[new_ind]
    y_data = y_data[new_ind]

    X_train = []
    y_train = []
    num = 0
    for i in range(len(X_data)):
        lab = y_data[i]
        if lab >= missclass:
            X_train.append(X_data[i])
            y_train.append(y_data[i])
            num = num + 1
        if num == train_data_num:
            break
    X_train = process_x(np.array(X_train))

    y_train_ohe = to_categorical(y_train, nb_classes)

   # print("unblance data counting: ",Counter(np.array(y_train)))
    return X_train, y_train_ohe



model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])



datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)



# gaussian var
#alp=[0.0001, 0.0025, 0.01, 0.04, 0.09,0.25, 0.49, 0.81, 1,5,10,20]
# pepper amount
#alp=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#resize
#alp=[24, 16, 8, 4]
#unmatch
#alp=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.95, 0.98,1]
alp=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
for i in alp:
	print("********************alp*************:",i)
	#X_train_data, y_train_data = random_data(X_train, y_train,train_data_num)
	X_train_data, y_train_data= mislabel(X_train, y_train, train_data_num, i)
	#X_train_data, y_train_data= noisydata(X_train, y_train, train_data_num, 0.8, 'gaussian', 0, i)
	#X_train_data, y_train_data= noisydata(X_train, y_train, train_data_num, i, 'pepper')  #data missing
	#X_train_data, y_train_data= down_sample(X_train, y_train, train_data_num, i, i) #
	#X_train_data, y_train_data= unmatch_data(X_train, y_train, i,'cifar10','oxflower17') #
	#X_train_data, y_train_data = unbalance(X_train, y_train, i)



	datagen.fit(X_train_data)

	# Fit the model on the batches generated by datagen.flow().
	model.fit_generator(datagen.flow(X_train_data, y_train_data,
	                                 batch_size=batch_size),
	                    epochs=epochs,
	                    validation_data=(X_test, y_test),
	                    workers=4)





	# Score trained model.
	scores = model.evaluate(X_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])