from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.datasets import cifar10
import cv2  #
from collections import Counter
import numpy as np
import random
import matplotlib.pyplot as plt
import keras
from keras.utils import plot_model
import os
import time
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from skimage import util
from tflearn.datasets import oxflower17
#os.environ['CUDA_VISIBLE_DEVICES']='0,1'

BATCH=32
EPOCH=20
steps_per_epoch=100
LR=0.001
DECAY=1e-6

fixed_layer=249

ishape=139

train_data_num=2000
test_data_num=10000

def tran_y(y):
    y_one = np.zeros(10)
    y_one[y] = 1
    return y_one


def process_x(X_data):
    X_ = [cv2.resize(i, (ishape, ishape)) for i in X_data]
    X_data = np.concatenate([arr[np.newaxis] for arr in X_]).astype('float32')
    X_data /= 255.0
    return X_data

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(epochs, acc, 'b-*',label='train acc')
    ax1.plot(epochs, val_acc, 'r-*', label='val acc')
    ax2.plot(epochs, loss, 'b-', label='train loss')
    ax2.plot(epochs, val_loss, 'r-',label='val loss')
    plt.legend(loc='upper left')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    ax1.set_title('Training and validation accuracy and loss')
    plt.show()

aug_gen = ImageDataGenerator(
    featurewise_center = False,  # set input mean to 0 over the dataset
    samplewise_center = False,  # set each sample mean to 0
    featurewise_std_normalization = False,  # divide inputs by std of the dataset
    samplewise_std_normalization = False,  # divide each input by its std
    zca_whitening = False,  # apply ZCA whitening
    rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip = True,  # randomly flip images
    vertical_flip = False,  # randomly flip images
)


def random_data(X_data, y_data, data_num):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]
    X_train=process_x(X_train)

    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_train_ohe = y_train_ohe.astype('float32')

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
    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_train_ohe = y_train_ohe.astype('float32')
    X_train=process_x(X_train)
    return X_train, y_train_ohe

def addGaussianNoise(image,percentage,means,sigma):  #定义添加高斯噪声的函数
    G_Noiseimg = image
    G_NoiseNum=int(percentage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum): 
        temp_x = np.random.randint(0,image.shape[0])
        temp_y = np.random.randint(0,image.shape[1])
        G_Noiseimg[temp_x][temp_y] = G_Noiseimg[temp_x][temp_y]+random.gauss(means,sigma)
        if G_Noiseimg[temp_x][temp_y]<0:
            G_Noiseimg[temp_x][temp_y]=0
        if G_Noiseimg[temp_x][temp_y]>255:
            G_Noiseimg[temp_x][temp_y]=255
    return G_Noiseimg

def SaltAndPepper(src,percetage, salt):   #定义添加椒盐噪声的函数
    SP_NoiseImg=src
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if salt==0:
            SP_NoiseImg[randX,randY]=0
        else:
            SP_NoiseImg[randX,randY]=255          
    return SP_NoiseImg


def noisydata(X_data, y_data, data_num, percentage, name, mean=0, sigma=1):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]

    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_train_ohe = y_train_ohe.astype('float32')

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
    #plt.show()
    X_train=process_x(X_1)

    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_train_ohe = y_train_ohe.astype('float32')

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
        y_ = np.array([random.randint(0, 9) for _ in range(y_.shape[0])])
        print("oxflower y shape: ", y_.shape)
        if size2 > y_.shape[0]:
            X_=np.concatenate([X_, X_])
            y_=np.concatenate([y_, y_])
        X2, y2 = random_data(X_, y_, size2)

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
        '''
        if unbalance_list[lab] > 0:
            X_train.append(X_data[i])
            y_train.append(y_data[i])
            unbalance_list[lab] = unbalance_list[lab] - 1
            num = num + 1
        if num == train_data_num:
            break
        '''
    X_train = process_x(X_train)

    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_train_ohe = y_train_ohe.astype('float32')

   # print("unblance data counting: ",Counter(np.array(y_train)))
    return X_train, y_train_ohe



(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
y_test_ohe = y_test_ohe.astype('float32')

test_ind = random.sample(range(0, len(X_test)), test_data_num)
X_test, y_test = X_test[test_ind], y_test_ohe[test_ind]
X_test=process_x(X_test)



model_incept = InceptionV3(weights='imagenet', include_top=False, pooling=None,
                                 input_shape=(ishape,ishape, 3),
                                 classes=10)
for layers in model_incept.layers[0:fixed_layer]:
    layers.trainable = False
model = Flatten()(model_incept.output)
model = Dense(512, activation='relu', name='fc1')(model)
model = Dropout(0.5)(model)
#model = Dense(4096, activation='relu', name='fc2')(model)
#model = Dropout(0.5)(model)
model = Dense(10, activation='softmax', name='prediction')(model)
model_incept_cifar10_pretrain = Model(inputs=model_incept.input, outputs=model, name='InceptV3_pretrain')
#print(model_incept_cifar10_pretrain.summary())

for i, layer in enumerate(model_incept.layers):
   print(i, layer.name)


time1 = time.time()
#sgd = SGD(lr=LR, decay=DECAY, momentum=0.9, nesterov=True)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)   #0.0001  fix4
model_incept_cifar10_pretrain.compile(optimizer= adam, loss='categorical_crossentropy',
                                   metrics=['accuracy'])


# gaussian var
#alp=[0.25, 0.49, 0.81]
# pepper amount
#alp=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#resize
#alp=[24, 16, 8, 4]
#unmatch
#alp=[0.5, 0.7, 0.9, 0.95, 0.98]
alp=[0]
for i in alp:
	print("********************alp*************:",i)
	X_train_data, y_train_data = random_data(X_train, y_train,train_data_num)
	#X_train_data, y_train_data= mislabel(X_train, y_train, train_data_num, i)
	#X_train_data, y_train_data= noisydata(X_train, y_train, train_data_num, 0.8, 'gaussian', 0, i)
	#X_train_data, y_train_data= noisydata(X_train, y_train, train_data_num, i, 'pepper')  #data missing
	#X_train_data, y_train_data= down_sample(X_train, y_train, train_data_num, i, i) #
	#X_train_data, y_train_data= unmatch_data(X_train, y_train, i,'cifar10','oxflower17') #
	#X_train_data, y_train_data = unbalance(X_train, y_train, i)

	'''
	history = model_incept_cifar10_pretrain.fit(X_train_data, y_train_data,
	                                         validation_data=(X_test, y_test),
	                                         epochs=EPOCH, batch_size=BATCH,
	                                         validation_split=0.1,
	                                         verbose=1,
	                                         shuffle=True)'''

	aug_gen.fit(X_train_data)
	gen = aug_gen.flow(X_train_data, y_train_data, batch_size=BATCH)
	h = model_incept_cifar10_pretrain.fit_generator(generator=gen, 
								steps_per_epoch=train_data_num//BATCH,
								epochs=EPOCH, validation_data=(X_test, y_test))


	scores=model_incept_cifar10_pretrain.evaluate(X_test,y_test,verbose=0)

	print ("testing score",scores)


	time2 = time.time()
	print("train_data_num: %d, fixed layer number: %d" % (train_data_num,fixed_layer-1))
	print(u'ok, testing over!')
	print(u' total time: ' + str(time2 - time1) + 's')
