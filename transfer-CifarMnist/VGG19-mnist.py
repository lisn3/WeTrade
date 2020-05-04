# encoding: utf-8
from keras.datasets import mnist,fashion_mnist
import gc
import os
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD, Adam
import time
from skimage import util
import matplotlib.pyplot as plt
from collections import Counter


import cv2
import h5py as h5py
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator

BATCH=8
steps_per_epoch=100
EPOCHS=30
train_data_num=500
test_data_num=1000
layer_num=4  #fixed 1 block

unbalance_list=[50,50,50,50,50,50,99,99,1,1]

ishape=48

#change model in line 70

##由于输入层需要10个节点，所以最好把目标数字0-9做成one Hot编码的形式。
def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe

def plot_training(history, fig_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
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
    plt.savefig(os.path.join('resulting_figs/{}.png'.format(fig_name)))
    plt.clf()

def process_x(X_data):
    X_ = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_data]
    X_data = np.concatenate([arr[np.newaxis] for arr in X_]).astype('float32')
    X_data /= 255.0
    return X_data

def data_gene(X_data):
    datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
    datagen.fit(X_data)
    return datagen

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


def noisydata(X_data, y_data, data_num, name, sigma, percentage):
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
    
    return X_train, y_train_ohe


def down_sample(X_data, y_data, data_num, new_w, new_h):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]
    
    w,h=X_train[0].shape[0], X_train[0].shape[1]
    #rew,reh= round(w*smallrate),round(h*smallrate)
    plt.imshow(X_train[0], cmap='gray')
    plt.show()
    X_1 = np.array([cv2.resize(i, (new_w, new_h)) for i in X_train])
    print("down sampling image shape: ",X_1.shape)
    plt.imshow(X_1[0], cmap='gray')
    plt.show()
    X_train=process_x(X_1)

    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_train_ohe = y_train_ohe.astype('float32')

    print("feed into model shape: ",X_train.shape)
    return X_train, y_train_ohe

def unmatch_data(unmatch_per, oriset='mnist',name='FashionMNIST'):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    size1=round(train_data_num*(1-unmatch_per))
    X1, y1=random_data(X_train, y_train, size1)

    size2=round(train_data_num* unmatch_per)
    if name=='FashionMNIST':
        (X_, y_), (X_test1, y_test1) = fashion_mnist.load_data()
        X2, y2= random_data(X_, y_, size2)

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

    print("unblance data counting: ",Counter(y_train))
    return X_train, y_train_ohe
# 如果硬件配置较高，比如主机具备32GB以上内存，GPU具备8GB以上显存，可以适当增大这个值。VGG要求至少48像素

(X_train, y_train), (X_test, y_test) = mnist.load_data()
'''
X_train = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_train]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')
X_train /= 255.0

X_test = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_test]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')
X_test /= 255.0

y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
y_train_ohe = y_train_ohe.astype('float32')
'''

y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
y_test_ohe = y_test_ohe.astype('float32')

test_ind = random.sample(range(0, len(X_test)), test_data_num)
X_test, y_test = X_test[test_ind], y_test_ohe[test_ind]
X_test=process_x(X_test)

# VGG16 全参重训迁移学习

# 很多时候需要多次回收垃圾才能彻底收回内存。如果不行，重新启动单独执行下面的模型
for i in range(10):
    gc.collect()

#这里可以选择VGG16,VGG19,ResNet等模型
model_vgg = VGG19(include_top = False, weights = 'imagenet', input_shape = (ishape, ishape, 3))

for layer in model_vgg.layers[0:layer_num]:  #
        layer.trainable = False
model = Flatten()(model_vgg.output)
model = Dense(128, activation='relu', name='fc1')(model)
model = Dropout(0.5)(model)
model = Dense(10, activation = 'softmax', name='prediction')(model)
model_vgg_mnist_pretrain = Model(model_vgg.input, model, name = 'vgg16_pretrain')
print (model_vgg_mnist_pretrain.summary())


##训练模型参数。
time1 = time.time()
sgd = SGD(lr=0.001, decay = 1e-5,momentum=0.9, nesterov=True)   #0.0001  fixed 7
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)   #0.0001  fix4
model_vgg_mnist_pretrain.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

#图片和标签来源：1，正常数据  2.mislabel 3.高斯噪声  4.像素缺失  5.降低分辨率  6.不匹配的数据
#X_train_data, y_train_data = random_data(X_train, y_train,train_data_num)
#X_train_data, y_train_data= mislabel(X_train, y_train, train_data_num, 0.7)
X_train_data, y_train_data= noisydata(X_train, y_train, train_data_num, 'gaussian', 0.49, 0)
#X_train_data, y_train_data= noisydata(X_train, y_train, train_data_num, 'pepper', 0.9)  #data missing
#X_train_data, y_train_data= down_sample(X_train, y_train, train_data_num, 6, 6) #缩小50%
#X_train_data, y_train_data= unmatch_data(0.95,'mnist','FashionMNIST') #50%的数据是不匹配的
#X_train_data, y_train_data = unbalance(X_train, y_train, 8)

datagen=data_gene(X_train_data)
history_ft=model_vgg_mnist_pretrain.fit_generator(datagen.flow(X_train_data, y_train_data, batch_size=BATCH),
                             validation_data = (X_test, y_test), steps_per_epoch=steps_per_epoch, epochs = EPOCHS)


#######在测试集上评价模型精确度
scores=model_vgg_mnist_pretrain.evaluate(X_test,y_test,verbose=0)

#####打印精确度
print ("testing score",scores)


time2 = time.time()
print("train_data_num: %d, fixed layer number: %d" % (train_data_num,layer_num-1))
print(u'ok,结束!')
print(u'总共耗时：' + str(time2 - time1) + 's')

plot_training(history_ft, 'noise_tl')
