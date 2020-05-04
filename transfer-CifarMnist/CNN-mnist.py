import numpy as np
import os
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist,fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import random
import matplotlib.pyplot as plt
import cv2
from scipy import misc
from skimage import util
from collections import Counter

# 全局变量
batch_size = 64
nb_classes = 10
epochs = 10
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
train_data_num=60000
test_data_num=10000
unbalance_list=[50,50,50,50,50,50,99,99,1,1]

def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe

def hot2single(y):
    le=len(y)
    rlt=0
    for i in range(le):
        if y[i]==1:
            rlt=i
            break
    return rlt

def plot_training(history,fig_name):
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

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 根据不同的backend定下不同的格式
'''
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
'''
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

def wholedata(X_data, y_data):
    X_train = X_data.astype('float32')
    X_train /= 255.0

    y_train_ohe = np.array([tran_y(y_data[i]) for i in range(len(y_data))])
    y_train_ohe = y_train_ohe.astype('float32')

    print(X_train.shape)
    return X_train, y_train_ohe



def random_data(X_data, y_data, data_num):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]
    X_train=X_train.astype('float32')
    X_train /= 255.0

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
    X_train = X_train.astype('float32')
    X_train /= 255.0
    return X_train, y_train_ohe

def addGaussianNoise(image,percentage,means,sigma):  #定义添加高斯噪声的函数
    G_Noiseimg = image
    G_NoiseNum=int(percentage*image.shape[0]*image.shape[1])
    random_ind = random.sample(range(0, image.shape[0] * image.shape[1]), G_NoiseNum)
    for t in random_ind:
        temp_x = t // image.shape[1]
        temp_y = t % image.shape[1]
        G_Noiseimg[temp_x][temp_y] = G_Noiseimg[temp_x][temp_y]+random.gauss(means,sigma)
        if G_Noiseimg[temp_x][temp_y]<0:
            G_Noiseimg[temp_x][temp_y]=0
        if G_Noiseimg[temp_x][temp_y]>255:
            G_Noiseimg[temp_x][temp_y]=255
    return G_Noiseimg

def SaltAndPepper(src,percetage, salt):   #定义添加椒盐噪声的函数
    SP_NoiseImg=src
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    random_ind=random.sample(range(0, src.shape[0]*src.shape[1]), SP_NoiseNum)
    for t in random_ind:
        randX=t//src.shape[1]
        randY=t%src.shape[1]
        if salt == 0:
            SP_NoiseImg[randX, randY] = 0
        else:
            SP_NoiseImg[randX, randY] = 255
    return SP_NoiseImg



def noisydata(X_data, y_data, data_num, name, sigma, percentage):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]

    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_train_ohe = y_train_ohe.astype('float32')

    X_train = X_train.astype('float32')
    X_train /= 255.0

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

    w, h = X_train[0].shape[0], X_train[0].shape[1]
    rew,reh= round(w*smallrate),round(h*smallrate)
    #plt.imshow(np.reshape(X_train[0],[img_rows,img_cols]), cmap='gray')
    #plt.show()
    X_1 = np.array([cv2.resize(i, (new_w, new_h)) for i in X_train])
    print("down sampling image shape: ", X_1.shape)
    #plt.imshow(np.reshape(X_1[0],[new_w,new_h]), cmap='gray')
    #plt.show()
    X_ = [cv2.resize(i, (img_rows, img_cols)) for i in X_1]
    X_1= np.concatenate([arr[np.newaxis] for arr in X_]).astype('float32')
    X_1=X_1.reshape(X_1.shape[0], img_rows, img_cols, 1)
    X_train = X_1/255.0

    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_train_ohe = y_train_ohe.astype('float32')

    print("feed into model shape: ", X_train.shape)
    return X_train, y_train_ohe


def unmatch_data(unmatch_per, oriset='mnist', name='FashionMNIST'):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    size1 = round(train_data_num * (1 - unmatch_per))
    X1, y1 = random_data(X_train, y_train, size1)

    size2 = round(train_data_num * unmatch_per)
    if name == 'FashionMNIST':
        (X_, y_), (X_test1, y_test1) = fashion_mnist.load_data()
        X2, y2 = random_data(X_, y_, size2)

    print("Unmatched dataset shape: ", X2.shape)
    if unmatch_per==1:
        X_train=X2
        y_train=y2
    else:
        X_train=np.concatenate([X1, X2])
        y_train=np.concatenate([y1, y2])

    X_train=X_train.reshape(X_train.shape[0],img_rows, img_cols, 1)

    print("feed into model shape: ", X_train.shape)
    return X_train, y_train


def unbalance(X_data, y_data, missclass):
    new_ind = np.random.permutation(len(X_data))
    X_data = X_data[new_ind]
    y_data = y_data[new_ind]

    X_train = []
    y_train = []
    num = 0
    for i in range(len(X_data)):
        lab = y_data[i]
        if lab>=missclass:
            X_train.append(X_data[i])
            y_train.append(y_data[i])
            num=num+1
        if num==train_data_num:
            break
    
    X_train = np.array(X_train).astype('float32')
    X_train /= 255.0

    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_train_ohe = y_train_ohe.astype('float32')

    print("unblance data counting: ", Counter(y_train))
    return X_train, y_train_ohe

y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
y_test_ohe = y_test_ohe.astype('float32')

test_ind = random.sample(range(0, len(X_test)), test_data_num)
X_test, y_test = X_test[test_ind], y_test_ohe[test_ind]
X_test = X_test.astype('float32')
X_test/= 255.0

#X_train = X_train.astype('float32')
#
# 转换为one_hot类型
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)


#X_train_data, y_train_data = random_data(X_train, y_train,train_data_num)
#X_train_data, y_train_data= mislabel(X_train, y_train, train_data_num, 0.8)
X_train_data, y_train_data= noisydata(X_train, y_train, train_data_num, 'gaussian', 0.49, 0)
#X_train_data, y_train_data= noisydata(X_train, y_train, train_data_num, 'pepper', 0.9)  #data missing
#X_train_data, y_train_data= down_sample(X_train, y_train, train_data_num, 6, 6) #缩小50%
#X_train_data, y_train_data= unmatch_data( 0.9,'mnist','FashionMNIST') #50%的数据是不匹配的
#X_train_data, y_train_data = unbalance(X_train, y_train, 7) # missclass=1
#X_train_data, y_train_data= wholedata(X_train, y_train)

print("training number:",X_train_data.shape[0])


# 构建模型--CNN

model = Sequential()
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape))  # 卷积层1
model.add(Activation('relu'))  # 激活层
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # 卷积层2
model.add(Activation('relu'))  # 激活层
model.add(MaxPooling2D(pool_size=pool_size))  # 池化层
model.add(Dropout(0.25))  # 神经元随机失活
model.add(Flatten())  # 拉成一维数据
model.add(Dense(128))  # 全连接层1
model.add(Activation('relu'))  # 激活层
model.add(Dropout(0.5))  # 随机失活
model.add(Dense(nb_classes))  # 全连接层2
model.add(Activation('softmax'))  # Softmax评分

print(model.summary())
# 编译模型
model.compile(loss='categorical_crossentropy',
              #optimizer='adadelta',
              optimizer='adam',
              metrics=['accuracy'])

history_ft=model.fit(X_train_data, y_train_data, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, y_test))
# 评估模型
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
plot_training(history_ft, 'gauss_whole')

