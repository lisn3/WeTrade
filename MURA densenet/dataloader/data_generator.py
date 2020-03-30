import sys
import os
import random
from keras.datasets import cifar10, fashion_mnist
#from dataloader.preprocess import mura_preprocess
import numpy as np
from skimage import util
import cv2
import matplotlib.pyplot as plt

downsample_dir= os.path.join('figs/down_sample')

def load_data(args):
    data_path = "MURA-v1.1"
    study_type =args.study_type
    # load original data label        'NPY_img/train_{}_img.npy'.format(study_type)
    #train_data_raw, train_labels, test_data_raw, test_labels = mura_preprocess(data_path, study_type)
    train_data_raw = np.load('dataloader/NPY_img/train_{}_img.npy'.format(study_type))
    train_labels = np.load('dataloader/NPY_lab/train_{}_lab.npy'.format(study_type))
    test_data_raw = np.load('dataloader/NPY_img/valid_{}_img.npy'.format(study_type))
    test_labels = np.load('dataloader/NPY_lab/valid_{}_lab.npy'.format(study_type))
    # define different providers with different data quality
    print("【Loading data, DONE!】--total data size", len(train_data_raw))
    train_data, train_labels = data_provided(train_data_raw, train_labels, args)
    return train_data, train_labels, test_data_raw, test_labels

def data_provided(train_data_raw, train_labels, args):
    provider_type=args.provider_type
    data_num=round(args.train_data_num*len(train_data_raw))

    if provider_type == 'None':
        train_data, train_labels=random_data(train_data_raw, train_labels, data_num)

    if provider_type == 'Mislabel':
        train_data, train_labels = mislabel(train_data_raw, train_labels, data_num, args.mislabel_rate)

    if provider_type == 'Gaussian':
        train_data, train_labels = noisydata(train_data_raw, train_labels, data_num,
                                             'gaussian', args.gauss_sigma, args.pepper_amount)

    if provider_type == 'Pepper':
        train_data, train_labels = noisydata(train_data_raw, train_labels, data_num,
                                             'pepper', args.gauss_sigma, args.pepper_amount)

    if provider_type == 'Down_sample':
        train_data, train_labels = down_sample(train_data_raw, train_labels, data_num,
                                               args.new_size, args.new_size)

    if provider_type == 'Unmatch':
        train_data, train_labels = unmatch_data(train_data_raw, train_labels, data_num,
                                                args.unmatch_rate)

    elif provider_type == 'Unbalance':
        train_data, train_labels = unbalance(train_data_raw, train_labels, data_num,
                                             args.mislabel_rate)
    print("【Training data:】", provider_type, len(train_data))
    return train_data, train_labels

def random_data(X_data, y_data, data_num):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]
    print(X_train.shape)
    return X_train, y_train

def mislabel(X_data, y_data, data_num, rate):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]
    mis_num=round(data_num*rate)
    for i in range(mis_num):
        b=(y_train[i]+1)%2
        y_train[i]=b
    return X_train, y_train

def noisydata(X_data, y_data, data_num, name, sigma, percentage):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]

    #y_train_ohe = to_categorical(y_train, nb_classes)
    #X_train=process_x(X_train)

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
    return X_train, y_train

def process_x(X_data, ishape):
    X_ = [cv2.resize(i, (ishape, ishape)) for i in X_data]
    X_data = np.concatenate([arr[np.newaxis] for arr in X_]).astype('float32')
    X_data /= 255.0
    return X_data

def down_sample(X_data, y_data, data_num, new_w, new_h):
    train_ind = random.sample(range(0, len(X_data)), data_num)
    X_train, y_train = X_data[train_ind], y_data[train_ind]

    w,h=X_train[0].shape[0], X_train[0].shape[1]
    #rew,reh= round(w*smallrate),round(h*smallrate)
    plt.imshow(X_train[0])
    plt.savefig(os.path.join(downsample_dir, "x0.png"))
    plt.clf()

    X_1 = np.array([cv2.resize(i, (new_w, new_h)) for i in X_train])
    print("down sampling image shape: ",X_1.shape)
    plt.imshow(X_1[0])
    plt.savefig(os.path.join(downsample_dir, "x0_down.png"))
    plt.clf()

    print("feed into model shape: ",X_train.shape)
    return X_train, y_train


def process_y(y, size2):
    ar=np.random.randint(0,2,size2)
    rlt=[]
    for i in range(size2):
        lst=[0,0]
        lst[ar[i]]=1
        rlt.append(lst)
    return np.array(rlt)

def unmatch_data(X_data, y_data, train_data_num, unmatch_per, name='cifar10'):
    size1 = round(train_data_num * (1 - unmatch_per))
    X1, y1 = random_data(X_data, y_data, size1)

    size2 = round(train_data_num * unmatch_per)
    if name == 'FashionMNIST':
        (X_, y_), (X_test1, y_test1) = fashion_mnist.load_data()
        X2, y2 = random_data(X_, y_, size2)
    if name == 'cifar10' :
        (X_, y_), (X_test1, y_test1) = cifar10.load_data()
        X2, y2 = random_data(X_, y_, size2)
        X2=process_x(X2, 224)
        y2=process_y(y2, size2)

    print("Unmatched dataset shape: ", X2.shape)
    return np.concatenate([X1, X2]), np.concatenate([y1, y2])

def unbalance(X_data, y_data, train_data_num, missclass):
    new_ind = np.random.permutation(len(X_data))
    X_data = X_data[new_ind]
    y_data = y_data[new_ind]

    X_train = []
    y_train = []
    num = 0
    for i in range(len(X_data)):
        lab = y_data[i]
        if lab == missclass:
            X_train.append(X_data[i])
            y_train.append(y_data[i])
            num = num + 1
        if num == train_data_num:
            break

   # print("unblance data counting: ",Counter(np.array(y_train)))
    return X_train, y_train
