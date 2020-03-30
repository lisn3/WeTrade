import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
import random
from sklearn.metrics import mean_squared_error
import os

PIXEL_MEANS =(0.485, 0.456, 0.406)  #RGB format mean and variances
PIXEL_STDS = (0.229, 0.224, 0.225)

def myloss_train(y_true, y_pred):
    phase='train'
    loss = - (0.505 * y_true * tf.log(y_pred+pow(10.0, -9)) + 0.495 * (1 - y_true) * tf.log(1 - y_pred+pow(10.0,-9)))
    return loss


def normalize(image):
    image = image/255.0
    image = image-np.array(PIXEL_MEANS)
    image = image/np.array(PIXEL_STDS)
    return image


def test_rec(n_rec, model_path, testdata, data_name):
    model=load_model(model_path)
    print('\n【Load AutoEncoder to reconstruct {} image'.format(data_name))
    x=testdata

    pred_img = model.predict(x)
    pred_img = pred_img.reshape(len(x), 224, 224, 3)
    pred_img = pred_img.astype('uint8')

    np.save('rec_{}.npy'.format(data_name),pred_img)
    print('\n【Saving recovered image to rec_{}.npy】'.format(data_name))

    index = random.sample(range(0, len(x)), n_rec)

    err=[]
    tt=0
    for i in index:
        plt.subplot(2, n_rec, tt + 1)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, n_rec, n_rec + tt + 1)
        plt.imshow(pred_img[i])
        plt.xticks([])
        plt.yticks([])

        imgerr=mean_squared_error(np.ravel(x[i]), np.ravel(pred_img[i]))
        imgerr=np.round(imgerr,2)
        plt.xlabel(str(imgerr))

        err.append(imgerr)
        tt=tt+1
    plt.savefig("figs/{}_rec.png".format(data_name))
    plt.clf()

    print("\nMean Square Error: ", err)
    return pred_img


def cls_rec(n_rec, model_path, ori_img, rec_img, label):
    model=load_model(model_path, custom_objects={'myloss_train': myloss_train})
    # TODO: DenseNet evaluate image need to normalize it
    ori_img_nor=np.array([normalize(img) for img in ori_img])
    rec_img_nor=np.array([normalize(img) for img in rec_img])
    scores=model.evaluate(ori_img_nor, label)
    print('Original img Testing scores:', scores)
    scores = model.evaluate(rec_img_nor, label)
    print('Reconstructed img Testing scores:', scores)

    prob_ori=model.predict(ori_img_nor.reshape(len(ori_img_nor), 224,224,3))
    prob_rec=model.predict(rec_img_nor.reshape(len(rec_img_nor),224,224,3))

    err = []
    true_label=[]
    idx=50
    for i in range(n_rec):
        while np.argmax(prob_ori[idx])==np.argmax(prob_rec[idx]):
            idx=idx+1
        plt.subplot(2, n_rec, i + 1)
        plt.imshow(ori_img[idx].astype('uint8'))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(str(np.argmax(np.round(np.ravel(prob_ori[idx],2)))))

        plt.subplot(2, n_rec, n_rec + i + 1)
        plt.imshow(rec_img[idx].astype('uint8'))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(str(np.argmax(np.round(np.ravel(prob_rec[idx],2)))))

        imgerr = mean_squared_error(np.ravel(ori_img[idx]), np.ravel(rec_img[idx]))
        imgerr = np.round(imgerr, 2)
        err.append(imgerr)
        true_label.append(np.argmax(label[idx]))
        idx=idx+1

    plt.savefig("figs/classify_rec/{}_rec.png".format(n_rec))
    plt.clf()
    print("\n True label: ", true_label)
    print("\nMean Square Error: ", err)


if __name__ == '__main__':
    path='data/'
    parser=argparse.ArgumentParser()
    parser.add_argument('--train_data',default='XR_FOREARM')
    parser.add_argument('--test_data', default='XR_ELBOW')
    args=parser.parse_args()

    train_data=args.train_data
    test_data=args.test_data
    n_rec=5

    model_path='models/ae_vgg_{}.h5'.format(train_data)

    imgpath=os.path.join(path, "MURA_{}_valid.npy".format(test_data))
    labpath=os.path.join(path, "MURA_{}_valid_lab.npy".format(test_data))
    testdata=np.load(imgpath)
    print('\n【Loading {} data】'.format(test_data))
    if os.path.exists('rec_{}.npy'.format(test_data)):
        rec_img=np.load('rec_{}.npy'.format(test_data))
    else:
        rec_img=test_rec(n_rec, model_path, testdata, test_data)

    print('\n【Using TL-whole model to classify data.】')
    TL_whole_path = '/home/lsn/MURA-TL/mura_densenet/models/TL-whole.h5'
    label=np.load(labpath)
    cls_rec(10, TL_whole_path, testdata, rec_img, label)


