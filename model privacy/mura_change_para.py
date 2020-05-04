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

def change(model, change_rt):
    layer=model.layers[layer_num]
    print("layer {}".format(layer.name))
    ori_w=np.array(layer.get_weights())
    #print("layer weight: ", ori_w)
    new_w=ori_w*change_rt
    layer.set_weights(new_w)
    score=model.evaluate(test_data_raw, test_labels)
    print("\n\n#####When change rate is {}, the testing score is {}".format(change_rt, score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--layer', default='367', type=int)
    parser.add_argument('--para', default='0.0005', type=float)
    args = parser.parse_args()
    layer_num=args.layer
    change_rate=args.para

    print('\n【Restoring TL-whole model to classify data.】')
    ori_model_path = '/home/lsn/MURA-TL/mura_densenet/models/TL-whole.h5'
    study_type='XR_ELBOW'

    model=load_model(ori_model_path, custom_objects={'myloss_train': myloss_train})
    # TODO: DenseNet evaluate image need to normalize it
    datapath="/home/lsn/MURA-TL/mura_densenet/dataloader/"
    train_data_raw = np.load(os.path.join(datapath, 'NPY_img/train_{}_img.npy'.format(study_type)))
    train_labels = np.load(os.path.join(datapath, 'NPY_lab/train_{}_lab.npy'.format(study_type)))
    test_data_raw = np.load(os.path.join(datapath, 'NPY_img/valid_{}_img.npy'.format(study_type)))
    test_labels = np.load(os.path.join(datapath, 'NPY_lab/valid_{}_lab.npy'.format(study_type)))

    scores=model.evaluate(test_data_raw, test_labels)
    print('Original img Testing scores:', scores)  #73.83

    #change_rate = [0.0005, 0.005, 0.05, 0.5, 1, 5, 50]

    change(model, change_rate)



    
    



