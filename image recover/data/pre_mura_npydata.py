import numpy as np
import sys
import os
from scipy.io import loadmat
import pandas as pd
import cv2
import glob
import matplotlib._png as png
import argparse

PIXEL_MEANS =(0.485, 0.456, 0.406)  #RGB format mean and variances
PIXEL_STDS = (0.229, 0.224, 0.225)

def normalize(image):
    #image = image/255.0
    #img =image-np.array(PIXEL_MEANS)
    #img = img/np.array(PIXEL_STDS)
    img = cv2.resize(image, (224, 224))
    return img

def mura_preprocess(train_path, study_type):
    #train_path = "MURA-v1.1/"
    csv_train_filess = os.path.join( train_path, "MURA-v1.1/", "train_{}.csv".format(study_type))
    csv_valid_filess = os.path.join( train_path, "MURA-v1.1/", "valid_{}.csv".format(study_type))

    train_df = pd.read_csv(csv_train_filess, names=['img','count', 'label'], header=None)
    valid_df = pd.read_csv(csv_valid_filess, names=['img', 'count','label'], header=None)

    train_img_paths = train_df.img.values.tolist()
    valid_img_paths = valid_df.img.values.tolist()
    train_labels_patient = train_df.label.values.tolist()
    valid_labels_patient = valid_df.label.values.tolist()
    train_data_list = []
    train_labels = []
    valid_data_list = []
    valid_labels = []


    for i in range(len(train_img_paths)):
        patient_dir = os.path.join(train_path, train_img_paths[i])
        msg = "\r Loading: %s (%d/%d)    " % (patient_dir, i + 1, len(train_img_paths))
        sys.stdout.write(msg)
        sys.stdout.flush()
        train_data_patient = []
        for f in glob.glob(patient_dir + "*"):
            train_img = png.read_png_int(f)
            if len(train_img.shape)==2:
                train_img=np.stack((train_img,)*3, -1)
            train_img = normalize(np.array(train_img))
            # you can replace 256 with other number but any number greater then 256 will exceed the memory limit of 12GB
            train_data_patient.append(train_img)
        train_data_list.extend(train_data_patient)
        for _ in range(len(train_data_patient)):
            lst = [0, 0]
            lst[train_labels_patient[i]] = 1
            #lst = train_labels_patient[i]
            train_labels.append(lst)
    train_data = np.asarray(train_data_list)

    for i in range(len(valid_img_paths)):
        patient_dir = os.path.join(train_path, valid_img_paths[i])
        msg = "\r Loading: %s (%d/%d)     " % (patient_dir, i + 1, len(valid_img_paths))
        sys.stdout.write(msg)
        sys.stdout.flush()
        valid_data_patient = []
        for f in glob.glob(patient_dir + "*"):
            valid_img = png.read_png_int(f)
            if len(valid_img.shape)==2:
                valid_img = np.stack((valid_img,) * 3, -1)
            valid_img = normalize(np.array(valid_img))
            valid_data_patient.append(valid_img)
        valid_data_list.extend(valid_data_patient)
        for _ in range(len(valid_data_patient)):
            lst = [0, 0]
            lst[valid_labels_patient[i]] = 1
            valid_labels.append(lst)
    valid_data = np.asarray(valid_data_list)

    np.save('MURA_{}_train.npy'.format(study_type), np.array(train_data))
    print("【Saving data to MURA_{}_train.npy】".format(study_type))
    np.save('MURA_{}_valid.npy'.format(study_type), np.array(valid_data))
    print("【Saving data to MURA_{}_valid.npy】".format(study_type))
    np.save('MURA_{}_train_lab.npy'.format(study_type), np.array(train_labels))
    np.save('MURA_{}_valid_lab.npy'.format(study_type), np.array(valid_labels))

parser=argparse.ArgumentParser()
parser.add_argument('--data_path', default='/home/lsn/MURA-TL/mura_densenet/dataloader/')
parser.add_argument('--study_type',default='XR_ELBOW')

args=parser.parse_args()

mura_preprocess(args.data_path, args.study_type)