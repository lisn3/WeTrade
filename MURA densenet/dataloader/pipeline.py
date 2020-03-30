import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import cv2
from PIL import Image
import argparse

#from torchvision.datasets.folder import pil_loader

data_cat = ['train', 'valid'] # data categories
PIXEL_MEANS =(0.485, 0.456, 0.406)  #RGB format mean and variances
PIXEL_STDS = (0.229, 0.224, 0.225)

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_study_level_data(study_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes, 
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset 
    """
    study_data = {}
    image_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:
        BASE_DIR = 'MURA-v1.1/%s/%s/' % (phase, study_type)
        patients = list(os.walk(BASE_DIR))[0][1] # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label']) #study level data
        image_data[phase] = pd.DataFrame(columns=['Path', 'label'])          # image level data
        i = 0
        idx = 0
        for patient in tqdm(patients): # for each patient folder
            for study in os.listdir(BASE_DIR + patient): # for each study in that patient folder
                label = study_label[study.split('_')[1]] # get label 0 or 1
                path = BASE_DIR + patient + '/' + study + '/' # path to this study
                countt=len(os.listdir(path))
                study_data[phase].loc[i] = [path, countt, label] # add new row
                i += 1
                for t in range(countt):
                    image_path=path + 'image%s.png'.format(t+1)
                    image_data[phase].loc[idx] = [image_path, label]
                    idx+=1
        outputpath = 'MURA-v1.1/%s_%s.csv'%(phase,study_type)
        study_data[phase].to_csv(outputpath, sep=',', index=False, header=False)
    return study_data, image_data

def gaussian_noise_layer(input_image, std):
    noise = tf.random_normal(shape=tf.shape(input_image), mean=0.0, stddev=std, dtype=tf.float32)
    noise_image = tf.cast(input_image, tf.float32) + noise
    noise_image = tf.clip_by_value(noise_image, 0, 1.0)
    return noise_image

def _parse_function(filename, label):
  image = tf.read_file(filename)
  #image_resized = tf.image.resize_images(image_decoded, [28, 28])
  img= tf.image.decode_png(image, channels=3)
  img = tf.cast(img, tf.float32)/255.0
  img = tf.image.resize(img, (224, 224))
  img = tf.cast(img, tf.float32)-np.array(PIXEL_MEANS)
  img = tf.cast(img, tf.float32)/np.array(PIXEL_STDS)
  return img, label

def load_img(df, batch_size):
    filenames = tf.constant(df['Path'])
    labels = tf.constant(np.array(df['label']).tolist())
    data = tf.data.Dataset.from_tensor_slices((filenames, labels))
    data = data.map(_parse_function)
    data=data.shuffle(1000).batch(batch_size)
    return data

def get_dataloaders(data, batch_size=8, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    '''
    #image_datasets = {x: ImageDataset(data[x]) for x in data_cat}
    dataloaders = {x: load_img(data[x], batch_size) for x in data_cat}
    return dataloaders

parser=argparse.ArgumentParser()
parser.add_argument('--study_type',default='XR_ELBOW')
args=parser.parse_args()

study_data, image_data = get_study_level_data(study_type=args.study_type)
