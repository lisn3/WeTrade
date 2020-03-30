import os
import argparse
import numpy as np
import tensorflow as tf
import time
from dataloader.data_generator import load_data
from models.densenet import build_transfer_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#fig_dir = os.path.join('figs')
#MODEL_PATH=os.path.join('model')

if __name__ == '__main__':
    ##  
    train_params = {'initial_lr': 0.1,
                    'weight_decay': 1e-4,
                    'batch_size': 10,
                    'total_epoch': 300,
                    'keep_prob': 0.8
                    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--study_type', default='XR_ELBOW')
    parser.add_argument('--fig_name', default='Figs')
    parser.add_argument('--save',default=False)
    parser.add_argument('--model_name', default='TL-whole')

    #parameters for pre-process dataset
    parser.add_argument('--train_data_num', default=0.2, type=float,
                        help='sampled data numbers')  #sampling 20% data to fine-tuning
    parser.add_argument('--provider_type', default='None',
                        help='define the provided data quality')
    parser.add_argument('--mislabel_rate', default=0, type=float)
    parser.add_argument('--gauss_sigma', default=0.1, type=float)
    parser.add_argument('--pepper_amount', default=0.1, type=float)
    parser.add_argument('--new_size', default=112, type=int)  #used for down sampling
    parser.add_argument('--unmatch_rate', default=0.1, type=float)

    #parameters for training the model
    parser.add_argument('--pre_trained', default=True, type=bool)
    parser.add_argument('--fix_num', default=368, type=int,
                        help='fixed layers in densenet')
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--epoch', default=50, type=int,
                        help='train epochs')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size')
    args = parser.parse_args()

    train_data, train_label, test_data, test_label = load_data(args)
    build_transfer_model(args.pre_trained, args.fix_num, train_data, train_label, test_data, test_label,
                         args.fig_name, args.model_name, args.lr, args.epoch, args.batch_size, args.save)


