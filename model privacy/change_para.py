import tensorflow.compat.v1 as tf
import argparse
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import argparse
from dataloader import load_data


change_rate = [0.005, 0.05, 0.5, 5, 50] # 参数乘以4

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', default='mnist')
args = parser.parse_args()

X_train, X_test, y_train, y_test=load_data(args.data)

if args.data == 'mnist':
    ori_ckpt_path = 'model/mnist_mlp.ckpt'
    new_checkpoint_path='model/changed/mnist_mlp_chg.ckpt'
else:
    if args.data == "cifar10":
        ori_ckpt_path = 'model/cifar_cnn.ckpt'
        new_checkpoint_path='model/changed/cifar_cnn_chg.ckpt'


def change(change_rate):
    with tf.Session(graph=tf.Graph()) as sess:  #添加了graph=tf.Graph()可以加载多个模型
        '''
        new_var_list=[] #新建一个空列表存储更新后的Variable变量
        for var_name, _ in tf.train.list_variables(args.checkpoint_path): #得到checkpoint文件中所有的参数（名字，形状）元组
            var = tf.train.load_variable(args.checkpoint_path, var_name) #得到上述参数的值
            new_var = var*change #修改参数值（var）
            print('change %s' % var_name)
            changed_var = tf.Variable(new_var, name=var_name) #使用加入前缀的新名称重新构造了参数
            new_var_list.append(changed_var) #把赋予新名称的参数加入空列表
        
        print('starting to write new checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list) #构造一个保存器
        sess.run(tf.global_variables_initializer()) #初始化一下参数（这一步必做）
        model_name = 'mnist_mlp_chg' #构造一个保存的模型名称
        checkpoint_path = os.path.join(args.new_checkpoint_path, model_name) #构造一下保存路径
        saver.save(sess, checkpoint_path) #直接进行保存
        print("done !")
        '''
        saver = tf.train.import_meta_graph('{}.meta'.format(ori_ckpt_path))
        saver.restore(sess, ori_ckpt_path)

        if args.data=='mnist':
            w2 = tf.get_default_graph().get_tensor_by_name("w2:0")
            b2 = tf.get_default_graph().get_tensor_by_name("b2:0")
        else:
            w2 = tf.get_default_graph().get_tensor_by_name("conv2_w:0")
            b2 = tf.get_default_graph().get_tensor_by_name("conv2_b:0")

        #print("【Original W】", sess.run(w2))

        accuracy = tf.get_collection("accuracy")
        x = tf.get_default_graph().get_tensor_by_name("x:0")
        y_ = tf.get_default_graph().get_tensor_by_name("y_labels:0")  # 真实标签
        if args.data=='mnist':
            print("\n【Original testing accuracy:】", sess.run(accuracy, feed_dict={x: X_test,
                                                                              y_: y_test}))
        else:
            #is_training=tf.get_default_graph().get_tensor_by_name("is_training:0")
            print("\n【Original testing accuracy:】", sess.run(accuracy, feed_dict={x: X_test,y_: y_test}))

        array_w2 = sess.run(w2)
        array_b2 = sess.run(b2)
        new_w2 = array_w2 * change_rate
        new_b2 = array_b2 * change_rate
        w2 = tf.assign(w2, new_w2)
        b2 = tf.assign(b2, new_b2)
        sess.run(w2)
        sess.run(b2)
        saver.save(sess, new_checkpoint_path)  # 直接进行保存

        print('\n【Testing with new parameters.】')
        print("\n【New testing accuracy:】", sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))


def test():
    with tf.Session() as sess1:
        saver = tf.train.import_meta_graph('{}.meta'.format(new_checkpoint_path))
        saver.restore(sess1, new_checkpoint_path)

        if args.data=='mnist':
            w2 = tf.get_default_graph().get_tensor_by_name("w2:0")
        else:
            if args.data=='cifar10':
                w2=tf.get_default_graph().get_tensor_by_name("conv2_w:0")
        #print("【New W】", sess1.run(w2))

        accuracy = tf.get_collection("accuracy")
        x = tf.get_default_graph().get_tensor_by_name("x:0")
        y_ = tf.get_default_graph().get_tensor_by_name("y_labels:0")  # 真实标签
        if args.data=='mnist':
            print("\n【New testing accuracy:】", sess1.run(accuracy, feed_dict={x: X_test,
                                                                              y_: y_test}))
        else:
            #is_training=tf.get_default_graph().get_tensor_by_name("is_training:0")
            print("\n【New testing accuracy:】", sess1.run(accuracy, feed_dict={x: X_test,y_: y_test}))


for i in change_rate:
    print('\n【Change rate is 】#############', i)
    change(i)
    #test()
