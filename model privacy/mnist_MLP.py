import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from dataloader import load_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = config)

batch_size=64
input_size = 784
hidden1 = 128
hidden2 = 64
hidden3=32
out_size = 10
X_train, X_test, y_train, y_test=load_data('mnist')

'''
def make_one_hot(data1):
    rlt=np.zeros([data1.shape[0],10])
    for i in range(data1.shape[0]):
        col=data1[i]
        rlt[i][col]=1
    return rlt
'''

x = tf.placeholder(tf.float32,shape = [None,784],name='x')
y_= tf.placeholder(tf.float32,shape=[None,10],name='y_labels')

w1=tf.Variable(tf.truncated_normal([input_size,hidden1], stddev = 0.1),name='w1')
b1=tf.Variable(tf.constant(0.01, shape = [hidden1]),name='b1')
layer1=tf.nn.relu(tf.matmul(x,w1)+b1,name='layer1')

w2=tf.Variable(tf.truncated_normal([hidden1,hidden2], stddev = 0.1),name='w2')
b2=tf.Variable(tf.constant(0.01, shape = [hidden2]),name='b2')
layer2=tf.nn.relu(tf.matmul(layer1,w2)+b2,name='layer2')

w3=tf.Variable(tf.truncated_normal([hidden2,hidden3], stddev = 0.1),name='w3')
b3=tf.Variable(tf.constant(0.01, shape = [hidden3]),name='b3')
layer3=tf.nn.relu(tf.matmul(layer2,w3)+b3,name='layer3')

w4=tf.Variable(tf.truncated_normal([hidden3,out_size], stddev = 0.1),name='w4')
b4=tf.Variable(tf.constant(0.01, shape = [out_size]),name='b4')
logits=tf.add(tf.matmul(layer3,w4),b4,name='layer4')
pred_y=tf.nn.softmax(logits)
pred_labels=tf.argmax(pred_y,1)

xent = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits)
loss = tf.reduce_mean(xent, name='loss')
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(pred_y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

tf.add_to_collection("accuracy",accuracy)

cur_pos=0
for i in range(5000):
    if cur_pos+batch_size<X_train.shape[0]:
        start=cur_pos
        end=cur_pos+batch_size
        cur_pos = end
    else:
        start=cur_pos
        end=X_train.shape[0]-1
        cur_pos=0
    batch_xs=X_train[start:end]
    batch_ys=y_train[start:end]

    sess.run(train_step,feed_dict={x:batch_xs, y_:batch_ys})
    if(i%100==0):
        train_accuracy = accuracy.eval(session=sess,
                                       feed_dict={x:batch_xs , y_: batch_ys})
        print("step %d, train_accuracy %g" % (i, train_accuracy))

print("testing accuracy:",sess.run(accuracy,feed_dict={x: X_test,
                                                       y_: y_test}))


save_path = saver.save(sess, "model/mnist_mlp.ckpt")
print ("Model saved in file: ", save_path)




