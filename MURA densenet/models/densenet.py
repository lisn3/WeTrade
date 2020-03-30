from keras.applications.densenet import DenseNet169
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import tensorflow as tf
from math import pow
from sklearn.metrics import confusion_matrix

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

def calculate_w(train_label, test_label):
    # tai = total abnormal images, tni = total normal images
    #tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
    #tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
    data_cat=['train', 'valid']
    tai={}
    tni={}
    tai['train']=np.sum(np.argmax(train_label,axis=1)==1)
    tai['valid']=np.sum(np.argmax(test_label,axis=1)==1)
    tni['train']=len(train_label)-tai['train']
    tni['valid']=len(test_label)-tai['valid']
    global Wt1, Wt0
    Wt1 = {x: tni[x] / (tni[x] + tai[x]) for x in data_cat}
    Wt0 = {x: tai[x] / (tni[x] + tai[x]) for x in data_cat}

    print('total abnormal images:', tai)
    print('total normal images:', tni, '\n')
    print('Wt0 train:', Wt0['train'])
    #print('Wt0 valid:', Wt0['valid'])
    print('Wt1 train:', Wt1['train'])
    #print('Wt1 valid:', Wt1['valid'])
    return Wt1, Wt0


def myloss_train(y_true, y_pred):
    phase='train'
    loss = - (Wt1[phase] * y_true * tf.log(y_pred+pow(10.0, -9)) + Wt0[phase] * (1 - y_true) * tf.log(1 - y_pred+pow(10.0,-9)))
    return loss


def build_transfer_model(pre_trained, fixed_layer_num, X_train_data, y_train_data, X_test, y_test,
                         fig_name, model_name, learning_rate, epoch, batch_size, save):
    if pre_trained:
        base_model=DenseNet169(weights='imagenet', include_top=False, pooling=None,
                               input_shape=(224,224,3))
        for layers in base_model.layers[0:fixed_layer_num]:
            layers.trainable = False
    else:
        base_model = DenseNet169(weights=None, include_top=False, pooling=None,
                                 input_shape=(224,224,3))

    model = Flatten()(base_model.output)
    model = Dense(512, activation='relu', name='fc1')(model)
    model = Dropout(0.5)(model)
    model = Dense(2, activation='sigmoid', name='prediction')(model)
    model_densenet_pretrain = Model(inputs=base_model.input, outputs=model, name='Densenet_pretrain')
    # print(model_incept_cifar10_pretrain.summary())
   # for i, layer in enumerate(base_model.layers):
    #    print(i, layer.name)

    calculate_w(y_train_data, y_test)

    time1 = time.time()
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # 0.0001  fix4 #'categorical_crossentropy', #myloss_train,
    model_densenet_pretrain.compile(optimizer=adam, loss=myloss_train, metrics=['accuracy'])

    print("【Start Training】#####")
    datagen=data_gene(X_train_data)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    history = model_densenet_pretrain.fit_generator(datagen.flow(X_train_data, y_train_data, batch_size=batch_size),
                                                validation_data=(X_test, y_test),
                                                validation_steps=len(X_test)// batch_size,
                                                epochs=epoch,
                                                steps_per_epoch=len(X_train_data)//batch_size,
                                               # callbacks=[reduce_lr],
                                                verbose=1,
                                                shuffle=True)

    if save:
        model_densenet_pretrain.save('models/{}.h5'.format(model_name))
    plot_training(history, fig_name)

    print("【Testing】######")
    scores = model_densenet_pretrain.evaluate(X_test, y_test, verbose=0)
    print("testing score", scores)
    y_pred=model_densenet_pretrain.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    print('Confusion matrix result:', confusion_matrix(y_true, y_pred))




