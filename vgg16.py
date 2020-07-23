# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, ZeroPadding2D
from keras.layers.core import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import datagenerator as datagenerator
import numpy as np
import math
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils import class_weight
import os
import cv2

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def vgg16(num_cls, num_channels):

    inp = Input(shape=(224, 224, num_channels), name='data')
    # block 1
    x = ZeroPadding2D((1, 1))(inp)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # block 2
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # block 3
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # block 4
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # block 5
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Add Fully Connected Layer
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_fc = Dense(1000, activation='softmax')(x)

    model = Model(inp, x_fc, name='vgg_standard')

    # Load weights into the new model
    model.load_weights('vgg16_weights.h5', by_name=True)
    x_new_fc = Dense(num_cls, activation='softmax')(x)
    model = Model(inp, x_new_fc, name='vgg_standard')

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    filepath = "weights-improvement_vgg_extrarun-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    weights_save_epoch = checkpoint

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    callbacks_list = [weights_save_epoch, reduce_lr]

    return model, callbacks_list

if __name__ == '__main__':

    batch_size = 8
    img_rows, img_cols = 224, 224 # Resolution of inputs
    num_channels = 10
    num_classes = 20
    nb_epoch = 50

    train_file = '/home/gpu/projects/VGG/create_txt/train_file_wref_nocls0_norect.txt'
    val_file = '/home/gpu/projects/VGG/create_txt/validation_file_wref_nocls0_norect.txt'

    training_generator = datagenerator.DataGenerator(train_file, train=True, shuffle=True, horizontal_flip=True)
    sorted_labels = sorted(training_generator.labels)
    class_weight = class_weight.compute_class_weight('balanced', np.unique(sorted_labels), sorted_labels)
    validation_generator = datagenerator.DataGenerator(val_file, train=False, shuffle=False)

    # Load our model
    [model, callbacks_list] = vgg16(num_classes, num_channels)

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=nb_epoch,
                        shuffle=True,
                        use_multiprocessing=True,
                        max_queue_size=10,
                        workers=5,
                        verbose=1,
                        class_weight=class_weight,
                        callbacks=callbacks_list)

    # Save the model architecture
    with open('vgg16_model_noattention_extrarun.json', 'w') as f:
        f.write(model.to_json())

    # Save the weights
    model.save_weights('vgg16_weights_noattention_extrarun.h5')

    # Make predictions
    predictions_valid = model.predict_generator(generator=validation_generator, use_multiprocessing=True, workers=5, verbose=1)

    val_trues = np.array(validation_generator.ind_labels)
    val_trues = val_trues[::num_channels]
    val_trues_cut = val_trues[:-(len(val_trues) % batch_size)]
    val_pred = np.argmax(predictions_valid, axis=1)
    # confusion matrix
    print('Confusion Matrix: ')
    cm = confusion_matrix(val_trues_cut, val_pred)
    np.set_printoptions(threshold=10000)
    #print(cm)
    # metrics calc
    precisions, recall, f1_score, _ = precision_recall_fscore_support(val_trues_cut, val_pred)
    print('Average precision is: ')
    print(np.ndarray.mean(precisions))
    print('Average recall is: ')
    print(np.ndarray.mean(recall))
    print('Average f1_score is: ')
    print(np.ndarray.mean(f1_score))