import numpy as np
import cv2
import random
import keras
import glob

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""


class DataGenerator(keras.utils.Sequence):

    def read_class_list(self, class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(" ".join(items[:-1]))
                self.labels.append(int(items[-1]))
            # store total number of data
            self.data_size = len(self.labels)

    def __init__(self, class_list, train, shuffle, horizontal_flip=False,
                 mean=np.array(82.06), scale_size=(224, 224),
                batch_size=8, n_channels=10):

        self.read_class_list(class_list)
        list_IDs = self.images
        labels = self.labels
        [uniq_lab_all, uniq_lab_cat_all] = np.unique(labels, return_inverse=True)
        self.ind_labels = uniq_lab_cat_all

        # Init params
        self.horizontal_flip = horizontal_flip
        self.train = train
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.labels = labels
        self.list_IDs = list_IDs
        self.on_epoch_end()

        # shuffle
        if self.shuffle:
            #create sublists, in order to shuffle in blocks of n_channels
            self.images = [self.images[i:i + n_channels] for i in range(0, len(self.images), n_channels)]
            self.labels = [self.labels[i:i + n_channels] for i in range(0, len(self.labels), n_channels)]
            #combine images and labels, in order to shuffle them together
            combined = list(zip(self.images, self.labels))
            random.shuffle(combined)
            self.images[:], self.labels[:] = zip(*combined)
            #return to regular list (not sublists)
            self.images = [item for sublist in self.images for item in sublist]
            self.labels = [item for sublist in self.labels for item in sublist]
        # create dictionary
        self.dict_img = dict(zip(self.images, self.labels))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / (self.batch_size*self.n_channels)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * (self.batch_size*self.n_channels):(index + 1) * (self.batch_size*self.n_channels)]

        # Find list of IDs
        '''
        dict_img_list = []
        for key, value in self.dict_img.items():
            temp = [key, value]
            dict_img_list.append(temp)
        list_IDs_temp = [dict_img_list[k] for k in indexes]
        '''
        dict_img_list = []
        for i in range(len(self.images)):
            temp = [self.images[i], self.labels[i]]
            dict_img_list.append(temp)
        list_IDs_temp = [dict_img_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        #if self.shuffle == True:
        #    np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __data_generation(self, list_IDs_temp):

        # Get images (path) and labels
        paths = self.images
        labels = self.labels
        [uniq_lab, _] = np.unique(labels, return_inverse=True)
        dict_labels = dict(zip(range(len(uniq_lab)), uniq_lab))
        n_cls = len(uniq_lab)

        # Read images
        X = np.ndarray([int(self.batch_size), self.scale_size[0], self.scale_size[1], self.n_channels])
        Y = np.ndarray(int(self.batch_size), dtype=int)
        for i, id in enumerate(list_IDs_temp):
            #img = cv2.imread(id, 0) # read grayscale
            img = cv2.imread(id[0], 0)

            # rescale image
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img = img.astype(np.float32)
            # subtract mean
            img -= self.mean

            [a, b] = divmod(i, self.n_channels)
            X[a,:,:,b] = img
            if i % self.n_channels == 0:
                Y[int(i/self.n_channels)] = self.dict_img[id[0]]

        # flip image at random if flag is selected
        for i in range(self.batch_size):
            if self.horizontal_flip and np.random.random() < 0.5:
                X[i, :, :, :] = cv2.flip(X[i,:,:,:], 1)

        '''
        img = X[0, :, :, 0]
        import matplotlib.pyplot as plt
        imgplot = plt.imshow(img)
        plt.show()
        '''

        #one hot encoding
        Y_ = []
        for i in range(len(Y)):
            Y_.append([k for (k, v) in dict_labels.items() if v == Y[i]])

        one_hot_labels = keras.utils.to_categorical(Y_, num_classes=n_cls)
        '''
        ### test individual images
        import matplotlib.pyplot as plt
        for i in range(15):
            img_test = images[i, :, :, :]
            img_test_norm = (img_test - img_test.min()) / (img_test.max() - img_test.min())
            imgplot = plt.imshow(img_test_norm)
            plt.show()
        '''

        '''
        # Expand labels to one hot encoding
        Y = np.zeros((self.batch_size, self.n_classes))
        Y = keras.utils.to_categorical(labels, num_classes=self.n_classes)
        for i in range(len(labels)):
            Y[i][labels[i]-1] = 1
        '''

        # return array of images and labels
        return X, one_hot_labels