###try
import numpy as np

#a = np.random.rand(2,2,3,1)
#a.mean(axis=(0,1,2))


import numpy as np
import cv2

train_file = '/home/gpu/projects/VGG/create_txt/train_file_wref_nocls0.txt'
val_file = '/home/gpu/projects/VGG/create_txt/validation_file_wref_nocls0.txt'

with open(train_file) as f:
    lines = f.readlines()
    images = []
    labels = []
    for l in lines:
        items = l.split()
        images.append(" ".join(items[:-1]))
        labels.append(int(items[-1]))

train = []
for i in (images):
    img = cv2.imread(i)
    img = cv2.resize(img, (224, 224))
    train.append(img)

train = np.asarray(train)
print(train.mean(axis=(0,1,2)))