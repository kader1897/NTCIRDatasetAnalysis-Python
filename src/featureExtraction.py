import cv2
import os

import numpy as np

from keras import backend as K
from keras.layers import BatchNormalization
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential


PATH = os.getcwd()
resnet_weights_path = "..\\models\\resnet50\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

file = open(PATH + "\\..\\data\\u2AllImages.txt", 'r')

data_list = []
i = 0
img_rows = 224
img_cols = 224
num_channel = 3


for line in file:
    try:
        input_img = cv2.imread(line.strip())
        input_img_rsz = cv2.resize(input_img, (img_rows, img_cols))
        data_list.append(input_img_rsz)

        i += 1
        if i % 1000 == 0:
            print(str(i) + ". Image loaded: " + line)

    except:
        print("Exception on " + str(i) + ". Image: " + line)
        i += 1

img_data = np.array(data_list)
img_data = img_data.astype('float32')
img_data = img_data / np.amax(img_data)
print(img_data.shape)

# Reshape data according to Keras dimension ordering
if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        img_data = np.expand_dims(img_data, axis=1)
        print(img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis=4)
        print(img_data.shape)
else:
    if K.image_dim_ordering() == 'th':
        img_data = np.rollaxis(img_data, 3, 1)
        print(img_data.shape)

print("Saving numpy array...")
np.save("..\\arrays\\u2AllImages.npy",img_data)

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
# my_new_model.add(BatchNormalization())


# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

print("Compiling the learning model...")
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

preds = my_new_model.predict(img_data, verbose=1)
np.save("..\\arrays\\u2ImageFeaturesResnet50.npy",preds)
print("Saving numpy array completed...")

np.savetxt("..\\data\\u2ImageFeaturesResnet50.txt",preds,fmt='%.5f')
print("Saving text file completed...")

