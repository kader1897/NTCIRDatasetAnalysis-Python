from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

import src.utility as util
import numpy as np
num_classes = 3
resnet_weights_path = "..\\models\\resnet50\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

print("Constructing ResNet50 Network...")
my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

print("Compiling the learning model...")
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# from tensorflow.python.keras.applications.resnet50 import preprocess_input
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224

print("Loading image data...")
# x, y = util.loadImgDataForCNN(img_rows=image_size, img_cols=image_size, label_exclude=["cycling", "running"])
# np.save("imgDataForResNet.npy",x)
# np.save("imgLabelsForResNet.npy",y)

x = np.load("imgDataForResNet.npy")
y = np.load("imgLabelsForResNet.npy")

print("Loading image data completed.")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print("Model fitting...")
hist = my_new_model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1, validation_split=0.2)
print(hist)
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
util.plotLossFigure(50, 'ResNet50 (50 epochs)', val_loss, train_loss)

# Evaluating the model
print("Evaluation on test data")
print("------------------------")
score = my_new_model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
#
# train_generator = data_generator.flow_from_directory(
#         '../input/urban-and-rural-photos/rural_and_urban_photos/train',
#         target_size=(image_size, image_size),
#         batch_size=24,
#         class_mode='categorical')
#
# validation_generator = data_generator.flow_from_directory(
#         '../input/urban-and-rural-photos/rural_and_urban_photos/val',
#         target_size=(image_size, image_size),
#         class_mode='categorical')
#
# my_new_model.fit_generator(
#         train_generator,
#         steps_per_epoch=3,
#         validation_data=validation_generator,
#         validation_steps=1)