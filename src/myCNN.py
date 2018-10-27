# Import libraries
import numpy as np
from keras import backend as K
from sklearn.cross_validation import train_test_split

from src import DataPreprocessing as dp, utility as util

K.set_image_dim_ordering('tf')

from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D

img_rows = 128
img_cols = 128
num_channel = 3
num_classes = 3

IMAGE_PATH = "D:/kader/kad_ER_/METU/CENG (Grad)/Master\'s Thesis/NTCIR13-lifelog2-phase2-images/Images/"
dict = {"transport": 0, "walking": 1, "airplane": 2}  # Classes
# lbl_exclude = ['cycling', 'running']

# x, y = util.loadImgDataForCNN(img_rows, img_cols, num_channel, num_classes, dict, lbl_exclude)

x = np.load('imgData.npy')
y = np.load('imgLabels.npy')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print("Train - test split completed. Constructing the CNN...")

input_shape = x[0].shape
print(input_shape)

LOAD_MODEL = False
PROCESS_DATA = False
BATCH_NORM = False
BATCH_NORM_AFTER_ACT = True

if LOAD_MODEL:
    model = util.loadCNNModelFromJson('myCNNModel (25 Ep).json', 'myCNNWeights (25 Ep).h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
else:
    if BATCH_NORM:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(32, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Convolution2D(64, 3, 3))
        # model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

    elif BATCH_NORM_AFTER_ACT:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Convolution2D(64, 3, 3))
        # model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        model.add(BatchNormalization())

    else:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding="same", input_shape= input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, 3, 3))
        model.add(Activation('relu'))
        # model.add(Convolution2D(64, 3, 3))
        # model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    # %%
    # Training
    num_epoch = 100
    batch_size = 32
    validation_split = 0.2
    jsonFileName = 'myCNNModel (100 Ep - 32 BS - BatchNormAfterAct).json'
    weightsFileName = 'myCNNWeights (100 Ep - 32 BS - BatchNormAfterAct).h5'
    # hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))

    print("Model fitting...")

    if PROCESS_DATA:
        hist = dp.processData(X_train,y_train,model,num_epoch,batch_size,validation_split)
    else:
        hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_split=validation_split)
        util.saveCNNModelAsJson(model, jsonFileName, weightsFileName)

    print("Visualizing training results...")
    # visualizing losses and accuracy
    print(hist)
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    util.plotLossFigure(10, 'CNN Model with Batch Norm. After Act. (32 BS 100 Epochs)', val_loss, train_loss)

# Viewing model_configuration

#
# model.get_config()
# model.layers[0].get_config()
# model.layers[0].input_shape
# model.layers[0].output_shape
# model.layers[0].get_weights()
# np.shape(model.layers[0].get_weights()[0])
# model.layers[0].trainable
# %%

# Evaluating the model
print("Evaluation on test data")
print("------------------------")
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print(test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

print("Evaluation on new image")
print("------------------------")
# Testing a new image

# u1/2016/2016-08-27/20160827_105309_000.jpg [Airplane]
# u1/2016/2016-08-28/20160828_134813_000.jpg [Walking]
# u2/2016/2016-10-03/20161003_071527_000.jpg [Transport]

test_image = util.loadSingleImage('u2/2016/2016-10-03/20161003_071527_000.jpg', img_rows, img_cols, num_channel)
# Predicting the test image
print(model.predict(test_image))
print(model.predict_classes(test_image))
print('Test Image Class: Transport [' + str(dict['transport']) + ']')
print()

test_image = util.loadSingleImage('u1/2016/2016-08-28/20160828_134813_000.jpg', img_rows, img_cols, num_channel)
# Predicting the test image
print(model.predict(test_image))
print(model.predict_classes(test_image))
print('Test Image Class: Walking [' + str(dict['walking']) + ']')
print()

test_image = util.loadSingleImage('u1/2016/2016-08-27/20160827_105309_000.jpg', img_rows, img_cols, num_channel)
# Predicting the test image
print(model.predict(test_image))
print(model.predict_classes(test_image))
print('Test Image Class: Airplane [' + str(dict['airplane']) + ']')
print()


#
# # %%
#
# # Visualizing the intermediate layer
#
# #
# def get_featuremaps(model, layer_idx, X_batch):
# 	get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output, ])
# 	activations = get_activations([X_batch, 0])
# 	return activations
#
#
# layer_num = 3
# filter_num = 0
#
# activations = get_featuremaps(model, int(layer_num), test_image)
#
# print(np.shape(activations))
# feature_maps = activations[0][0]
# print(np.shape(feature_maps))
#
# if K.image_dim_ordering() == 'th':
# 	feature_maps = np.rollaxis((np.rollaxis(feature_maps, 2, 0)), 2, 0)
# print(feature_maps.shape)
#
# fig = plt.figure(figsize=(16, 16))
# plt.imshow(feature_maps[:, :, filter_num], cmap='gray')
# plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num) + '.jpg')
#
# num_of_featuremaps = feature_maps.shape[2]
# fig = plt.figure(figsize=(16, 16))
# plt.title("featuremaps-layer-{}".format(layer_num))
# subplot_num = int(np.ceil(np.sqrt(num_of_featuremaps)))
# for i in range(int(num_of_featuremaps)):
# 	ax = fig.add_subplot(subplot_num, subplot_num, i + 1)
# 	# ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
# 	ax.imshow(feature_maps[:, :, i], cmap='gray')
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.tight_layout()
# plt.show()
# fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')
#
# # %%
# # Printing the confusion matrix
# from sklearn.metrics import classification_report, confusion_matrix
# import itertools
#
# Y_pred = model.predict(X_test)
# print(Y_pred)
# y_pred = np.argmax(Y_pred, axis=1)
# print(y_pred)
# # y_pred = model.predict_classes(X_test)
# # print(y_pred)
# target_names = ['class 0(cats)', 'class 1(Dogs)', 'class 2(Horses)', 'class 3(Humans)']
#
# print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
#
# print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))
#
#
# # Plotting the confusion matrix
# def plot_confusion_matrix(cm, classes,
# 						  normalize=False,
# 						  title='Confusion matrix',
# 						  cmap=plt.cm.Blues):
# 	"""
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
# 	plt.imshow(cm, interpolation='nearest', cmap=cmap)
# 	plt.title(title)
# 	plt.colorbar()
# 	tick_marks = np.arange(len(classes))
# 	plt.xticks(tick_marks, classes, rotation=45)
# 	plt.yticks(tick_marks, classes)
#
# 	if normalize:
# 		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# 		print("Normalized confusion matrix")
# 	else:
# 		print('Confusion matrix, without normalization')
#
# 	print(cm)
#
# 	thresh = cm.max() / 2.
# 	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
# 		plt.text(j, i, cm[i, j],
# 				 horizontalalignment="center",
# 				 color="white" if cm[i, j] > thresh else "black")
#
# 	plt.tight_layout()
# 	plt.ylabel('True label')
# 	plt.xlabel('Predicted label')
#
#
# # Compute confusion matrix
# cnf_matrix = (confusion_matrix(np.argmax(y_test, axis=1), y_pred))
#
# np.set_printoptions(precision=2)
#
# plt.figure()
#
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(cnf_matrix, classes=target_names,
# 					  title='Confusion matrix')
# # plt.figure()
# # Plot normalized confusion matrix
# # plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
# #                      title='Normalized confusion matrix')
# # plt.figure()
# plt.show()
#

# Ceeee