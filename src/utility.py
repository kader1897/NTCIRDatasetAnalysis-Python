import matplotlib.pyplot as plt
import numpy as np
import os, cv2

from keras import backend as K
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import load_model
import h5py
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

K.set_image_dim_ordering('tf')

IMAGE_PATH = "D:/kader/kad_ER_/METU/CENG (Grad)/Master\'s Thesis/NTCIR13-lifelog2-phase2-images/Images/"


def plotLossFigure(figId, figTitle, arrValLoss, arrTrainLoss):
    plt.figure(figId)
    plt.plot(np.arange(1, len(arrValLoss) + 1).tolist(), arrValLoss, 'r', label='Val. Loss')
    plt.plot(np.arange(1, len(arrTrainLoss) + 1).tolist(), arrTrainLoss, 'b', label='Training Loss')
    plt.title('Epochs vs. Loss (' + figTitle + ')')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right', shadow=True)
    plt.show()


def loadImgDataForCNN(img_rows=128, img_cols=128, num_channel=3, num_classes=3,
                      dict={"transport": 0, "walking": 1, "airplane": 2}, label_exclude=[],
                      useSklearnPreprocessing=False, fileName="data\\img_wl_and_activity.txt", pathIndex=0,
                      labelIndex=1):
    PATH = os.getcwd()
    file = open(PATH + "\\..\\" + fileName, 'r')

    data_list = []
    label_list = []

    i = 0
    for line in file:
        tokens = line.strip().split(',')
        img_path = tokens[pathIndex]
        label = tokens[labelIndex]
        if (label in label_exclude):
            continue

        input_img = cv2.imread(IMAGE_PATH + img_path)
        # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_rsz = cv2.resize(input_img, (img_rows, img_cols))
        data_list.append(input_img_rsz)
        label_list.append(dict.get(label))

        i += 1
        if i % 1000 == 0:
            print(str(i) + ". Image loaded: " + img_path)

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

    if useSklearnPreprocessing:
        # using sklearn for preprocessing
        from sklearn import preprocessing

        def image_to_feature_vector(image, size=(img_rows, img_cols)):
            # resize the image to a fixed size, then flatten the image into
            # a list of raw pixel intensities
            return cv2.resize(image, size).flatten()

        file = open(PATH + "/" + fileName, 'r')

        img_data_list = []
        img_label_list = []
        for line in file:
            tokens = line.strip().split(',')
            img_path = tokens[0]
            label = tokens[1]
            input_img = cv2.imread(IMAGE_PATH + img_path)
            # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_flatten = image_to_feature_vector(input_img, (img_rows, img_cols))
            img_data_list.append(input_img_flatten)
            img_label_list.append(label)

        img_data = np.array(img_data_list)
        img_data = img_data.astype('float32')
        print(img_data.shape)
        img_data_scaled = preprocessing.scale(img_data)
        print(img_data_scaled.shape)

        if K.image_dim_ordering() == 'th':
            img_data_scaled = img_data_scaled.reshape(img_data.shape[0], num_channel, img_rows, img_cols)
            print(img_data_scaled.shape)

        else:
            img_data_scaled = img_data_scaled.reshape(img_data.shape[0], img_rows, img_cols, num_channel)
            print(img_data_scaled.shape)

        if K.image_dim_ordering() == 'th':
            img_data_scaled = img_data_scaled.reshape(img_data.shape[0], num_channel, img_rows, img_cols)
            print(img_data_scaled.shape)

        else:
            img_data_scaled = img_data_scaled.reshape(img_data.shape[0], img_rows, img_cols, num_channel)
            print(img_data_scaled.shape)

    if useSklearnPreprocessing:
        img_data = img_data_scaled

    Y = np_utils.to_categorical(label_list, num_classes)

    return img_data, Y


def loadSingleImage(filePath, img_rows, img_cols, num_channel):
    test_image = cv2.imread(IMAGE_PATH + filePath)
    # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, (img_rows, img_cols))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= np.amax(test_image)

    if num_channel == 1:
        if K.image_dim_ordering() == 'th':
            test_image = np.expand_dims(test_image, axis=0)
            test_image = np.expand_dims(test_image, axis=0)
        else:
            test_image = np.expand_dims(test_image, axis=3)
            test_image = np.expand_dims(test_image, axis=0)
    else:
        if K.image_dim_ordering() == 'th':
            test_image = np.rollaxis(test_image, 2, 0)
            test_image = np.expand_dims(test_image, axis=0)
        else:
            test_image = np.expand_dims(test_image, axis=0)

    print('Test Image Shape: ' + str(test_image.shape))
    return test_image


# serialize model to JSON
def saveCNNModelAsJson(model, fileName, weightsFileName):
    model_json = model.to_json()
    with open(fileName, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weightsFileName)
    print("Saved model to disk")


def loadCNNModelFromJson(fileName, weightsFileName):
    # load json and create model
    json_file = open(fileName, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightsFileName)
    print("Loaded model from disk")
    return loaded_model

    # model.save('model.hdf5')
    # loaded_model = load_model('model.hdf5')


def wordToVec(data):
    vec = CountVectorizer(tokenizer=tokenize)
    return vec.fit_transform(data).toarray()


def tokenize(txt):
    return txt.split(",")


def clusterAssignment(n_clusters, kmeans_result, activityList, activityCounter):
    resultList = [None] * activityList.__len__()
    for i in range(n_clusters):
        tempList = np.where(kmeans_result.labels_ == i)[0]
        activities = Counter([activityList[ind] for ind in tempList])
        for act in activities:
            if act is not None:
                activities[act] /= activityCounter[act]
            else:
                activities[act] = 0
        activity = activities.most_common()[0]
        for index in tempList:
            resultList[index] = activity[0]
    return resultList


def clusterAssignmentIgnoreFreq(n_clusters, kmeans_result, activityList):
    resultList = [None] * activityList.__len__()
    for i in range(n_clusters):
        tempList = np.where(kmeans_result.labels_ == i)[0]
        activities = Counter([activityList[ind] for ind in tempList])
        activity = activities.most_common()[0]
        for index in tempList:
            resultList[index] = activity[0]
    return resultList