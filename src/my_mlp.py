import numpy as np
from src import utility as util
from keras import metrics
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

dataFileName = "data_updated.txt"
labelDict = {'transport': 0, 'walking': 1, 'airplane': 2}
num_classes = len(labelDict)

print('Loading data...')
with open(dataFileName) as f:
    content = f.readlines()

content = [x.strip() for x in content]

labels = [x.split(",")[-1] for x in content]
data = [','.join(x.split(",")[0:-1]) for x in content]


def tokenize(txt):
    return txt.split(",")

vec = CountVectorizer(tokenizer=tokenize)
transformed_data = vec.fit_transform(data).toarray()
normalized_data = normalize(transformed_data, norm='l1')
max_words = len(normalized_data[0])


def labelToArray(x):
    out_row = list([0]*3)
    out_row[labelDict.get(x)] = 1
    return out_row

transformed_labels = list(map(labelToArray, labels))

data_train, data_test, lbl_train, lbl_test = train_test_split(normalized_data, transformed_labels,
                                                              test_size=0.2, random_state=1, shuffle=True)

x_train = np.array(data_train)
x_test = np.array(data_test)
y_train = np.array(lbl_train)
y_test = np.array(lbl_test )

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(num_classes, 'classes')

print()
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print()
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(max_words,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=20,
                    verbose=0,
                    validation_split=0.2)

util.plotLossFigure(1, 'MLP Model 1', history.history['val_loss'], history.history['loss'])
score = model.evaluate(x_test, y_test,
                       batch_size=32, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


print('Building 2nd model...')
model2 = Sequential()
model2.add(Dense(128, input_shape=(max_words,)))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes))
model2.add(Activation('softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()

history = model2.fit(x_train, y_train,
                    batch_size=32,
                    epochs=20,
                    verbose=0,
                    validation_split=0.2)

util.plotLossFigure(2, 'MLP Model 2', history.history['val_loss'], history.history['loss'])
score = model2.evaluate(x_test, y_test,
                       batch_size=32, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Building 3rd model...')
model3 = Sequential()
model3.add(Dense(128, activation='relu', input_shape=(max_words,)))
model3.add(Dropout(0.5))
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(num_classes, activation='sigmoid'))
model3.summary()
model3.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['acc',metrics.categorical_accuracy])
history = model3.fit(x_train, y_train,
                    batch_size=32,
                    epochs=20,
                    verbose=0,
                    validation_split=0.2)

util.plotLossFigure(3, 'MLP Model 3', history.history['val_loss'], history.history['loss'])
score = model3.evaluate(x_test, y_test,
                       batch_size=32, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# Number of epochs were 10. Set it to 20, for better training.
# Performance of the 1st model was OK. Dropout level for 2nd and 3rd models were 0,2. They are set to 0,5.
# First layers of 2nd and 3rd models were Dense(512), learning too fast. They were dropped to Dense(128).