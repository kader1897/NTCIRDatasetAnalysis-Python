from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def processData(X_train,y_train,model, num_epoch, batch_size, validation_split):
    x_train,x_val,y_train,y_val = train_test_split(X_train, y_train, test_size=validation_split, random_state=1)
    # Image generator
    trainDataGenerator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True)
    trainDataGenerator.fit(x_train)

    valDataGenerator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True)
    valDataGenerator.fit(x_val)

    return model.fit_generator(trainDataGenerator.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size, epochs=num_epoch, verbose=1,
                    validation_data=valDataGenerator.flow(x_val,y_val,batch_size=batch_size),
                    validation_steps=len(x_val) / batch_size)




