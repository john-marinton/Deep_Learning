#import libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Implementing CNN

#intializing
classifier=Sequential()

#Step-1 Convolutional Layer
classifier.add(Convolution2D(32,(3,3),activation='relu',input_shape=(32,32,3)))

#step-2 MaxPooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding Additional covolutional layer
classifier.add(Convolution2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3 Flattenning
classifier.add(Flatten())


#step 4 Adding the  fully connected layer
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))

#output Layer
classifier.add(Dense(units=10,activation='softmax'))

#compilation
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=25)










