#import the dataset

from keras.datasets import cifar10

(X_train , y_train) , (X_test , y_test) = cifar10.load_data()

# rescale the images [0,255] -> [0,1]

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# one-hot encode the labels

from keras.utils import np_utils
import numpy as np

total_classes = len(np.unique(y_train))

y_train = np_utils.to_categorical(y_train , total_classes)
y_test = np_utils.to_categorical(y_test , total_classes)

# break the training data into training + validation data

(X_train , X_valid) = X_train[5000:] , X_train[:5000]
(y_train , y_valid) = y_train[5000:] , y_train[:5000]

print("X_train shape = " , X_train.shape)
print("X_test shape = " , X_test.shape)
print("X_valid shape = " , X_valid.shape)


# creating the model architecture

from keras.models import Sequential
from keras.layers import Conv2D , MaxPool2D , Flatten , Dense , Dropout

model = Sequential()

model.add(Conv2D(filters=16 , kernel_size=3 , strides=1 , padding='same' , activation='relu' , input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=2 , padding='same'))
model.add(Conv2D(filters=32 , kernel_size=3 , strides=1 , padding='same' , activation='relu' ))
model.add(MaxPool2D(pool_size=2 , padding='same'))
model.add(Conv2D(filters=64 , kernel_size=3 , strides=1 , padding='same' , activation='relu' ))
model.add(MaxPool2D(pool_size=2 , padding='same'))
model.add(Flatten())
model.add(Dense(512 , activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(total_classes , activation='softmax'))

model.summary()

# compile the model
model.compile(loss='categorical_crossentropy' , optimizer='rmsprop' , metrics=['accuracy'])

# checking the accuracy score before training
score = model.evaluate(X_test , y_test , verbose=0)
accuracy = 100*score[1]

print("Classification Accuracy Score (before training) = " , accuracy )

# train the model
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='cifar10_cnn.best.hdf5' , verbose=2 , save_best_only=True)
hist = model.fit(X_train , y_train , batch_size=32 , epochs=100 , verbose=2 , callbacks=[checkpoint] , validation_data=(X_valid , y_valid) , shuffle=True)

# load the model with best weights
model.load_weights('cifar10_cnn.best.hdf5')

# evaluate the model

score = model.evaluate(X_test , y_test , verbose=0)
accuracy = 100*score[1]

print("Classification Accuracy Score = " , accuracy)
