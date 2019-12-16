from keras.datasets import cifar10

(X_train , y_train) , (X_test , y_test) = cifar10.load_data()

print("Number of samples in training data = " , len(X_train))
print("Number of samples in testing data = " , len((X_test)))

# rescaling the images [0,255] -> [0,1]

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# one-hot encode the labels

from keras.utils import np_utils
import numpy as np

total_classes = len(np.unique(y_train))
y_train = np_utils.to_categorical(y_train , total_classes)
y_test = np_utils.to_categorical(y_test , total_classes)

# breaking the training set to training and validation set

(X_train , X_valid) = X_train[5000:] , X_train[:5000]
(y_train , y_valid) = y_train[5000:] , y_train[:5000]

print("X_train shape : " , X_train.shape)
print("X_test shape : " , X_test.shape)
print("X_valid shape : " , X_valid.shape)

# defining the model architecture

from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten

model = Sequential()

model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512 , activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100 , activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(total_classes , activation='softmax'))

model.summary()

# compile the model

model.compile(loss='categorical_crossentropy' , optimizer='rmsprop' , metrics=['accuracy'])

# accuracy score before training
score = model.fit(X_test , y_test , batch_size=49 , epochs=21 , verbose=2 , validation_data=(X_valid , y_valid) , shuffle=True)
accuracy = 100*score[1]

print("Classification accuracy score (before training) = " , accuracy)


#train the model
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='cifar10_mlp.hdf5' , verbose=1 , save_best_only=True)
hist = model.fit(X_train , y_train , batch_size=51 , epochs=17 , validation_data=(X_valid , y_valid) , callbacks=[checkpointer] , verbose=2 , shuffle=True)

# load the model
model.load_weights('cifar10_mlp.hdf5')

# evaluate the model

score = model.evaluate(X_test , y_test , verbose=0)
accuracy = 100*score[1]

print('Classification Accuracy Score (after training) = ' , accuracy)
