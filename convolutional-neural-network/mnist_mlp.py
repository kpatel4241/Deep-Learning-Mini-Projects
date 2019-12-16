from keras.datasets import mnist

#loading the MNIST dataset
(X_train , y_train) , (X_test , y_test) = mnist.load_data()

print("\n Number of samples in Training data = " , len(X_train))
print("\n Number of samples in Testing data = " , len(X_test))

# rescale the images
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# on-hot encoding the labels

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train , 10)
y_test = np_utils.to_categorical(y_test , 10)

# defining the model architecture

from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten

model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512 , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512 , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10 , activation='softmax'))

model.summary()

# compiling the model
model.compile(loss='categorical_crossentropy' , optimizer='rmsprop' , metrics=['accuracy'])

# calculate the classification accuracy on testset before training
score = model.evaluate(X_test , y_test , verbose=0)
print(score , type(score))
accuracy = 100*score[1]

print("Accuracy of test-set before training = " , accuracy)


# train the model
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='mnist.best.hdf5' , monitor='val_loss' , verbose=1 , save_best_only=True)

hist = model.fit(X_train , y_train , batch_size=200 , epochs=10 , validation_split=0.2 , callbacks=[checkpoint] , verbose=0 , shuffle=True)

# load the weights
model.load_weights('mnist.best.hdf5')

# calculate the classification accuracy after training
score = model.evaluate(X_test , y_test , verbose=0)
accuracy = 100*score[1]

print("Classificatio accuracy score after training = ",accuracy)

