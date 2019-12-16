#importing the dataset
# download the dataset of dog images here (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and place it in the respository

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')


# load ordered list of dog names
dog_names = [item[25:-1] for item in glob('dogImages/train/*/')]


# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % str(len(train_files) + len(valid_files) + len(test_files)))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


# obtain the VGG16 bottleneck feature.
#  download the file linked here(https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) and place it in the bottleneck_features/ folder

bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_vgg16 = bottleneck_features['train']
valid_vgg16 = bottleneck_features['valid']
test_vgg16 = bottleneck_features['test']


# define the model architecture

from keras.models import Sequential
from keras.layers import Dense , Flatten

model = Sequential()

model.add(Flatten(input_shape=(7,7,512)))
model.add(Dense(133 , activation='softmax'))

model.compile(loss='categorical_crossentropy' , optimizer='rmsprop' , metrics=['accuracy'])

model.summary()


# defin the model-2 with GlobalAveragePooling

from keras.layers import GlobalAveragePooling2D

model = Sequential()

model.add(GlobalAveragePooling2D(input_shape=(7,7,512)))
model.add(Dense(133 , activation='softmax'))

model.compile(loss='categorical_crossentropy' , optimizer='rmsprop' , metrics=['accuracy'])
model.summary()


# train the model

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='dogvgg16.weights.best.hdf5' , verbose=2 , save_best_only=True)

hist = model.fit(train_vgg16 , train_targets , batch_size=32 , epochs=11 , validation_data=(valid_vgg16 , valid_targets) , verbose=2 , shuffle=True)

# load the model with best-weights

model.load_weights('dogvgg16.weights.best.hdf5')

# calculate  the classification accuracy score

score = model.evaluate(test_vgg16 , test_targets)
accuracy = 100*score[0]

print("Accuracy Score = " , accuracy)