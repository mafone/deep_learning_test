"""Usage:
    CIFAR10.py <set_folder> destination_folder"""
#<preds.csv> and the corrected images will be outputted to destination_folder

import sys
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

from PIL import Image #pip install image
import glob
import cv2 as cv
import numpy #for arrays and complex objects management
#from numpy import*

#from scipy import misc
#from scipy.misc import imread
#from skimage.io import imread

def load_img (folder): 
        image_list = []
        for filename in glob.glob(folder + '/*.jpg'): #assuming jpg
            #im=Image.open(filename)
            #im = imread(filename)
            im = cv.imread (filename)
            image_list.append(im)
            #print ("****", im, "****")
        #Convert array
        return image_list

model = Sequential()

#image_list = load_img (sys.argv[1])

#model.add(Convolution2D(...)) # Output shape of convolution is 4d
#model.add(Flatten()) # Flatten input into 2d
#model.add(Dense(...)) # Dense layer require a 2d input

#Stacking layers (add)
#model.add(LSTM(3))
#model.add(LSTM(64, input_dim = 1))
#model.add(Dropout(0.2))
#model.add(LSTM(16))

#nb_of_features, len_of_sequence = len (image_list), len (image_list[0])
#model.add(LSTM(64, input_shape = (len_of_sequence, nb_of_features), return_sequences=True))
#model.add(LSTM(64, input_dim = nb_of_features, input_len = len_of_sequence, return_sequences=True))

model.add(Dense(units=3, activation='relu')) #model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=3, activation='softmax')) #model.add(Dense(units=10, activation='softmax'))

#once the model seems good configuring the learning process with .compile():
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#Changing configuration eventually
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

#iterating the training data in batches
image_list = load_img (sys.argv[1])
print ("Number of elements in the training set -> ", len (image_list))

x_train = numpy.array(image_list)
#x_train=asarray(Image.open('foto.jpg'))
#x=x_train.shape[0] #y=x_train.shape[1]*x_train.shape[2] #x_train.resize((x,y)) # a 2D array

y_train = numpy.array(image_list)
#y_train=asarray(Image.open('foto.jpg'))
#new_im = Image.fromarray (y_train)
#new_im.save("numpy_altered_sample.png")

#x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
#x=y_train.shape[0] #y=y_train.shape[1]*y_train.shape[2] #y_train.resize((x,y)) # a 2D array #numpy.resize(y_train,(x,y))

#print (y_train)

model.fit (x_train, y_train, epochs=1, batch_size=32) #32 is the default batch_size

#Alternatively, you we can feed batches to the model manually:
#model.train_on_batch(x_batch, y_batch)

#Evaluate the performance in one line:
#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

#Or generate predictions on new data:
#classes = model.predict(x_test, batch_size=128)


#Eval the y_train output and save in the output folder
#new_im = Image.fromarray (y_train)
#new_im.save("numpy_altered_sample.png")

