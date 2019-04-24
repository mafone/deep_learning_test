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
import csv

def load_csv (fn): 
    res = {}
    for row in csv.DictReader(open(fn)):
            res[row['fn']] = row['label']
    return res


def get_tag (orientation):
    if orientation == 'upright':
        return [0, 0, 0]
    elif orientation == 'rotated_left':
        return [0, 1, 0]
    elif orientation == 'rotated_right':
        return [0, 0, 1]
    else: #upside_down
        return [0, 1, 1]


def load_train_img (folder): 
    image_list = []
    truth = {}
    #get the tags
    for filename in glob.glob(folder + '/*.csv'): #assuming csv
        truth = load_csv (filename)

    for filename in glob.glob(folder + '/*.jpg'): #assuming jpg
        im = cv.imread (filename)
        #filename = filename.split ("/")[-1] #Just the filename withput the full directory
        #tag = get_tag (truth [filename])
        #im = numpy.insert (im, 0, tag, 0)
        image_list.append(im)
    return image_list


def load_test_img (folder): 
    image_list = []

    for filename in glob.glob(folder + '/*.jpg'): #assuming jpg
        im = cv.imread (filename)
        filename = filename.split ("/")[-1] #Just the filename withput the full directory
        image_list.append(im)
    return image_list


model = Sequential()

model.add(Dense(units=3, activation='relu')) #model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=3, activation='softmax')) #model.add(Dense(units=10, activation='softmax'))
model.add(Dense(units=3, activation='relu')) #model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=3, activation='softmax')) #model.add(Dense(units=10, activation='softmax'))


#once the model seems good, configure the learning process with .compile():
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#Changing configuration (eventually)
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

print ("\n************TRAINING************\n")
#iterating the training data in batches
image_list1 = load_train_img (sys.argv[1])
#print ("Number of elements in the training set -> ", len (image_list))

x_train = numpy.array(image_list1) #x=x_train.shape[0] #y=x_train.shape[1]*x_train.shape[2] #x_train.resize((x,y)) # a 2D array
y_train = numpy.array(image_list1)


#x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
#x=y_train.shape[0] #y=y_train.shape[1]*y_train.shape[2] #y_train.resize((x,y)) # a 2D array #numpy.resize(y_train,(x,y))

#print (y_train)

#The batch size is a hyperparameter of gradient descent that controls the number of training samples to work through before the model’s internal parameters are updated.
#The number of epochs is a hyperparameter of gradient descent that controls the number of complete passes through the training dataset.
model.fit (x_train, y_train, epochs=1, batch_size=32) #32 is the default batch_size


#Alternatively, we can feed batches to the model manually:
#x_batch = 32
#y_batch = 8
#model.train_on_batch(x_batch, y_batch)

print ("\n************TESTING************\n")
image_list2 = load_test_img (sys.argv[2])
x_test = numpy.array(image_list2)
y_test = numpy.array(image_list2)

#Evaluate the performance in one line:
loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
print (loss_and_metrics)

#Or generate predictions on new data:
#classes = model.predict(x_test, batch_size=128)
#print (classes)


#Eval the y_train output and save in the output folder
#new_im = Image.fromarray (y_train)
#new_im.save("numpy_altered_sample.png")

