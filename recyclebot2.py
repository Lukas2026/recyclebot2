#importing all of the libaries necessary
import PIL.ImageFile
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras import regularizers
from matplotlib import pyplot as plt
import numpy as np
import random, os
import time

import seaborn as sns

from sklearn.metrics import confusion_matrix

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

img_width, img_height = 512, 384
#All posible categories
categories = ["cardboard", "glass", "metal", "paper", "plastic"]
#This is the path to the dataset
train_data_dir = '/Users/lukasrois/ve/Train_Data'
test_data_dir = '/Users/lukasrois/ve/Test_Data'

classifier = Sequential()
# This is the learning rate of the model. It defines how fast the model learns.
opt = optimizers.Adam(lr=0.00005)



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)





#This is the neural network!
classifier.add(Conv2D(64,(3,3),input_shape = (64,64,3), activation= 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Conv2D(32,(3,3),input_shape = (32,32,3), activation= 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())
classifier.add(Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.001)))


classifier.add(Dense(5, activation='softmax'))
#The neural network needs to end in 5 possible options: cardboard, metal, glass, plastic, and paper

classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])





train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True
)

test_datagen = image.ImageDataGenerator(rescale=1./255)


#This makes a set for training the model
train_set = train_datagen.flow_from_directory(train_data_dir, target_size=(64,64),
                                              batch_size=16, class_mode='categorical', shuffle=True)
#This makes a set for testing the model
test_set = test_datagen.flow_from_directory(test_data_dir, target_size=(64,64),
                                              batch_size=16, class_mode='categorical', shuffle=True)


nb_train_samples = len(train_set)
nb_validation_samples = len(test_set)

train_labels = train_set.classes

#Training the model for 200 generations.
hist = classifier.fit_generator(train_set, steps_per_epoch=None, epochs=200,
                                validation_data=test_set, shuffle=True)

#Plotting the accuracy after training to analyze
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()

plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()

#Making a confusion matrix with the actual and predicted values
def config_confusion_matrix():
    actual_values = []
    predicted_values = []
    for i in range(50):
        c = categories.index(random.choice(categories))
        r = categories[c]
        path = "/Users/lukasrois/ve/Test_Data/"+r+"/"
        random_filename = random.choice([x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))])
        new_path = "/Users/lukasrois/ve/Test_Data/"+r+"/"+random_filename
        result = cast_predict(new_path)
        print(new_path, result)
        predicted_values.append(result)
        actual_values.append(c)

    return (actual_values, predicted_values)





#This is a function to predict 1 images, it needs the path to be passed into the functions
#and it returns the prediction of the neural network

def cast_predict(image_link):
    test_image = image.load_img(image_link, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    for i in range(len(result[0])):
        if result[0][i] >= .5:
            return i

#/Volumes/recyclebot2 is a shared folder between the raspberry pi and the laptop
file_name = '/Volumes/recyclebot2/sample.jpg'
text_file_name = '/Volumes/recyclebot2/result.txt'

#This code snipped checks if the camera on the raspberry pi has made a new picture
#If so it predicts what trash is on it and saves it in result.txt
time_stamp = 0
while True:
    time_stamp_2 = time.ctime(os.path.getctime(file_name))
    if time_stamp == time_stamp_2:
        continue
    else:
        with open(text_file_name, 'w') as my_file:
            print("new_file", time.strftime("%H:%M:%S"))
            time_stamp = time.ctime(os.path.getctime(file_name))
            time.sleep(0.5)
            try:
                pr = categories[cast_predict(file_name)]
            except OSError:
                pass
            print(pr)
            my_file.write(pr)



