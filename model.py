#Behavioral Cloning Project
#Jose Carlos Marti Monter - April’17

import csv
import cv2
import numpy as np

#Open the CSV file to get the path a filenames for the saved images
lines = []
with open('C:/Users/Blanca/Desktop/DATA/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
correction = 0.2 #To add os subst to the side camera images
for line in lines:
	for i in range(3):
		source_path = line[i]
		image = cv2.imread(source_path)
		images.append(image)
		measurement = float(line[3])
		if i==0:
			measurements.append(measurement)
		if i==1:
			measurements.append(measurement+correction)
		if i==2:
			measurements.append(measurement-correction)
			
#Flipped images are added to the dataset	
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
		augmented_images.append(image)
		augmented_measurements.append(measurement)
		augmented_images.append(cv2.flip(image,1))
		augmented_measurements.append(measurement*-1.0)
	

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#Description of the Neural Network
model = Sequential()
model.add(Cropping2D(cropping=((60,30), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))

model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))

model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))

model.add(Dense(1))

#Load previous weights as starting point for the NN after additional data was added to the dataset
model = load_model('model.h5')
model.summary()


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)
#Save model
model.save('model.h5')
exit()

