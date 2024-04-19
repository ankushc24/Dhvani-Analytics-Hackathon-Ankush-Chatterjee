import numpy as np 
import pandas as pd  
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

########################################################
for dirname, _, filenames in os.walk('dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#ignores if any warning arises to continue running of code
warnings.filterwarnings("ignore")

##########################################################
data_path = 'dataset/'

labels=[]
for folder in os.listdir(data_path):
    labels.append(folder)


#########################################################
train_images=[]
train_labels=[]

for i,folder in enumerate(labels):
    try:
        for image in os.listdir(data_path+'/'+folder):
            img = os.path.join(data_path+'/'+folder+'/'+image)
            img = cv2.imread(img)
            img = cv2.resize(img,(256,256))
            train_images.append(img)
            train_labels.append(i)
    except:
        print(i,folder,image,img)
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels).astype('int64')

#########################################################

for i in [0,1]:
    plt.imshow(train_images[i])
    plt.title(labels[train_labels[i]])
    plt.show()

########################################################
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels, 3)
print(f'After preprocessing, our dataset has {train_images.shape[0]} images with shape {train_images.shape[1:]}')
print(f'After preprocessing, our dataset has {train_labels.shape[0]} rows with {train_labels.shape[1]} labels')

#########################################################
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_images,train_labels,test_size=0.1,shuffle=True)

# Split data into training and validation sets(used for validation while training)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

########################################################
print(f'After spiltting, shape of our train dataset: {X_train.shape}')
print(f'After spiltting, shape of our test dataset: {X_test.shape}')

##########################################################
import tensorflow.keras.backend  as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense,\
            Dropout,Rescaling,Dense,Flatten,Activation,BatchNormalization
from tensorflow.keras.models import load_model




###########################################################################################
# Create an ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,  # Rotate images randomly up to 20 degrees
    width_shift_range=0.2,  # Shift images horizontally by up to 20% of the width
    height_shift_range=0.2,  # Shift images vertically by up to 20% of the height
    shear_range=0.2,  # Shear angle in radians
    zoom_range=0.2,  # Zoom range [1-zoom_range, 1+zoom_range]
    horizontal_flip=True,  # Randomly flip images horizontally
    rescale=1.0 / 255  # Rescale pixel values to [0, 1]
)

# Create augmented training data generator
train_generator = train_datagen.flow(X_train, y_train, batch_size=1)


##########################################################
K.clear_session()
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(256, 256, 3)),

    #Data augmentation
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    Rescaling(1.0 / 255),

    #Convolution layers
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    Flatten(),
    #Fully connected layers
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    #Output layer
    Dense(3, activation='softmax')

])
model.summary()
##############################################################
from tensorflow.keras.callbacks import EarlyStopping

# Define the EarlyStopping callback with a patience of 5 epochs

early_stopping = EarlyStopping(
    min_delta=0.005, #minimum change in the monitored metric to be considered as an improvement.
    patience=5, # number of epochs with no improvement after which training will be stopped.
    restore_best_weights=True #weights of model at epoch with best performance will be restored before training stop
)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
            )

history_model = model.fit(train_generator,
epochs=15, steps_per_epoch=len(X_train) // 1,  # Since batch_size=1, steps_per_epoch=len(X_train)
validation_data=(X_val, y_val))

##############################################################

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

###############################################################



# Predict on the test set
y_pred = model.predict(X_test)
y_pred_categorical = np.argmax(y_pred, axis=1)


#################################################################
test_img = cv2.imread('dataset/flashes/defect1.png')
test_img = cv2.resize(test_img,(256,256))
test_img1 = np.asarray(test_img)
test_img = test_img1.reshape(-1,256,256,3)
p = model.predict(test_img)

plt.imshow(test_img1)
plt.title(labels[np.argmax(p)])
np.argmax(p)
plt.show()