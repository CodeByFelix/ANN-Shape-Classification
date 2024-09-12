# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:08:50 2024

@author: Felix
"""

import pandas as pd
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, History
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


#Image reading
imageName = {"circle": [], "triangle": [], "square": []}
basePath = {
        "circle": r"Datasets\shapes\circles",
        "triangle": r"Datasets\shapes\triangles",
        "square": r"Datasets\shapes\squares"
}

#os.listdir(path)
for key in basePath.keys():
    for img in os.listdir(basePath[key]):
        imageName[key].append(img)
    
        
#Image Data Generation and Preprocessing
data = {"label": [], "imageData": []}

for key in imageName.keys():
    for n in imageName[key]:
        path = os.path.join(basePath[key], n)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        img = cv2.bitwise_not(img)
        img[img>0] = 1
        img = img.flatten()
        data['label'].append(key)
        data['imageData'].append(img)

data
df = pd.DataFrame(data)
df

col = [x for x in range(0, len(df['imageData'][0]))]
dat = pd.DataFrame (columns=col)


for i in range (0, len(df['imageData'])):
    row = pd.DataFrame([df['imageData'][i]], columns=col)
    dat = pd.concat([dat, row], ignore_index=True)
    

#Generating Feautures and labels and data spliting
y = np.array(pd.get_dummies(df['label']), np.int64)
x = np.array(dat, np.int64)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#Model Training
model = Sequential()
model.add(Dense(2048, input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=40, restore_best_weights=True)
history = model.fit (x_train, y_train, validation_data=(x_test, y_test), callbacks=[stop], epochs=100, batch_size=20, verbose=1)


#History Plotting
history.history.keys()
his = history.history

accuracy = his['accuracy']
val_accuracy = his['val_accuracy']
loss = his['loss']
val_loss = his['val_loss']

plt.plot(accuracy, color='red', label='Accuracy')
plt.plot(val_accuracy, color='orange', label='Val Accuracy')
plt.plot(loss, color='blue', label='Loss')
plt.xlabel('Epoches')
plt.legend()
plt.show()


#Predicting on test data
pred = model.predict(x_test)
y_pred = []

for i in range(0, len(pred)):
    index = np.argmax(pred[i])

    if index == 0:
        y_pred.append('circle')
    if index == 1:
        y_pred.append('square')
    if index == 2:
        y_pred.append('triangle')

y_true = []
for i in range(0, len(y_test)):
    index = np.argmax(y_test[i])

    if index == 0:
        y_true.append('circle')
    if index == 1:
        y_true.append('square')
    if index == 2:
        y_true.append('triangle')
       

#Accuracy Check
accuracy_score(y_pred, y_true)


#Confusion Metrix
con = confusion_matrix (y_pred, y_true)
sns.heatmap(con, annot=True, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')


#Saving Model
model.save("shape_classifier.h5")