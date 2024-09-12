from keras.models import load_model
import cv2
import numpy as np


class ModelPredict:
    def __init__(self):
        self.model = load_model("shape_classifier.h5")
        
    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
        contours, h = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cont = max(contours, key=cv2.contourArea)
        imDraw = np.full((img.shape[0], img.shape[1]), 0, np.uint8)
        cv2.drawContours(imDraw, [cont], -1, (255, 255, 255), 2)
        imDraw = cv2.resize(imDraw, (28,28))
        imDraw[imDraw>0] = 1
        imDraw = imDraw.flatten()
    
    
        singlePred = self.model.predict(imDraw.reshape(1,784))
        
    
        index = np.argmax(singlePred)
        shape = ""
        conf = 0.0
        if index == 0:
            shape = 'Circle'
            conf = singlePred[0][index]
        if index == 1:
            shape = 'square'
            conf = singlePred[0][index]
        if index == 2:
            shape = 'triangle'
            conf = singlePred[0][index]

        return shape, conf