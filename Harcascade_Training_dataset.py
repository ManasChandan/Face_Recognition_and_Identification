# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:34:46 2020

@author: Manas C Behera
"""
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
scale_factor = 1.3
ide = input('enter the id')
sampleno = 0
while 1:
    ret , pic = cap.read()
    gray = cv2.cvtColor(pic , cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,scale_factor,1)
    for (x,y,w,h) in face:
        sampleno = sampleno +1 
        cv2.imwrite('dataset2/User.'+str(ide)+'.'+str(sampleno)+'.jpg',gray[y:y+h,x:x+w])
        cv2.rectangle(pic,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('face',pic)
        cv2.waitKey(100)
    if sampleno > 25:
        break
cap.release()
cv2.destroyAllWindows()
