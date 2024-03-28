#import libraries
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
path='img_data'
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')#read images from path folder and cl namme
    images.append(curImg)#add them to images list
    classNames.append(os.path.splitext(cl)[0])
#train for the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#convert to rgb
        encoded_face = face_recognition.face_encodings(img)[0]# use face_recognition library to get the face encodings
        encodeList.append(encoded_face)# append encodings in list
    return encodeList
encoded_face_train = findEncodings(images)
#print(encoded_face_train)

with open('trained_data.csv','r+') as t:
    for i in range(len(encoded_face_train)):
        x=encoded_face_train[i]
        x=x.tolist()
        #print(x,type(x))
        for j in range(len(x)):
            #print(x[j],type(x[j]))
            t.writelines(str(x[j])+',')
        t.writelines("\n")
    t.close()
    