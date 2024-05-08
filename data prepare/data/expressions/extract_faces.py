# Author: Peter Kovac xkovac66
# Date: 2024-03-27
# Description: Finds faces in images and saves them to 'cropped' folder in the same directory as the image

import glob
import cv2
import os
import shutil

# extract folders from current directory
folders = glob.glob('*/')

# load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for folder in folders:
    if "confusion" not in folder:
        continue
    print(folder)

    folder_cropped = os.path.join(folder, 'cropped')
    # remove existing cropped folder
    if os.path.exists(folder_cropped):
        shutil.rmtree(folder_cropped)
    
    os.mkdir(folder_cropped)

    # list all image for current expression
    files = glob.glob(os.path.join(folder, '*.jpg'))
    for file in files:
        img = cv2.imread(file)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=10)
        x, y, w, h = faces[0]

        face  = img[y:y+h, x:x+w]
        
        cv2.imwrite(os.path.join(folder_cropped, os.path.basename(file)), face)