import tensorflow as tf
import tensorflow.keras.models as M
import numpy as np
import efficientnet.tfkeras as efn 
import wget
import cv2
import os 

print('Input URL')
url = input()
try:
    os.remove('car.jpg')
except:
    pass

wget.download(url,'car.jpg')

model = M.load_model('module_8/step4_12.hdf5')
img_size = np.array(model.input.shape)[[2, 1]]

image = cv2.imread('car.jpg', cv2.IMREAD_COLOR)[..., ::-1] 
image = cv2.resize(image, img_size)

image = image[None, ...]
image = image / 255

pred = model.predict(image)[0]

class_idx = pred.argmax()

dict ={'0': 'vaz2170',
  '1':'fordfocus', 
  '2': 'vaz2114',
  '3':'vaz2110',
  '4':'vaz2107',
  '5':'vaz2121', 
  '6':'vazkalina',
  '7':'vaz2109',
  '8':'Volkswagen Passat',
  '9': 'vaz21099' }

print(dict[str(class_idx)])