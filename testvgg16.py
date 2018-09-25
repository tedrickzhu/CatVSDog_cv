#encoding=utf-8

from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img

import numpy as np

label = np.array(['cat','dog'])
model = load_model('vgg16catVSdog.h5')

image = load_img('/home/zzy/TrainData/test1/1.jpg')

image = image.resize((150,150))
image = img_to_array(image)
image = image/255
image = np.expand_dims(image,0)

print(label[model.predict_classes(image)])
