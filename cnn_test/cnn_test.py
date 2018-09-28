#encoding=utf-8

from keras import models
import numpy as np
import cv2

from os import listdir

import cnn_data

def get_inputs(src=[]):
    pre_x = []
    for s in src:
        input = cv2.imread(s)
        # print(type(input))
        input = cv2.resize(input, (128, 128))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)  # input一张图片
    pre_x = np.array(pre_x) / 255.0
    return pre_x

def put_prey(pre_y,label):
    output=[]
    for y in pre_y:
        if y[0]<0.5:#二分类，此处只用一个神经元输出
            output.append([label[0],1-y[0]])
        else:
            output.append([label[1], y[0]])
    return output

def getfilelist(datadir):

    return

    # print(type(testlist))
    # print(str(testlist[0]))

model=models.load_model('catdogs_model.h5')

datadir = '../test500/dog/'
file = ['../test500/cat/cat.4.jpg','../test500/cat/cat.8.jpg','../test500/cat/cat.35.jpg','../test500/cat/cat.56.jpg']
filelist = [datadir + name for name in listdir(datadir)]

pre_x=get_inputs(filelist)

pre_y=model.predict(pre_x)

output=put_prey(pre_y, list(cnn_data.train_flow.class_indices.keys()))
for i in range(len(output)):
    res = output[i]
    # print(res, i)
    if res[0]=='cat':
        print(res,i)
print(output)
# cv2.waitKey(0)