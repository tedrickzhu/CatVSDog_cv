#encoding=utf-8

from os import listdir
from shutil import copyfile
# from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
# import numpy as np
import cv2

def splitfile(trainpath,catpath,dogpath):
    allfile = listdir(trainpath)
    count =0
    for filename in allfile:
        # print(filename)
        # break
        if filename[0]=='c':
            copyfile(trainpath+filename,catpath+filename)
        if filename[0]=='d':
            copyfile(trainpath + filename, dogpath + filename)
        count+=1
        if count%100==0:
            print(count,' images have done')
    pass

def chooseData(trainpath,traincat,traindog,testcat,testdog):
    allfile = listdir(trainpath)
    cat =0
    dog = 0
    for filename in allfile:
        # print(filename)
        # break

        if filename[0]=='c' and cat <1000:
            copyfile(trainpath+filename,traincat+filename)
        elif filename[0] == 'c' and cat >=1000 and cat <1499:
            copyfile(trainpath + filename, testcat + filename)

        if filename[0]=='d' and dog <1000:
            copyfile(trainpath+filename,dogcat+filename)
        elif filename[0] == 'd' and dog >=1000 and dog <1499:
            copyfile(trainpath + filename, dogcat + filename)


def imgGen_test():
    # 图片生成器
    datagen = ImageDataGenerator(
        rotation_range=40,  # 随机旋转度数
        width_shift_range=0.2,  # 随机水平平移
        height_shift_range=0.2,  # 随机竖直平移
        rescale=1 / 255,  # 数据归一化
        shear_range=0.2,  # 随机裁剪
        zoom_range=0.2,  # 随机放大
        horizontal_flip=True,  # 水平翻转
        fill_mode='nearest',  # 填充方式
    )

    test_datagen = ImageDataGenerator(
        rescale=1 / 255
    )

    img = load_img(imgPath)  # 读入一张图片
    x = img_to_array(img)
    x = np.expand_dims(x, 0)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=savePath + 'temp', save_prefix='new_cate',
                              save_format='jpg'):
        i += 1
        if i == 20:
            break

if __name__ == '__main__':
    trainpath = '/home/zzy/TrainData/kaggle_cat_dog/train/'
    testpath = '/home/zzy/TrainData/kaggle_cat_dog/test1/'
    catpath = '/home/zzy/TrainData/kaggle_cat_dog/cat/'
    dogpath = '/home/zzy/TrainData/kaggle_cat_dog/dog/'

    imgPath = '/home/zzy/TrainData/kaggle_cat_dog/test1/34.jpg'
    savePath = '/home/zzy/TrainData/kaggle_cat_dog/'

    splitfile(trainpath,catpath,dogpath)

    # image = cv2.imread(trainpath+'cat.4567.jpg')
    # print(image.shape)

