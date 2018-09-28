
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense,Dropout
from keras.callbacks import TensorBoard

import cnn_data

model=Sequential([
    Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Convolution2D(64,3,3,input_shape=(128,128,3),activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.summary()

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.fit_generator(
    cnn_data.train_flow,steps_per_epoch=10,epochs=10,verbose=1,validation_data=cnn_data.test_flow,validation_steps=10,
    callbacks=[TensorBoard(log_dir='./logs/1')]
)

model.save('./catdogs_model.h5')