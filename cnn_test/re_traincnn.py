from keras.callbacks import TensorBoard
from keras.models import load_model

import cnn_data

model=load_model('./catdogs_model_cnn.h5')

model.summary()

model.fit_generator(
    cnn_data.train_flow,steps_per_epoch=100,epochs=50,verbose=1,validation_data=cnn_data.test_flow,validation_steps=100,
    callbacks=[TensorBoard(log_dir='./logs/2')]
)

model.save('./catdogs_model_cnn.h5')