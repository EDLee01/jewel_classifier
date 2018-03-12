from keras.models import Sequential
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import TensorBoard
import time
import os

s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
train_data_dir = 'train_data'
validation_data_dir = 'test_data'
img_width,img_height = 100,100
batch_size = 100

train_datagen = ImageDataGenerator(
         rescale=1./255)

test_datagen = ImageDataGenerator( rescale=1./255 )
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',# this means our generator will only yield batches of data, no labels
        classes=['bixi','changshi','ganlanshi','hongbaoshi','hupo','kongqueshi','lanbaoshi','shiliushi','shuijing','zhenzhu','zuanshi','zumulv'],
        shuffle=False)

validation_generator = train_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',# this means our generator will only yield batches of data, no labels
        classes=['bixi','changshi','ganlanshi','hongbaoshi','hupo','kongqueshi','lanbaoshi','shiliushi','shuijing','zhenzhu','zuanshi','zumulv'],
        shuffle=False)

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
logs_path = 'log_%s'%(s_time)


try:
    os.makedirs(logs_path)
except:
    pass

##tensorboard = TensorBoard(log_dir=logs_path, histogram_freq=1,write_graph=True)

model.fit_generator(
        train_generator,
        steps_per_epoch=60,
        epochs=30,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=60,
)
model.save('my_model1.h5')
