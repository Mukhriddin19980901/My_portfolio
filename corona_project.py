#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
import os 
import glob
import matplotlib.pyplot as plt
tf.config.list_physical_devices()


# In[2]:


def datas(path):
    imgs=[]
    labs=[]
    label=['NORMAL','PNEUMONIA']
    for i in range(len(label)):
        new_dir=path+label[i]+"/*"
        for image in glob.glob(new_dir):
            image=cv2.imread(image)
            image=cv2.resize(image,(96,96))
            imgs.append(image)
            if label[i]=="PNEUMONIA":
                labs.append(0)
            else:
                labs.append(1)
    return imgs,labs
train_dir=r'../datasets2021/chest_xray/train/'
test_dir=r'../datasets2021/chest_xray/test/'


# In[4]:


x_train=[]
y_train=[]
x_train,y_train=datas(train_dir)
x_train=np.array(x_train,dtype=np.float32)/255
y_train=np.array(y_train)
print(x_train.shape,y_train.shape)
print("...trained...")
x_test=[]
y_test=[]
x_test,y_test=datas(test_dir)
x_test=np.array(x_test,dtype=np.float32)/255
y_test=np.array(y_test)
print(x_test.shape,y_test.shape)
print("...tested...")


# In[6]:


data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
])


# In[10]:


model = keras.Sequential([
    data_augmentation,
    keras.layers.Conv2D(32,(3,3) ,activation='relu' ,input_shape=(96,96,3)),
    keras.layers.MaxPooling2D((3,3)),
    
    keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128,(3,3),activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dense(64,activation = 'relu'),
    keras.layers.Dense(32,activation = 'relu'),
    
    keras.layers.Dense(1,activation = 'sigmoid'),
])

with tf.device("/GPU:0"):
    model2=model
    model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    H=model2.fit(x_train,y_train,epochs=10)
model2.evaluate(x_test,y_test)
model2.summary()


# In[13]:


plt.plot(np.arange(10) , H.history['loss'][:10],label="loss")
plt.plot(np.arange(10) , H.history['accuracy'][:10],label="accuracy")
plt.title("corona detection using chest xray")
plt.xlabel("Epochs")
plt.legend(loc = 'best')


# In[14]:


model2.save('corona_detect_xray.model',save_format='h5')

