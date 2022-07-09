
import os
import glob
import random
import numpy as np
import pandas as pd

import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm

from PIL import Image
from tensorflow.keras.utils import to_categorical

from azureml.core.workspace import Workspace
from azureml.core.run import Run
# from azureml.core.dataset import Dataset
from azureml.core.model import Model as azure_model



ws = Workspace(
    subscription_id = "88ffd436-6b2f-4a5c-942f-72cc66d5bfab",
    resource_group = "aarav_resources",
    workspace_name = "aarav_workspace",
)


train_csv=pd.read_csv("Human Action Recognition data/Training_set.csv")
test_csv=pd.read_csv("Human Action Recognition data/Testing_set.csv")


train_fol=glob.glob('Human Action Recognition data/Train/*')
test_fol=glob.glob('Human Action Recognition data/Test/*')


filename= train_csv['filename']
action=train_csv['label']

img_data=[]
img_label=[]
lenght=len(train_fol)
for i in range(len(train_fol)-1):
    t='Human Action Recognition data/Train/' + filename[i]
    temp_img=Image.open(t)
    img_data.append(np.asarray(temp_img.resize((160,160))))
    img_label.append(action[i])
    
img_dta = img_data
img_dta = np.asarray(img_dta)
y_train = to_categorical(np.asarray(train_csv['label'].factorize()[0]))
print(y_train[0])

vgg_model = Sequential()

trained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet',classifier_activation='softmax')


for layer in trained_model.layers:
        layer.trainable=False

vgg_model.add(trained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))

vgg_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
run = Run.get_context()

# vgg_model.summary()

history = vgg_model.fit(img_dta,y_train, epochs=1)

#saving


vgg16_path= os.path.join('','model_weight')
os.makedirs(vgg16_path,exist_ok=True)



vgg_model.save(os.path.join(vgg16_path,'full_model.h5'))
azure_model.register(workspace=ws,model_path=vgg16_path + "/full_model.h5",model_name="actrecog")
run.complete()