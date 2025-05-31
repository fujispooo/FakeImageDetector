#!/usr/bin/env python
# coding: utf-8

# # Fake Image Detection with ELA and CNN

# Agus Gunawan, Holy Lovenia, Adrian Hartanto Pramudita

# For more information: [Indonesian documentation](https://github.com/agusgun/FakeImageDetector/blob/master/docs/Deteksi%20Pemalsuan%20Gambar%20dengan%20ELA%20dan%20Deep%20Learning.pdf)

# ![full-architecture](docs/model-architecture.jpg "Architecture")

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

sns.set(style='white', context='notebook', palette='deep')


# ## Initial preparation

# In[2]:


from PIL import Image
import os
from pylab import *
import re
from PIL import Image, ImageChops, ImageEnhance


# #### Functions

# In[3]:


def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]


# In[4]:


def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    
    return ela_im


# ### Sample: Real Image

# Let's open a real (not-fake) image as a sample.

# In[20]:


Image.open('datasets/train/real/Au_ani_00001.jpg')


# This is how it looks like after it is processed with error-level analysis (ELA).

# In[21]:


convert_to_ela_image('datasets/train/real/Au_ani_00001.jpg', 90)


# ### Sample: Fake Image

# This is how it looks like after it has been edited.

# In[22]:


Image.open('datasets/train/fake/Tp_D_NRN_S_N_ani10171_ani00001_12458.jpg')


# This is the result of the fake image after getting through ELA. We can compare the difference between the picture below and the real picture's ELA result.

# In[23]:


convert_to_ela_image('datasets/train/fake/Tp_D_NRN_S_N_ani10171_ani00001_12458.jpg', 90)


# With our own naked eyes, we are able to differ which is the ELA result of the real picture and which one is the result of fake image. By saying real here, what we mean is a non-CGI picture that is not fabricated/edited in any way, e.g. splicing.

# ## Data preparation

# ### Read dataset and conversion to ELA

# In[5]:


dataset = pd.read_csv('datasets/dataset.csv')


# In[6]:


X = []
Y = []


# In[7]:


for index, row in dataset.iterrows():
    X.append(array(convert_to_ela_image(row[0], 90).resize((128, 128))).flatten() / 255.0)
    Y.append(row[1])


# ### Normalization

# In[8]:


X = np.array(X)
Y = to_categorical(Y, 2)


# ### Reshape X

# In[9]:


X = X.reshape(-1, 128, 128, 3)


# ## Train-test split

# In[11]:


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)


# ## CNN building

# In[12]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', 
                 activation ='relu', input_shape = (128,128,3)))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', 
                 activation ='relu'))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))


# In[13]:


model.summary()


# ### Add optimizer

# In[14]:


optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)


# In[15]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# ### Define early stopping

# In[16]:


early_stopping = EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')


# ### Model training

# In[17]:


epochs = 30
batch_size = 100


# In[18]:


history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, Y_val), verbose = 2, callbacks=[early_stopping])


# ## Performance measure

# ### Accuracy and loss curves during training-validation

# In[19]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# ### Confusion matrix

# In[20]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2))

