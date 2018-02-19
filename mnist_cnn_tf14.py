
# coding: utf-8

# In[19]:


# Simple CNN for the MNIST Dataset
import numpy

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.utils import to_categorical
# fix dimension ordering issue
from tensorflow.python.keras import backend as K
print(K.image_data_format())

import numpy as np


# In[2]:


#K.set_image_data_format('channels_first')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[3]:


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)


# In[4]:


# reshape to be [samples][channels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32')

print(X_train.shape,X_test.shape)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


# In[5]:


# define a simple CNN model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = baseline_model()


# In[46]:


# Fit the model
model.fit(X_train, y_train, validation_split=0.25, 
          epochs=3, batch_size=10)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


# In[47]:


print(model.summary())


# In[52]:


# Save train
from tensorflow.python.keras.models import save_model


# In[53]:


save_model(model,'./mnist_model')

