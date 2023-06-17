#!/usr/bin/env python
# coding: utf-8

# # Student number:217633567
# Name:Wenceslas Assinga Mbourou

# ### Importing Libraries

# In[ ]:


### importing libraries 
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier, plot_tree


import warnings
warnings.filterwarnings("ignore")



### Loading the data
data = pd.read_csv('spam.csv', encoding='latin')
data.shape


data.head()

data.tail()

data.info()

data.describe()

### Data Cleaning 

# Drop unnecessary columns
data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
data.head()

# Update the colum names 
data.rename(columns={"v1":"type", "v2":"message"}, inplace=True)
data.head()


# Change the datatype of variable type to be categorical
data["type"] = data["type"].astype("category")
data.info()


###   Feature Engineering and Univariate Analysis


data['message_length'] = data['message'].apply(len)
data.head()

# Plot the frequency distribution for the message_length
plt.title("MESSAGE LENTH DISTRIBUTION")
data["message_length"].plot(bins=50, kind='hist', figsize=(10,7))

# Plot the frequency distribution based on the label
data.hist(column='message_length', by='type', bins=50, figsize=(15,8))

#create an instance of label encoder
encoder = LabelEncoder()

#label encode the variable type [0, 1]
data["type"] = encoder.fit_transform(data["type"])  
data.head()

### Creating Training and Test set

# Vectorize the variable message that will be used as x
count = CountVectorizer()
input = ['REMINDER FROM O2: To get 2.50 pounds free call credit and details of great offers pls reply 2 this text with your valid name, house no and postcode']

x = count.fit_transform(data['message'])
y = data["type"]

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.20, random_state=52)
x

# Shape of train data
(x_train.shape), (y_train.shape)

# Shape of test data
(x_test.shape), (y_test.shape)


### Creating and Training the model

# Creating an instance of Decision Tree Classifier model
model = DecisionTreeClassifier(random_state=99)

# Fitting the model
model.fit(x_train, y_train)

# Make predictions on the test set
prediction = model.predict(x_test)
prediction

### Evaluating the model

print("ACCURACY SCORE : {}". format(accuracy_score(y_test, prediction)))
print("PRECISION SCORE : {}". format(precision_score(y_test, prediction)))

 

