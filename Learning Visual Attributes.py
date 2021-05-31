#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt  


# In[43]:


# Read in training data
data = pd.read_csv('data_train.csv')


# In[44]:


# Drop rows with missing data
data = data.dropna()


# In[45]:


# get y values for color and convert with labelencoding
colour_categ = data.loc[:,'color']
color_le = LabelEncoder()
color_le = color_le.fit(colour_categ)
y_color = color_le.transform(colour_categ)


# In[5]:


# Shows the color dataset is not balanced
unique, frequency = np.unique(colour_categ, 
                              return_counts = True)
# print unique values array
print("Unique Values:", 
      unique)
  
# print frequency array
print("Frequency Values:",
      frequency)


# In[6]:


# Get X values for the color model
X_color = data.loc[:,'lightness_0_0':'blueyellow_2_2']


# In[7]:


# Split into test and training data, 25% is using for the validation set.
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_color, y_color, test_size = 0.25, random_state = 69
)


# In[8]:


# Determine best parameters for mlp through cross-validation
mlp = MLPClassifier(random_state = 420, max_iter = 500) # predetermined random state for reproducible results
color_scaler = MinMaxScaler().fit(X_train_c)
X_train_c_scaled = color_scaler.transform(X_train_c)
param = [{
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation':['relu','tanh'],
    'alpha':[0.001,0.0001,0.00001,0.01],
    'solver':['lbfgs', 'adam']
}]
gscv = GridSearchCV(estimator = mlp, param_grid = param)


# In[9]:


# Cross Validation Process
gscv.fit(X_train_c_scaled,y_train_c)


# In[10]:


# Results of Above
print('Best Params: ', gscv.best_params_)


# In[11]:


# Make MLP Classifier with best settings. 
# NOTE = activation and solver are at their default values, but are added for clarity
best_mlp = MLPClassifier(max_iter = 500, random_state = 420, activation='relu', solver = 'adam', alpha = 0.00001)


# In[12]:


c_pipe = make_pipeline(color_scaler, best_mlp)
c_pipe.fit(X_train_c, y_train_c)
c_pipe.score(X_test_c, y_test_c)


# In[39]:


plot_confusion_matrix(c_pipe, X_test_c,y_test_c)
plt.show()


# In[41]:


print(color_le.classes_)


# In[13]:


texture_categ = data.loc[:, 'texture']
texture_le = LabelEncoder().fit(texture_categ)
y_texture = texture_le.transform(texture_categ)
X_texture = data.loc[:, 'hog_0_0_0':'bimp_8_16_single_2_2']


# In[14]:


# Shows the texture dataset is not balanced
unique, frequency = np.unique(texture_categ, 
                              return_counts = True)
# print unique values array
print("Unique Values:", 
      unique)
  
# print frequency array
print("Frequency Values:",
      frequency)


# In[15]:


X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_texture, y_texture, test_size = 0.25, random_state = 69
)


# In[16]:


texture_scaler = MinMaxScaler().fit(X_train_t)
X_train_t_scaled = texture_scaler.transform(X_train_t)


# In[17]:


gscv.fit(X_train_t_scaled,y_train_t)


# In[18]:


# Results of Above
print('Best Params: ', gscv.best_params_)


# In[19]:


#MLP classifier with best parameters
best_mlp_t = MLPClassifier(max_iter = 500, random_state = 420, activation='relu', solver = 'adam', alpha = 0.01)


# In[20]:


#Pipeline for texture
t_pipe = make_pipeline(texture_scaler, best_mlp_t)
t_pipe.fit(X_train_t, y_train_t)
t_pipe.score(X_test_t, y_test_t)


# In[37]:


plot_confusion_matrix(t_pipe, X_test_t,y_test_t)
plt.show()


# In[38]:


print(texture_le.classes_)


# In[21]:


test_data = pd.read_csv('data_test.csv')


# In[22]:


# Impute the missing test data in the colour predictors
test_data_X_c = test_data.loc[:,'lightness_0_0':'blueyellow_2_2']
imputer_c = IterativeImputer()
imputer_c = imputer_c.fit(test_data_X_c)
test_data_X_c_imputed = imputer_c.transform(test_data_X_c)


# In[23]:


#Ensure the scaler used is the same used for the training data, and the classifier paramters are the same
#Predicting color values for data_test.csv
c_pipe_final = make_pipeline(MinMaxScaler().fit(X_color), best_mlp)
c_results = color_le.inverse_transform(c_pipe_final.predict(test_data_X_c_imputed))


# In[24]:


# print(c_results)


# In[25]:


# Impute the missing test data in the colour predictors
test_data_X_t = test_data.loc[:, 'hog_0_0_0':'bimp_8_16_single_2_2']
imputer_t = IterativeImputer()
imputer_t = imputer_t.fit(test_data_X_t)
test_data_X_t_imputed = imputer_t.transform(test_data_X_t)


# In[26]:


#Predicting texture values for data_test.csv
t_pipe_final = make_pipeline(MinMaxScaler().fit(X_texture), best_mlp_t)
t_results = texture_le.inverse_transform(t_pipe_final.predict(test_data_X_t_imputed))


# In[27]:


print(t_results)


# In[28]:


pd.DataFrame(c_results).to_csv('color_results.csv', header = None, index = False)


# In[29]:


pd.DataFrame(t_results).to_csv('texture_results.csv', header = None, index = False)


# In[30]:


#Question 8 LogisticRegression Pipelines
c_log_reg_pipe = make_pipeline(MinMaxScaler(), LogisticRegression())
c_log_reg_pipe.fit(X_train_c, y_train_c)
c_log_reg_pipe.score(X_test_c,y_test_c)


# In[31]:



t_log_reg_pipe = make_pipeline(MinMaxScaler(), LogisticRegression())
t_log_reg_pipe.fit(X_train_t, y_train_t)
t_log_reg_pipe.score(X_test_t,y_test_t)

