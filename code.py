#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the dataset

# In[2]:


get_ipython().run_cell_magic('time', '', '\n# load data in dataset\ndf_train = pd.read_csv("train.csv")\ndf_test = pd.read_csv("test.csv")')


# In[3]:


# Size and types of the dataframe
df_train.info()


# In[4]:


#Exploring the train_data
df_train.head()


# In[5]:


df_train.shape


# In[6]:


df_test.shape


# In[7]:


#Exploring the columns of data
df_train.columns


# In[8]:


df_test.columns


# ### Data breakdown

# In[10]:


# Separate features from Loss value in train dataset.

pd.set_option('display.max_columns',None)
loss = df_train['loss']
features = df_train.drop('loss', axis=1)


# In[11]:


# Analyse stats on continuous features
features.describe()


# ### Exploratory Visualization

# In[12]:


# Plot heat map to understand correlation between continous features 

plt.figure(figsize=(10, 6))
sns.set()
sns.heatmap(features.iloc[:,117:].corr())


# Heatmap
# 
# To find out correlation among continuous feature. From the figure we can see lighter region in heatmap shows strong correlation and darker region depicts weaker correlation among feature. Features showing strong correlation are listed below.
# 
# - Cont11 and cont12
# - Cont1 and cont9
# - Cont6 and cont10
# - Cont1 and cont10

# In[13]:


# Plot scatter plots to understand correlation between continuous features

pd.plotting.scatter_matrix(features.iloc[:2000,117:], alpha=0.1, figsize = (40,25), diagonal='kde')


# Scatter Matrix
# 
# Scatter Matrix is plotted between each continuous features to review scatter plot distribution to understand data correlation and skewness. Here i have sampled first 2000 rows to review data distribution. Scatter plot visualization depicts similar correlations between features compared to what i discovered in heatmap visuals.

# ## Data Preprocessing
# 
# Hot encoding
# 
# Categorical variables will be one hot encoded to numeric values from string values as we can only input numeric values to model for training. One hot encoding is done through pandas get_dummies function which converts all categorical variable into one hot representation of numeric values. After hot encoding we observe 116 categorical features are converted into 1139 numeric features.

# In[14]:


# Hot encode categorical features and review dimension of dataset after one hot encoding, 116 cat features are converted
# into 1139 + 14 cont features + 1 id feature = 1154 features  

features = pd.get_dummies(features)
print(features.shape)


# Seperating Categorical and Continuous feature
# dropping the 'Id' column from dataframe since it of no use
# Cat and Cont features are seperated so that we can perform feature transformation seperately on each of the dataset.

# In[15]:


features = features.drop('id',axis=1)
cont_features = features.iloc[:,:14]
cat_features = features.iloc[:,14:]


# In[17]:


from sklearn.decomposition import PCA


# Principal Component Analysis on Continuous Features.

# In[18]:



# PCA on continuous features

list = []
pca = PCA(n_components=11)
pca.fit(cont_features)
reduced_cont_feature = pca.transform(cont_features)
list.append(pca.explained_variance_ratio_)
print(list)


# Principal Component Analysis on categorical Features.

# In[19]:


# Perform PCA for dimensionality reduction. Run PCA for number of components = total number of features after hot encoding to 
#understand explained variance ratio for all dimensions

list = []
pca = PCA(n_components=1139)
pca.fit(cat_features)
transformed_feature = pca.transform(cat_features)
list.append(pca.explained_variance_ratio_)


# In[20]:


# Review explained variance and derive number of dimensions to be considered for 99% variance.  

np_list = np.array(list)
np.set_printoptions(threshold=1500)
print(np_list)


# In[21]:


list = []

pca = PCA(n_components=345)
pca.fit(cat_features)
reduced_cat_feature = pca.transform(cat_features)
list.append(pca.explained_variance_ratio_)
print(list)


# In[22]:


# After PCA combine Cat and Cont features into single dataset.  
reduced_feature = np.hstack((reduced_cat_feature,reduced_cont_feature))
reduced_feature.shape


# In[29]:


loss_log_bench = np.log(loss)
# Review data skewness. Log transform to get normal or uniform distribution.

print("data skewness before log transform {}".format(loss.skew()))
loss_log = np.log(loss+1)
print("data skewness after log transform {}".format(loss_log.skew()))


# In[38]:


# Mean absolute error performance metric.
from sklearn.metrics import mean_absolute_error

def performance_metric(y_true, y_predict):
    mae = mean_absolute_error(y_true, y_predict) #Calculate the performance score between 'y_true' and 'y_predict'
    return mae


# ## Building model by using Keras

# In[35]:


# Deep NN training using Keras 

#Import required libraries 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint  
from keras import optimizers


# In[34]:


from sklearn.model_selection import train_test_split


# In[36]:


# Split the original data into train and test in ratio of 9:1 
X_train, X_test, y_train, y_test = train_test_split(reduced_feature, 
                                                    loss_log, 
                                                    test_size = 0.1, 
                                                    random_state = 0)

model = Sequential()

# First Dense Layer
model.add(Dense(356, input_dim=reduced_feature.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.3))

# Hidden Layer
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Initialize optimizer with lr
adam = optimizers.adam(lr=0.0001)

# Compile model
model.compile(loss='mean_absolute_error', optimizer=adam)

# Define model checkpoint with filepath to save trained params
checkpointer = ModelCheckpoint(filepath='weights.best.DeepNN.hdf5', 
                               verbose=1, save_best_only=True)


# In[40]:


# Fit Train data with 10% data used for model inferencing. 
model.fit(X_train, y_train, 
          validation_split=0.1,
          epochs=50, batch_size=250, callbacks=[checkpointer], verbose=1)

# Load best weights from saved model checkpoint
model.load_weights('weights.best.DeepNN.hdf5')

# predict on test data
y_pred = model.predict(X_test)


# In[39]:


# Compute MAE from predicted and actual loss
print("Mean Absolute error: %.2f"
      % performance_metric(np.exp(y_test), np.exp(y_pred)))


# In[48]:


test_ids = df_test['id']


# In[53]:


def save_predictions(ids = None, predictions = None, file = None):
    
    # prepare file
    submission = pd.DataFrame({'id': ids, 'loss': predictions})
    
    # CSV
    submission.to_csv(path_or_buf = file, index = False, encoding='utf8')
    print("Data prediction stored!")


# In[54]:


save_predictions(ids = test_ids, 
                 predictions = predictions, 
                 file = 'keras_submission.csv')


# In[ ]:




