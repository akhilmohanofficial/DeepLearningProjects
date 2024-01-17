#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers


# In[4]:


#reading the dataset
hotel_df = pd.read_csv('/Users/akhilmohan/Downloads/Cleaned_Hotel_Info_Updated.csv')
hotel_df.info()


# In[5]:


# Remove the irrelevant columns from the dataset
cols_to_remove = ['_id', 'name', 'address','label']
hotel_df = hotel_df.drop(cols_to_remove, axis=1)


# In[6]:


#To make this price predictor, let's grow a CART tree
#We want to include 'City' and 'Room Type' as part of our decision tree, because they are important factors that may affect the price
#Let's use one hot encoding to encode these two variables.
# extract categorical variables
cat_vars = ['City','Room_type']

# one-hot encode categorical variables
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(hotel_df[cat_vars])
onehot = enc.transform(hotel_df[cat_vars]).toarray()

# concatenate one-hot encoded variables with numerical variables

hotel_df = pd.concat([hotel_df, pd.DataFrame(onehot)], axis=1)

#We want to highlight that since our categorical variables are not ordinal (NY/Chicago/LA do not represent ordered data), we may get arbitrary splits in our decision trees.


# In[8]:


X = hotel_df.drop(['price','City','Room_type'], axis=1)
X.columns = X.columns.astype(str)
Y = hotel_df['price']
print(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[9]:


# Fit the CART Tree model on the training data
cart_model = DecisionTreeRegressor(random_state=42)
cart_model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = cart_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error (CART Tree):", mse)
print("R-squared value (CART Tree) on test data (Out of Sample Rsq):", r2)


# The CART model provided an MSE of 10487 and an out of sample Rsq of 17.9%. Clearly , it is not a great model at predicting the price as it is not able to generalize well to the new data and has poor performance on the test set.

# In[10]:


# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(cart_model, feature_names=X.columns, filled=True)
plt.show()


# In[15]:


# Fit the XGBoost model on the training data
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error (XGBoost):", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared (XGBoost):", r2)


rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root mean squared error (XGBoost):", rmse)


# Calculate and print feature importances
feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False)
print("Feature importances:\n", feat_imp)


# The model using Boosting (XGBoost) gives an MSE of 12458 and an out of sample Rsq of just 2.5%

# In[13]:


# Fit the Random Forest model on the training data
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error (Random Forest):", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared (Random Forest):", r2)


# The Random forest provided a much smaller MSE (6341) and better out of sample Rsq (50%), clearly indicating that the ensemble method is better at predicting the price of the Hotel rooms.

# In[16]:


#Fit the data on a neural network

# Define the model architecture
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(1)
])

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the testing data
loss, mse, mae = model.evaluate(X_test, y_test)
print("Mean squared error (Neural Network):", mse)
r2 = r2_score(y_test, model.predict(X_test))
print("R-squared (Neural Network):", r2)


# The neural network provides an MSE of 10013 and an out of sample Rsq of 21.6%. It is performing better than the CART and Boosted Trees, but not better than the Random Forest.

# In[20]:


#Fit the data on a neural network

# Define the model architecture
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(1)
])

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2)

# Evaluate the model on the testing data
loss, mse, mae = model.evaluate(X_test, y_test)
print("Mean squared error (Neural Network):", mse)
r2 = r2_score(y_test, model.predict(X_test))
print("R-squared (Neural Network):", r2)


# Thus,considering the MSE and the out of sample Rsq, the final model for price prediction should be Random Forest.
