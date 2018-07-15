#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs a RandomForestRegressor on the processed_order_data.csv
Given a geohash and time slot, The model can predict the number of orders
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import geohash
from sklearn.metrics import mean_squared_error
import math
from sklearn.externals import joblib

# Get the longitude and latitude from the geohash
def decodegeo(geo, latlong):
    if len(geo) >= 5:
        geodecoded = geohash.decode(geo)
        return geodecoded[latlong]
    else:
        return 0

def transform_dataset(dataset):
    # Find the lat-long of the center of each geohash
    dataset['latitude'] = dataset['geohash'].apply(lambda geo: decodegeo(geo, 0))
    dataset['longitude'] = dataset['geohash'].apply(lambda geo: decodegeo(geo, 1))

    # Scale Features
    dataset['hour_scaled'] = dataset['hour_of_day']/24.0
    dataset['day_scaled'] = dataset['day_of_week']/7.0
    
    # Add Features
    dataset['hour_sin'] = (dataset['hour_of_day'] * 2 * math.pi).apply(math.sin)
    dataset['hour_cos'] = (dataset['hour_of_day'] * 2 * math.pi).apply(math.cos)
    dataset['day_sin'] = (dataset['day_of_week'] * 2 * math.pi).apply(math.sin)
    dataset['day_cos'] = (dataset['day_of_week'] * 2 * math.pi).apply(math.cos)
    return dataset

# Read data
dataset = pd.read_csv('processed_order_data.csv')

# Transform data
dataset = transform_dataset(dataset)

# Select the features to be used in ML model
Xfeatures = ['latitude', 'longitude', 'hour_scaled', 'day_scaled', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'weekend']
X = dataset[Xfeatures]
y = dataset['num_orders']

# Split dataset to training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Take Logarithm of the num_orders to reduce the amount of impact by outliers
y_train = np.log10(y_train)
y_test = np.log10(y_test)

# Fit Regression Model
reg = RandomForestRegressor(n_estimators=50, max_depth=30, n_jobs=-1, warm_start=True)
reg.fit(X_train,y_train)

# Calculate Training and Test Accuracy
training_accuracy = reg.score(X_train, y_train)
test_accuracy = reg.score(X_test, y_test)

# Calculate Root mean squared error
rmse_train = np.sqrt(mean_squared_error(reg.predict(X_train),y_train))
rmse_test = np.sqrt(mean_squared_error(reg.predict(X_test),y_test))
print("Training Accuracy = %0.3f, Test Accuracy = %0.3f, RMSE (train) = %0.3f, RMSE (test) = %0.3f" % (training_accuracy, test_accuracy, rmse_train, rmse_test))

# Print Actual and Predicted values for first 50 test set
#pd.DataFrame(np.round(np.power(10,np.column_stack((reg.predict(X_test),y_test))), decimals=0).astype(int)).head(20)

# Store the trained model
joblib.dump(reg, 'trained_random_forest.pkl')
