#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Predictions as a JSON List for the current day for all unique Geohashes. 
It is used in Frontend for Visualization.
"""

from sklearn.externals import joblib
import pandas as pd
from RandomForest import decodegeo, transform_dataset
import numpy as np
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler

    
dataset = pd.read_csv('processed_order_data.csv')

# Prepare DataSet for the current day (To be used by the client for visualization)
# Every unique geohash combined with every time slot of the current day
unique_geohashes = pd.DataFrame(dataset['geohash'].drop_duplicates())
unique_geohashes['key'] = 1

hour_of_day = pd.DataFrame(np.array(range(24)), columns=['hour_of_day'])
hour_of_day['key'] = 1

dataset_pred = pd.merge(unique_geohashes, hour_of_day).drop(['key'], axis=1)

day_of_week = datetime.now().weekday()
weekend = 0
if day_of_week in [5,6]:
    weekend = 1
    
dataset_pred['day_of_week'] = day_of_week
dataset_pred['weekend'] = weekend

# Generate Predictions
dataset_pred = transform_dataset(dataset_pred)

Xfeatures = ['latitude', 'longitude', 'hour_scaled', 'day_scaled', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'weekend']
X = dataset_pred[Xfeatures]

reg = joblib.load('trained_random_forest.pkl')
y = pd.DataFrame(np.round(np.power(10,reg.predict(X)),decimals=0).astype(int), columns=['num_orders'])

# Transform data to be sent to front-end for plotting
scaler = MinMaxScaler()
order_weights = scaler.fit_transform(y)

output = pd.concat([dataset_pred[['geohash', 'hour_of_day']], pd.DataFrame(order_weights, columns=['weight'])], axis=1)

with open('predictions.json', 'w') as predictions_file:
    predictions_file.write(output.to_json(orient='records'))


