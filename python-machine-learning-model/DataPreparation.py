#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREPROCESS Swiggy order_data.csv 
    => Generates `processed_order_data.csv`
    => processed data containst Number of orders based on (GeoHashes, Hour_Of_Day, Day_Of_Week, Is_Weekend)
"""

import pandas as pd
from datetime import datetime
import numpy as np
from multiprocessing import cpu_count, Pool
cores = cpu_count()
partitions = cores

weekday_to_str = {0: "Monday",
                  1: "Tuesday",
                  2: "Wednesday",
                  3: "Thursday",
                  4: "Friday",
                  5: "Saturday",
                  6: "Sunday"}

geohash_accuracy = 6

def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def map_record(record):
    order_datetime = datetime.strptime(record['ordered_time'], 
                                         '%Y-%m-%d %H:%M:%S.%f')

    weekend = 0
    if order_datetime.weekday() in [5,6]:
        weekend = 1

    # Categorical Feature
    hour_of_day = order_datetime.hour  

    # Categorical Feature
    day_of_week = order_datetime.weekday()    
    
    customer_geohash = record['customer_geohash'][0:geohash_accuracy]
    new_record = [customer_geohash, hour_of_day, day_of_week, weekend]
    return new_record

def map_dataset(dataset):
    dataset = dataset.apply(lambda record: pd.Series(map_record(record), index=['geohash', 'hour_of_day', 'day_of_week', 'weekend']), axis=1)
    return dataset
    
dataset = pd.read_csv('dataset/Dataset_1_Order_Data/order_data.csv')
dataset = dataset.dropna(subset=['ordered_time', 'customer_geohash'])
dataset = parallelize(dataset, map_dataset)
dataset = dataset.groupby(['geohash', 'hour_of_day', 'day_of_week', 'weekend']).size().reset_index(name='num_orders')

dataset.to_csv('processed_order_data.csv', encoding='utf-8', index=False)
