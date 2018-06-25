#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:12:39 2018

@author: kyleguan
"""

from tl_detector import TLClassifier
import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import time
from glob import glob

import boto
import findspark
findspark.init("/Users/kyleguan/spark-2.1.1-bin-hadoop2.7")
import pyspark
import psycopg2
import sys
from psycopg2 import sql
#spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
#           .config("spark.jars.packages", "databricks:spark-deep-learning:0.3.0-spark2.1-s_2.11") \
#           .getOrCreate()
#import sparkdl


import time
cwd = os.path.dirname(os.path.realpath(__file__))

def read_from_s3(bucket_name, folder_name, file_name):
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID', 'default')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', 'default')
    
    
    
    conn = boto.connect_s3(aws_access_key, aws_secret_access_key)
    bucket = conn.get_bucket(bucket_name)
    key = bucket.get_key(folder_name + file_name)
    data = key.get_contents_as_string()

    return data


def fetch_data(s3key, bucket):
    """
    Fetch data with the given s3 key and pass along the contents as a string.

    :param s3key: An s3 key path string.
            bucket:  bucket from boto.connect_s3
    :return: data:  wdata is the contents of the 
        file in a string. 
    """
    k = bucket.get_key(s3key)
    data = k.get_contents_as_string()
    
    return data


def bytes_to_np_array(message):
    tmp_np = np.frombuffer(message, dtype=np.uint8)
    image_np = cv2.imdecode(tmp_np, flags=1)
    image_np=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np

def detection_output(message):
    tl_cls = TLClassifier()
    image_np = bytes_to_np_array(message)
    box, conf, cls_idx  = b, conf, cls_idx = tl_cls.get_localization_classification(image_np, visual=False)
    return (box.tolist(), conf, cls_idx)


if __name__ == '__main__':
        
        
        os.chdir(cwd)
        
        bucket_name = "traffic-light-images"
        folder_name = ''
        file_name = "site_image_11.jpg"
        
        
        
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID', 'default')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', 'default')

        conn = boto.connect_s3(aws_access_key, aws_secret_access_key)

        bucket = conn.get_bucket(bucket_name)
        keys = bucket.list()
     
        file_list = [key.name for key in keys]
        num_files = len(file_list)
        
        
        
#        sc = pyspark.SparkContext()
#     
#        file_list_rdd = sc.parallelize(file_list)
#     
#        start = time.time()
#        boxes = file_list_rdd.map(lambda x: fetch_data(x, bucket)).map(lambda x: detection_output(x))
#        for idx, item in enumerate(boxes.take(num_files)):
#          
#            box = item[0] if len(item[0])>0 else []
#            box_list = np.array(box).tolist()
#          
#            val = [idx, file_list[idx], box_list]
#          
#            print(val)
#            
#        end = time.time()    
     
        jpg_in_text = read_from_s3(bucket_name, folder_name, file_name)
        jpg_in_np = np.frombuffer(jpg_in_text, dtype=np.uint8)
        img_np = cv2.imdecode(jpg_in_np, flags=1)
        img_np=cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        img_local = np.copy(img_np)
        
            
        start = time.time()
        
        tl_cls =TLClassifier()
        b, conf, cls_idx = tl_cls.get_localization_classification(img_local, visual=False)
            
        end = time.time()
        print('Localization time: ', end-start)
        print(cls_idx)
            
              
               
                  
            
           
       
          