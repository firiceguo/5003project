#!/usr/bin/python2
# -*- coding: utf-8 -*-

from pyspark.sql.functions import col
import json
import math
from pyspark.sql import Row


def loadDataJson(test_path='', usr_path='', bus_path=''):
    # load test df
    testDF = spark.read.json(test_path)
    testDF = testDF.select(col('business_id').alias('bus_id'), col('user_id').alias('usr_id'),col('stars').alias('label'))

    # load usr df
    usrDF = spark.read.json(usr_path)
    usrDF = usrDF.select(col('user_id').alias('usr_id'), col('review_count').alias('urew_no'),col('average_stars').alias('uavg_stars'))
    usrDF = usrDF.select('usr_id','urew_no',(col('urew_no')*col('uavg_stars')).alias('usrtemp'))

    # load bus df
    busDF = spark.read.json(bus_path)
    busDF = busDF.select(col('business_id').alias('bus_id'),col('stars').alias('bavg_stars'),col('review_count').alias('brew_no'))
    busDF = busDF.select('bus_id','brew_no',(col('brew_no')*col('bavg_stars')).alias('bustemp'))
    #busDF.show()
    return testDF,usrDF,busDF


def calculateRMSE(testDF,usrDF,busDF):
    # real calculation
    baseDF = testDF.join(usrDF, testDF.usr_id == usrDF.usr_id).drop(usrDF.usr_id) \
               .join(busDF, testDF.bus_id == busDF.bus_id).drop(busDF.bus_id)

    baseDF = baseDF.select('*',((col('usrtemp')+col('bustemp'))/(col('urew_no')+col('brew_no'))).alias('baserating'))
    #baseDF.show()
    rmseDF = baseDF.select('label','baserating',((col('label') - col('baserating'))**2).alias('mes'))
    errors = rmseDF.rdd.map(lambda x: x.mes).collect()
    RMSE = math.sqrt(sum(errors)/len(errors))
    return baseDF,RMSE


if __name__ == '__main__':

    #initial some path
    test_path = './yelp_academic_dataset_review.json'
    usr_path = './yelp_academic_dataset_user.json'
    bus_path = './yelp_academic_dataset_business.json'
        #json_path = './ratings.json'
    training_fraction= 0.7
    test_fraction = 0.3

       
    testDF,usrDF,busDF = loadDataJson(test_path=test_path, usr_path=usr_path, bus_path=bus_path)
    baseDF, RMSE = calculateRMSE(testDF,usrDF,busDF)
    print 'Baseline RMSE: %.8f' % RMSE 
