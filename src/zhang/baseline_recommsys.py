#!/usr/bin/python2
# -*- coding: utf-8 -*-

from pyspark.sql.functions import col
import json
import math
import cPickle as pickle
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
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


# ratings is a df, ['usr_id','bus_id','label','prerating']
def covtDataFormat(ratings):

    usr_list = ratings.select('usr_id').distinct().rdd.map(lambda x : x.usr_id).collect()
    bus_list = ratings.select('bus_id').distinct().rdd.map(lambda x : x.bus_id).collect()
    
    usr = {}
    bus = {}
    i = 0
    for ele in usr_list:
        usr[ele] = i
        i = i+1
      
    j = 0
    for ele in bus_list:
        bus[ele] = j
        j = j+1

    usrnoDF = sc.parallelize([k, v] for k, v in usr.items()) \
                .toDF(['usr_id', 'usr_no'])
    #usrnoDF.show()

    busnoDF = sc.parallelize([k, v] for k, v in bus.items()) \
                .toDF(['bus_id', 'bus_no'])
    #busnoDF.show()
    
    recomDF = ratings.join(usrnoDF, ratings.usr_id == usrnoDF.usr_id).drop(usrnoDF.usr_id) \
                   .join(busnoDF, ratings.bus_id == busnoDF.bus_id).drop(busnoDF.bus_id)
    recomDF.show()
    recomDF = recomDF.select('usr_no','bus_no','label','prerating')
    return recomDF

  
def recommsys(training,test):
    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=5, regParam=0.01, userCol="usr_no", itemCol="bus_no", ratingCol="label")
    # data = df.union(df)
    model = als.fit(training)
    # Evaluate the model by computing the RMSE on the test data
    # prediction is a dataframe DataFrame[movieId: bigint, rating: double, timestamp: bigint, userId: bigint, prediction: float]
    predictions = model.transform(test).na.drop()
    #print predictions
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="label",predictionCol="prediction")
    rmse= evaluator.evaluate(predictions)
    
    return rmse


if __name__ == '__main__':

    #initial some para
    test_path = './yelp_academic_dataset_review.json'
    usr_path = './yelp_academic_dataset_user.json'
    bus_path = './yelp_academic_dataset_business.json'
    #json_path = './ratings.json'
    training_fraction= 0.7
    reserve_fraction = 0.2
    test_fraction = 0.1

    #calulation of baselines's rmse
    testDF,usrDF,busDF = loadDataJson(test_path=test_path, usr_path=usr_path, bus_path=bus_path)
    baseDF, RMSE = calculateRMSE(testDF,usrDF,busDF)
    print 'baselin RMSE: %.8f' % RMSE 


    #recommendation
    #using only single part of data
    ratings = baseDF.select('usr_id','bus_id','label','baserating').withColumnRenamed('baserating', 'prerating')
    recomDF = covtDataFormat(ratings)

    training, reserve, test = recomDF.randomSplit([training_fraction,reserve_fraction, test_fraction])

    rmse = recommsys(training,test)
    print("Root-mean-square error originally = " + str(rmse))

    #also using prediction data
    reserve = reserve.drop('label').withColumnRenamed('prerating','label')
    training= training.drop('prerating')
    training=training.union(reserve)

    rmse2 = recommsys(training,test)
    print("Root-mean-square error with rating prediction = " + str(rmse2))
