#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import sys
import math
import time

# Set the path for spark installation
# this is the path where you have built spark using sbt/sbt assembly
spark_home = '/SPARK_HOME/'
os.environ['SPARK_HOME'] = spark_home
mongouri = "mongodb://127.0.0.1/users"
mongo_spark_connector_path = '/MONGO_SPARK/mongo-spark-connector_2.10-2.0.0.jar'

# Append to PYTHONPATH so that pyspark could be found
sys.path.append(spark_home + "/python")

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    # "--packages org.mongodb.spark:mongo-spark-connector_2.10:2.0.0 "
    "--driver-memory 5g  --executor-memory 5g pyspark-shell "
    "--packages graphframes:graphframes:0.3.0-spark2.0-s_2.11")

from pyspark import SparkContext
sc = SparkContext()
# sc.addPyFile(mongo_spark_connector_path)

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.mongodb.input.uri", mongouri) \
    .config("spark.mongodb.output.uri", mongouri) \
    .getOrCreate()


from pyspark.ml.regression import GBTRegressor
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col

from baseline import baseline, recommedation
from gbt import traingbt, dataprocessing
from nlp import sentiment_based_model, topic_based_model


def loadData(datafrom, name='', mongouri='', jsonpath='', Datalimit=False, DatalimitNum=0):
    """
    datafrom:     specify where the data come from ('mongodb' or 'json')

    If datafrom='mongodb', please set the following options:
        name:     specify the name of collection in mongodb.
        mongouri: mongouri, e.g. 'mongodb://127.0.0.1/users'

    If datafrom='json', please set the following options:
        path:     sepcify your path of json file.

    Datalimit:    specify if you want to use limit data. and DatalimitNum
    DatalimitNum: specify the number of data you want to use.
    """
    if datafrom == 'json':
        assert jsonpath, 'Please set the json file path.'
        DF = spark.read.json(jsonpath)

    elif datafrom == 'mongodb':
        assert name, 'Please set the mongo databse name.'
        assert mongouri, 'Please set the mongo uri.'
        DF = spark.read.format("com.mongodb.spark.sql.DefaultSource"). \
            option("uri", mongouri + '.' + name).load()

    print '*' * 100
    print "This is the schema in original %s collection" % datafrom
    print '*' * 100

    DF.printSchema()
    if Datalimit == True:
        # Use limited dataset: enable the limit and cache()
        DF = DF.limit(DatalimitNum)
    return DF


def GetRecomList(predictions, partition_by, order_by, rank_num):
    """
    predictions:  a dataframe which contain id and predict result(score).
    partition_by: a name of column in predictions which identify the key to partition by
    order_by:     a name of column in predictions which identify the score we need to sort
    rank_num:     specify how many records you want to return for each partition
    """
    window = Window \
        .partitionBy(predictions[partition_by]) \
        .orderBy(predictions[order_by].desc())

    recomlistdDF = predictions.select(partition_by, order_by, rank().over(window).alias('rank')) \
        .filter(col('rank') <= rank_num)

    print "Get num of review list: ", recomlistdDF.count()

    return recomlistdDF


def runBaseline(sc, ori_revDF='', ori_usrDF='', ori_busDF='', fraction=[0.7, 0.3]):
    print 'Baseline:'
    print '  Rating Prediction:'
    bl_testDF, bl_usrDF, bl_busDF = baseline.dataClean(testDF=ori_revDF,
                                                       usrDF=ori_usrDF,
                                                       busDF=ori_busDF)
    baseDF, RMSE = baseline.calculateRMSE(bl_testDF, bl_usrDF, bl_busDF)
    print '    Baseline RMSE: %.8f' % RMSE

    print '  Recommendation:'
    ratingsDF = ori_revDF.select(col('business_id').alias('bus_id'),
                                 col('user_id').alias('usr_id'),
                                 col('stars').alias('label'))

    recomDF = recommedation.covtDataFormat(ratingsDF, sc)
    training, test = recomDF.randomSplit(fraction)

    rmse = recommedation.recommsys(training, test, 10, 5, 0.01)
    print "    RMSE by = " + str(rmse)


def runGBT(busDF='', usrDF='', revDF='', fraction=[0.7, 0.3]):
    print 'GBTRegressor:'
    print '  Rating Prediction:'
    gbt = GBTRegressor(maxIter=1, maxDepth=1, seed=42)
    businessDF, userDF, starDF = traingbt.dataClean(busDF=busDF,
                                                    usrDF=usrDF,
                                                    revDF=revDF)
    # split starDF to training data and test data
    trainStarDF, testStarDF = starDF.randomSplit(fraction)

    trainDF = traingbt.transData4GBT(businessDF, userDF, trainStarDF)

    model = gbt.fit(trainDF)

    testDF = traingbt.transData4GBT(businessDF, userDF, testStarDF)
    predDF = model.transform(testDF)

    predDF.show()
    errors = predDF.rdd.map(lambda x: (x.label - x.prediction)**2).collect()
    RMSE = math.sqrt(sum(errors) / len(errors))
    print '    GBTRegressor RMSE: %.8f' % RMSE

    print '  Recommendation:'
    # recDF = traingbt.recommendation(businessDF, testStarDF, testDF, model)
    # recDF.printSchema()
    print '    Recommendation RMSE: %.8f' % RMSE


def runNetwork(sc, spark):
    sc.addPyFile(os.getcwd() + '/graphframes-0.3.0-spark2.0-s_2.11.jar')
    from network import network
    print 'Network:'
    network.network(sc, spark)


def runSentiment(revDF=''):
    print 'Sentiment based model:'
    t1 = time.time()
    selectreviewDF = revDF.select(revDF['review_id'], revDF['business_id'], revDF['user_id'],
                                  revDF['text'], revDF['useful']) \
        .withColumnRenamed('useful', 'label') \
        .withColumnRenamed('text', 'review_text')

    print selectreviewDF.count()
    selectreviewDF.cache()

    selectreviewDF = sentiment_based_model.SentimentFeatureEngineer(
        selectreviewDF)
    selectreviewDF.cache()

    (trainingData, testData) = selectreviewDF.randomSplit([0.7, 0.3])
    testData, baseline_rmse = sentiment_based_model.GetBaselineModelError(
        trainingData, testData, 'rmse', 'baseline_prediction')
    nMode = sentiment_based_model.UsefulnessPredictionSentmentWithoutCV(
        trainingData, 'RandomForest')

    predictions, rmse = sentiment_based_model.GetPredictionError(
        nMode, testData, 'rmse', 'prediction')

    recomlistdDF = GetRecomList(predictions, 'business_id', 'prediction', 1)
    recomlistdDF.show()
    t2 = time.time()
    print "Time Cost Totally: ", str(t2 - t1), \
          "Minutes: ", str((t2 - t1) / 60),  \
          "Seconds: ", str((t2 - t1) % 60)


def runTopic(revDF=''):
    print 'Topic based model:'
    t1 = time.time()
    selectreviewDF = revDF.select(revDF['review_id'], revDF['business_id'], revDF['user_id'],
                                  revDF['text'], revDF['useful']) \
        .withColumnRenamed('useful', 'label') \
        .withColumnRenamed('text', 'review_text')
    selectreviewDF.cache()
    t2 = time.time()

    print str(t2 - t1), "This is the time cost for loading data from MongoDB."
    print 'num of rows in Dataframe selectreviewDF:', "*" * 100
    print selectreviewDF.count()

    (trainingData, testData) = selectreviewDF.randomSplit([0.7, 0.3], seed=111)

    # trainingData.show()
    print 'num of rows in Dataframe trainingData:', "*" * 100
    print trainingData.count()
    print 'num of rows in Dataframe testData:', "*" * 100
    # testData.show()
    print testData.count()

    testData, baseline_rmse = topic_based_model.GetBaselineModelError(
        trainingData, testData, "rmse", "baseline_prediction")
    nModel = topic_based_model.UsefulnessPredictionLDAWithoutCV(
        trainingData, 'RandomForest')
    predictions, rmse = topic_based_model.GetPredictionError(
        nModel, testData, "rmse", "prediction")
    print 'num of rows in Dataframe predictions:', "*" * 100
    print predictions.count()

    recomlistdDF = GetRecomList(predictions, 'business_id', 'prediction', 1)
    recomlistdDF.show()


if __name__ == '__main__':

    # If already have gbt data, don't do the following
    data_path = os.getcwd() + '/yelp/'
    user_path = data_path + 'user.json'
    review_path = data_path + 'review.json'
    checkin_path = data_path + 'checkin.json'
    business_path = data_path + 'business.json'

    dataprocessing.genJsonDataset(
        [user_path, review_path, checkin_path, business_path])

    ori_revDF = loadData('mongodb', name='review', mongouri=mongouri)
    ori_usrDF = loadData('mongodb', name='user', mongouri=mongouri)
    ori_busDF = loadData('mongodb', name='business', mongouri=mongouri)
    gbt_busDF = loadData('mongodb', name='new_business', mongouri=mongouri)
    gbt_usrDF = loadData('mongodb', name='new_usr', mongouri=mongouri)

    # ori_revDF = loadData('json', jsonpath=data_path + 'review.json')
    # ori_usrDF = loadData('json', jsonpath=data_path + 'user.json')
    # ori_busDF = loadData('json', jsonpath=data_path + 'business.json')
    # gbt_busDF = loadData('json', jsonpath=data_path +
    #                      'big_dataset/gbt_businesses.json')
    # gbt_usrDF = loadData('json', jsonpath=data_path +
    #                      'big_dataset/gbt_users.json')

    training_fraction = 0.7
    test_fraction = 0.3

    runBaseline(sc, ori_revDF=ori_revDF,
                ori_busDF=ori_busDF,
                ori_usrDF=ori_usrDF,
                fraction=[training_fraction, test_fraction])

    runGBT(busDF=gbt_busDF,
           usrDF=gbt_usrDF,
           revDF=ori_revDF,
           fraction=[training_fraction, test_fraction])

    runSentiment(revDF=ori_revDF)

    runTopic(revDF=ori_revDF)

    # runNetwork(sc, spark)
