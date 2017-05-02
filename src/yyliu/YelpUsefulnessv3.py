import os
import sys
import numpy as np
import nltk
import time


# Set the path for spark installation
# this is the path where you have built spark using sbt/sbt assembly
os.environ['SPARK_HOME'] = "/Applications/spark-2.1.0"
# os.environ['SPARK_HOME'] = "/home/jie/d2/spark-0.9.1"
# Append to PYTHONPATH so that pyspark could be found
sys.path.append("/Applications/spark-2.1.0/python")

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages org.mongodb.spark:mongo-spark-connector_2.10:2.0.0 --driver-memory 5g  --executor-memory 5g pyspark-shell")



# Now we are ready to import Spark Modules
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.ml import Pipeline
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import *
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.sql import Row
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.sql import SparkSession
    from pyspark.mllib.linalg import SparseVector, DenseVector
    from pyspark.ml.feature import CountVectorizer
    from pyspark.ml.clustering import LDA
    from pyspark.ml.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.ml.feature import Tokenizer, RegexTokenizer
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import IntegerType
    from pyspark.ml.feature import StopWordsRemover
    from pyspark.ml.feature import HashingTF, IDF, Tokenizer
    from pyspark.ml import Pipeline
    from pyspark.ml.regression import RandomForestRegressor
    from pyspark.ml.feature import VectorIndexer
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType, FloatType
    from pyspark.sql.window import Window
    from pyspark.sql.functions import rank, col

except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)

sc = SparkContext()
sc.addPyFile("/Users/yanyunliu/Downloads/mongo-spark-connector_2.10-2.0.0.jar")

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/users") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/users") \
    .getOrCreate()

"""
Load Review data and read it into a Dataframe named as reviewDF, select useful columns and save it as selectreviewDF
"""



def loadReviewDataJson(datafrom='', review_path=''):

    if datafrom == 'json':
        reviewDF = spark.read.json(review_path)

        print '*'*100
        print "This is the schema in original json review file"
        print '*'* 100

        reviewDF.printSchema()
        selectreviewDF = reviewDF.select(reviewDF['review_id'], reviewDF['business_id'], reviewDF['user_id'],
                                         reviewDF['text'], reviewDF['useful']) \
            .withColumnRenamed('useful', 'label') \
            .withColumnRenamed('text', 'review_text')


        print '*'*100
        print "This is the schema for extracted data"
        print '*'* 100

        selectreviewDF = selectreviewDF.limit(100)
        selectreviewDF.printSchema()

    elif datafrom == 'mongodb':

        reviewDF = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri",
        "mongodb://127.0.0.1/users.review").load()

        print '*'*100
        print "This is the schema in original mongoDB review collection"
        print '*'* 100

        reviewDF.printSchema()

        selectreviewDF = reviewDF.select(reviewDF['review_id'], reviewDF['business_id'], reviewDF['user_id'],
                                         reviewDF['text'], reviewDF['useful']) \
                                .withColumnRenamed('useful', 'label') \
                                .withColumnRenamed('text', 'review_text')


        print '*'*100
        print "This is the schema for extracted data"
        print '*'* 100

        selectreviewDF = selectreviewDF.limit(100)
        selectreviewDF.printSchema()


    return selectreviewDF




def UsefulnessPrediction(trainingdata,model):

    #Data Preprocessing
    tokenizer = Tokenizer(inputCol="review_text", outputCol="tokens_word")

    remover = StopWordsRemover(inputCol="tokens_word", outputCol="filtered_tokens_word")

    cv = CountVectorizer(inputCol="filtered_tokens_word", outputCol="raw_features", minDF=2.0)
    idf = IDF(inputCol="raw_features", outputCol="features")

    #Extract LDA topic feature
    lda = LDA(k=30, maxIter=10)

    if model == 'RandomForest':
        model = RandomForestRegressor(featuresCol="topicDistribution")

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lda, model])


    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")


    paramGrid = ParamGridBuilder() \
        .addGrid(cv.vocabSize, [150, 200, 250]) \
        .build()

    #    .addGrid(lda.k, [20, 30,50]) \
    #
    crossval = CrossValidator(estimator=pipeline, \
                              estimatorParamMaps=paramGrid, \
                              evaluator=evaluator_rmse, \
                              numFolds=4)  # use 3+ folds in practice

    cvModel = crossval.fit(trainingData)


    #Explain params for the selected model
    print cvModel.explainParams()

    return cvModel


def GetPredictionError(cvModel,testData,evaluation_method, col_name):


    predictions = cvModel.transform(testData)

    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol= col_name, metricName=evaluation_method) #"rmse"

    rmse = evaluator_rmse.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    return predictions, rmse


def GetRecomReviewList(predictions):

    window = Window \
        .partitionBy(predictions['business_id'])\
        .orderBy(predictions['prediction'].desc())

    recomlistdDF = predictions.select('business_id','prediction', rank().over(window).alias('rank')) \
                              .filter(col('rank') <= 5)

    recomlistdDF.show()

    return recomlistdDF


def GetBaselineModelError(trainingData, testData,evaluation_method, col_name):

    baseline_globalavg = trainingData.select('label').agg({"label": "avg"}).collect()[0]['avg(label)']
    testData = testData.select('*', lit(float(baseline_globalavg)).alias(col_name))

    print "The global average for usefulness on training data is", baseline_globalavg

    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol= col_name, metricName=evaluation_method) #"rmse"

    baseline_rmse = evaluator_rmse.evaluate(testData)

    print "Root Mean Squared Error (RMSE) for baseline model is", baseline_rmse

    return testData, baseline_globalavg


if __name__ == '__main__':


    reviewpath = '/Users/yanyunliu/Downloads/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json'


    print "Test efficiency: Loading data from mongoDB vs loading data from json file: "
    t1 = time.time()
    selectreviewDF = loadReviewDataJson('json', review_path=reviewpath)
    t2 = time.time()

    print str(t2-t1), "This is the time cost for loading data from json file."
    #Output: 10.2610740662 This is the time cost for loading data from json file.

    t1 = time.time()
    selectreviewDF = loadReviewDataJson('mongodb', review_path=reviewpath)
    t2 = time.time()

    print str(t2-t1),"This is the time cost for loading data from MongoDB."
    #Output: 2.66875004768 This is the time cost for loading data from MongoDB.



    (trainingData, testData) = selectreviewDF.randomSplit([0.7, 0.3])

    testData, baseline_globalavg = GetBaselineModelError(trainingData, testData, "rmse","baseline")


    cvModel = UsefulnessPrediction(trainingData,'RandomForest')
    predictions, rmse = GetPredictionError(cvModel,testData,"rmse","prediction")

    GetRecomReviewList(predictions)



sc.stop()

