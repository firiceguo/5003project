import os
import sys
import numpy as np
import nltk


# Set the path for spark installation
# this is the path where you have built spark using sbt/sbt assembly
os.environ['SPARK_HOME'] = "/Applications/spark-2.1.0"
# os.environ['SPARK_HOME'] = "/home/jie/d2/spark-0.9.1"
# Append to PYTHONPATH so that pyspark could be found
sys.path.append("/Applications/spark-2.1.0/python")


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

except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)


sc = SparkContext()
spark = SparkSession \
    .builder \
    .appName("Yelp Review Usefulness Prediction & Baseline") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


"""
Load Review data and read it into a Dataframe named as reviewDF, select useful columns and save it as selectreviewDF
"""

reviewDF = spark.read.json('/Users/yanyunliu/Downloads/yelp_training_set/yelp_training_set_review.json')

reviewDF.printSchema()
selectreviewDF = reviewDF.select(reviewDF['review_id'],reviewDF['business_id'],reviewDF['user_id'],reviewDF['text'],reviewDF['votes.useful']) \
                    .withColumnRenamed('useful','label') \
                    .withColumnRenamed('text','review_text')

selectreviewDF = selectreviewDF.limit(100)

"""

Data Preprocessing:

1. Tokenize the text
2. Remove stopword
3. Convert Text into Vector
4. Calculate IDF
5. Load tf-idf features into LDA topic extraction model

"""

tokenizer = Tokenizer(inputCol="review_text", outputCol="tokens_word")
remover = StopWordsRemover(inputCol="tokens_word", outputCol="filtered_tokens_word")
cv = CountVectorizer(inputCol="filtered_tokens_word", outputCol="raw_features", minDF=2.0)
idf = IDF(inputCol="raw_features", outputCol="features")
lda = LDA(k=30, maxIter=10)

"""
Use RandomForestRegressor to predict usefulness
"""

rf = RandomForestRegressor(featuresCol="topicDistribution")

pipeline = Pipeline(stages=[tokenizer,remover, cv, idf, lda, rf])



(trainingData, testData) = selectreviewDF.randomSplit([0.7, 0.3])


evaluator_rmse = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")


paramGrid = ParamGridBuilder() \
    .addGrid(cv.vocabSize, [150, 200, 250]) \
    .build()

#    .addGrid(lda.k, [20, 30,50]) \
#
crossval = CrossValidator(estimator=pipeline, \
                          estimatorParamMaps=paramGrid,\
                          evaluator=evaluator_rmse,\
                          numFolds=4)  # use 3+ folds in practice

cvModel = crossval.fit(trainingData)
predictions = cvModel.transform(testData)


"""
Evaluation for LDA Prediction
"""
rmse = evaluator_rmse.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


"""
Evaluation for baseline model (Global Average)
"""

baseline_globalavg =  trainingData.select('label').agg({"label": "avg"}).collect()[0]['avg(label)']
testData = testData.select('*', lit(float(baseline_globalavg)).alias('baseline_prediction'))


evaluator_rmse_baseline = RegressionEvaluator(labelCol="label", predictionCol="baseline_prediction", metricName="rmse")
rmse = evaluator_rmse_baseline.evaluate(testData)
print("Root Mean Squared Error (RMSE) on test data for baseline = %g" % rmse)


print cvModel.explainParams()


sc.stop()

