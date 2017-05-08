import json
test_path = './yelp_academic_dataset_review.json'
testDF = spark.read.json(test_path)

from pyspark.sql.functions import col
ratings= testDF.select(col('business_id').alias('bus_id'), col('user_id').alias('usr_id'),col('stars').alias('label'))
print ratings

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
    #recomDF.show()
    recomDF = recomDF.select('usr_no','bus_no','label')
    return recomDF

recomDF = covtDataFormat(ratings)
print recomDF

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
from pyspark.sql import functions as F
def GetRecomList(predictions, partition_by, order_by, rank_num):

    #predictions is a dataframe which contain id and predict result(score).
    #partition_by is a name of column in predictions which identify the key to partition by
    # order_by is a name of column in predictions which identify the score we need to sort
    # rank_num specify how many records you want to return for each partition


    window = Window \
        .partitionBy(predictions[partition_by])\
        .orderBy(predictions[order_by].desc())


    recomlistdDF = predictions.select(partition_by,order_by, rank().over(window).alias('rank')) \
                              .filter(col('rank') <= rank_num)

    print "Get num of review list: ",recomlistdDF.count()

    return recomlistdDF
def recommsys(training,test,r,it,para):
    # Build the recommendation model using ALS on the training data
    als = ALS(rank =r, maxIter=it, regParam=para, userCol="usr_no", itemCol="bus_no", ratingCol="label")
    # data = df.union(df)
    model = als.fit(training)
    # Evaluate the model by computing the RMSE on the test data
    # prediction is a dataframe DataFrame[movieId: bigint, rating: double, timestamp: bigint, userId: bigint, prediction: float]
    predictions = model.transform(test).na.drop()
    #print predictions

    predictions =  predictions.select('*',F.when(predictions.prediction > 5, 5) \
                                      .otherwise(predictions.prediction).alias('temp'))
    predictions =  predictions.select('*', F.when(predictions.temp < 1, 1) \
                                      .otherwise(predictions.temp).alias('newrating'))
    predictions.drop('prediction','temp')
    
    predictions.filter(predictions.newrating>3.5).show()
    #print "Find the business with the highest prediction rating for a user:"
    #recolistDF = GetRecomList(predictions, 'usr_no','newrating', 1)
    #recolistDF.show()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="label",predictionCol="newrating")
    rmse= evaluator.evaluate(predictions)
    return rmse

training_fraction= 0.7
test_fraction = 0.3
training,test = recomDF.randomSplit([training_fraction,test_fraction])
print training,test

rmse = recommsys(training,test,10,5,0.01)
print("Root-mean-square error by = " + str(rmse))
