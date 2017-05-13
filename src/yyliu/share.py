import os
import sys
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
    from pyspark.sql import SparkSession
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

def loadDataJson(name, datafrom='', path='', Datalimit = False, DatalimitNum = 0):

    #name specify the name of collection in mongodb or type of json file.
    #datafrom specify where the data come from ('mongodb' or 'json')
    #Datalimit specify if you want to use limit data. and DatalimitNum specify the number of data you want to use.

    if datafrom == 'json':

        DF = spark.read.json(path)

        print '*'*100
        print "This is the schema in original json review file"
        print '*'* 100

        DF.printSchema()

        if Datalimit == True:
            # Use limited dataset: enable the limit and cache()
            DF = DF.limit(DatalimitNum)


    elif datafrom == 'mongodb':

        DF = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri",
        "mongodb://127.0.0.1/users."+name).load()

        print '*'*100
        print "This is the schema in original mongoDB review collection"
        print '*'* 100

        DF.printSchema()

        if Datalimit == True:
            # Use limited dataset: enable the limit and cache()
            DF = DF.limit(DatalimitNum)


    return DF

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

if __name__ == '__main__':
    selectreviewDF = loadDataJson('review', datafrom='mongodb', Datalimit=True, DatalimitNum=1000)