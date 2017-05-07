#JUST EXAMPLES
import pyspark.ml.recommendation
df = spark.createDataFrame(
...     [(0, 0, 4.0), (0, 1, 2.0), (1, 1, 3.0), (1, 2, 4.0), (2, 1, 1.0), (2, 2, 5.0)],
...     ["user", "item", "rating"])

als = ALS(rank=10, maxIter=5, seed=0)

model = als.fit(df)
model.rank
#10
model.userFactors.orderBy("id").collect()
#[Row(id=0, features=[...]), Row(id=1, ...), Row(id=2, ...)]

test = spark.createDataFrame([(0, 2), (1, 0), (2, 0)], ["user", "item"])
predictions = sorted(model.transform(test).collect(), key=lambda r: r[0])
predictions[0]
#Row(user=0, item=2, prediction=-0.13807615637779236)
predictions[1]
#Row(user=1, item=0, prediction=2.6258413791656494)
predictions[2]
#Row(user=2, item=0, prediction=-1.5018409490585327)
als_path = temp_path + "/als"
als.save(als_path)
als2 = ALS.load(als_path)
als.getMaxIter()
#5
model_path = temp_path + "/als_model"
model.save(model_path)
model2 = ALSModel.load(model_path)
model.rank == model2.rank
#True
sorted(model.userFactors.collect()) == sorted(model2.userFactors.collect())
#True
sorted(model.itemFactors.collect()) == sorted(model2.itemFactors.collect())
#True

# ---------------------------------------
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines = spark.read.text("../zhang/proj/sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=long(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
# prediction is a dataframe DataFrame[movieId: bigint, rating: double, timestamp: bigint, userId: bigint, prediction: float]
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
