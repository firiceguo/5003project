#!/usr/bin/python2
# -*- coding: utf-8 -*-

from pyspark.ml.linalg import Vectors, SparseVector, DenseVector
from collections import defaultdict
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressor
import os
import json
import math
import cPickle as pickle


def tuple2sparse(tp, size=43, begin=19, end=42):
    dic = {}
    for i in xrange(end-begin):
        if (tp[i] - 0) > 10e-4:
            dic[i+begin] = tp[i]
    v = Vectors.sparse(size, dic)
    return v


def add(v1, v2):
    assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector)
    assert v1.size == v2.size
    values = defaultdict(float) # Dictionary with default value 0.0
    # Add values from v1
    for i in range(v1.indices.size):
        values[v1.indices[i]] += v1.values[i]
    # Add values from v2
    for i in range(v2.indices.size):
        values[v2.indices[i]] += v2.values[i]
    return Vectors.sparse(v1.size, dict(values))


def loadDataJson(business_path='', user_path='', star_path=''):
    bDF = spark.read.json(business_path)
    uDF = spark.read.json(user_path)

    businessDF = bDF.rdd.map(lambda x: (x['b_id'], tuple2sparse(
                             tuple(x['loc']) + tuple(x['votes']) + (x['avg_star'], ) +
                             tuple(x['cates']) + (x['rev_num'], ) + tuple(x['ckins']),
                             begin=19, end=42)))\
                    .toDF(['b_id', 'b_features'])

    userDF = uDF.rdd.map(lambda x: (x['u_id'], tuple2sparse(
                         tuple(x['loc']) + tuple(x['votes']) +
                         (x['loc_num'], x['avg_star'], x['rev_num']) + tuple(x['cates']),
                         begin=0, end=19)))\
                .toDF(['u_id', 'u_features'])

    stars = pickle.load(open(star_path, 'rb'))
    starDF = sc.parallelize([(list(k)[0], list(k)[1]) + (v['stars'], v['rev_id']) for k, v in stars.items()]) \
               .toDF(['b_id', 'u_id', 'label', 'rev_id'])

    return businessDF, userDF, starDF


def loadDataMongo():
    return businessDF, userDF, starDF


def transData4GBT(businessDF, userDF, starDF):
    alldata = starDF.select(starDF.b_id, starDF.u_id, starDF.stars) \
                    .join(businessDF, starDF.b_id == businessDF.b_id).drop(businessDF.b_id) \
                    .join(userDF, starDF.u_id == userDF.u_id).drop(userDF.u_id)
    data = alldata.select('label', 'features')
    return data


def traingbt(datafrom='json', business_path='', user_path='', star_path=''):
    gbt = GBTRegressor(maxIter=5, maxDepth=2, seed=42)
    if datafrom == 'json':
        businessDF, userDF, starDF = loadDataJson(business_path=business_path,
                                                  user_path=user_path,
                                                  star_path=star_path)
    elif datafrom == 'mongodb':
        businessDF, userDF, starDF = loadDataMongo()
    data = transData4GBT(businessDF, userDF, starDF)
    model = gbt.fit(data)
    return model


if __name__ == '__main__':
    now_path = os.getcwd() + '/'
    data_path = now_path + '../dataset/'
    business_path = data_path + 'businesses.json'
    user_path = data_path + 'users.json'
    star_path = data_path + 'stars.pk'
    sc = SparkContext()
    spark = SparkSession(sc)

    gbt = GBTRegressor(maxIter=50, maxDepth=6, seed=42)
    businessDF, userDF, starDF = loadDataJson(business_path=business_path,
                                              user_path=user_path,
                                              star_path=star_path)
    # split starDF to training data and test data
    trainStarDF, testStarDF = starDF.randomSplit([0.7, 0.3])

    trainDF = transData4GBT(businessDF, userDF, trainStarDF)
    model = gbt.fit(trainDF)

    testDF = transData4GBT(businessDF, userDF, testStarDF)
    prediction = model.transform(testDF)

    errors = prediction.rdd.map(lambda x: (x.label - x.prediction)**2).collect()
    RMSE = math.sqrt(sum(errors)/len(errors))
    print 'RMSE: %.8f' % RMSE
