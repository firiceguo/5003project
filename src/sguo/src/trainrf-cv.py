#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import math
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils

now_path = os.getcwd() + '/'
train_path = now_path + 'dataset/libsvm_train.txt'
# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, train_path)
(data, test_data) = data.randomSplit([0.9, 0.1])
num_cv = 5
test_rate = 0.2
block = []
model = []
RMSE = []

# Cross Validation
for i in xrange(num_cv):
    block = [1/float(num_cv)*test_rate, 1/float(num_cv)*(1-test_rate), 1-1/float(num_cv)*(i+1)]
    (train, test, data) = data.randomSplit(block)
    model.append(RandomForest.trainRegressor(train, categoricalFeaturesInfo={},
                                             numTrees=10, featureSubsetStrategy="auto",
                                             impurity='variance', maxDepth=10, maxBins=32))
    predictions = model[len(model)-1].predict(test.map(lambda x: x.features))
    labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
    testRMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(test.count())
    RMSE.append(testRMSE)
    print 'Round %d: Test Root Mean Squared Error = %f' % (i, math.sqrt(testRMSE))

minn = min(RMSE)
index = RMSE.index(minn)
print 'Using model %d: %f' % (index, math.sqrt(minn))

# Final test
predictions = model[i].predict(test_data.map(lambda x: x.features))
labelsAndPredictions = test_data.map(lambda lp: lp.label).zip(predictions)
testRMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(test_data.count())
print 'Final test Root Mean Squared Error = %f' % math.sqrt(testRMSE)
