#!/usr/bin/python2
# -*- coding: utf-8 -*-

from pyspark.sql.functions import *
from graphframes import *

"""
If you do it on the Jupyter Notebook, do the following config.
%%configure -f
{ "conf": {"spark.jars.packages": "graphframes:graphframes:0.3.0-spark2.0-s_2.11" }}

sc.addPyFile('wasb://5003@network5003.blob.core.windows.net/graphframes-0.3.0-spark2.0-s_2.11.jar')
"""


def network(sc, spark):
    # step1: create graph according to yelp network data
    v = spark.read.csv(
        'https://5003@network5003.blob.core.windows.net/yelpNetwork_i.csv', header=True, inferSchema=True)
    # v.count() 1029432
    e = spark.read.csv(
        'wasb://5003@network5003.blob.core.windows.net/yelpNetwork_e.csv', header=True, inferSchema=True)
    # e.count() 29723247
    g = GraphFrame(v, e)

    # step2: we need to make sure that this graph is a directed graph
    # then we can run pagerank algorithm on it
    a = g.inDegrees
    # b=g.outDegrees.withColumnRenamed('id','out_id')
    b = g.outDegrees
    # inOut=a.join(b,a['id']==b['out_id'])
    inOut = a.join(b, 'id')
    static = inOut.select(
        '*', (inOut['inDegree'] / inOut['outDegree']).alias('ratio')).select('id', 'ratio')
    bio_ratio = float(static.filter("ratio=1").count()) / \
        float(g.vertices.count())
    print bio_ratio

    # step3: detect connected component
    sc.setCheckpointDir(
        'wasb://5003@network5003.blob.core.windows.net/checkpoint')
    result = g.connectedComponents()
    r = result.select("id", "component")
    r.groupBy('component').count().orderBy('count', ascending=False).show()

    # step4: choose the largest connected component, create a new subset
    # graph, and run pagerank algorithm on this new graph
    subset_0 = result.filter('component=0')
    subset_id = subset_0.select('id')
    subset_edge = e.join(subset_id, e['dst'] == subset_0['id'], 'leftsemi').join(
        subset_id, e['src'] == subset_0['id'], 'leftsemi')

    g_cc = GraphFrame(subset_id, subset_edge)
    pr = g_cc.pageRank(resetProbability=0.01, maxIter=10)
    pr.vertices.select("id", "pagerank").orderBy(
        "pagerank", ascending=False).show()

    # step5: we want to get the max pagerank vertices for each business, so we
    # need (business_id,user_id) pair, extracted from review
    review = spark.read.csv(
        'wasb://5003@network5003.blob.core.windows.net/yelpNetwork_b_u.csv', header=True, inferSchema=True)

    # but if the number of one business's comment is too small, it will be meaningless for them to distribute coupons according
    # to this network's results, for they do not have enough data and do not have enough user to expand influence in cascanding.
    # so we first groupBy business id and extract subset of business whose users' number is more than 100
    # we consider these business is meaningful to use max pagerank user to express their coupons or make advertisement influence
    # on new dishes or event

    # in order to avoid spark bug on groupBy, we add withColumnRenamed before
    # every groupBy operation
    cnt = review.withColumnRenamed('business_id', 'business_id').groupBy(
        'business_id').count().filter('count>200')
    subset = cnt.join(review, 'business_id')
    # pr_results_business=pr.join(subset,pr['id']==subset['user_id']).select("user_id","pagerank","business_id") /
    #                    .withColumnRenamed('business_id','business_id').groupBy('business_id').max()

    pr_table = pr.vertices.select("id", "pagerank").orderBy(
        "pagerank", ascending=False)
    pr_results_business = pr_table.join(
        subset, pr_table['id'] == subset['user_id'])

    pr_results_business.select("user_id", "pagerank", "business_id").show()

    t1 = pr_results_business.select("user_id", "pagerank", "business_id").withColumnRenamed(
        'business_id', 'business_id').groupBy('business_id').max()
    t2 = t1.join(pr_table, t1['max(pagerank)'] == pr_table['pagerank']).withColumnRenamed(
        'id', 'user_id').select('business_id', 'user_id')
    t2.show()

    # step6: write result into csv file.
    # For default setting, spark will write it into multi-csvfile
    # distributely, we need to merge them into one csv file.
    import os
    from subprocess import call

    t2.write.format('com.databricks.spark.csv').save(
        'wasb://5003@network5003.blob.core.windows.net/result.csv')
    os.system("cat wasb://5003@network5003.blob.core.windows.net/result/p* > wasb://5003@network5003.blob.core.windows.net/result.csv")

    pr_table.write.format('com.databricks.spark.csv').save(
        'wasb://5003@network5003.blob.core.windows.net/pr.csv')
    os.system("cat wasb://5003@network5003.blob.core.windows.net/pr/p* > wasb://5003@network5003.blob.core.windows.net/pr.csv")

    # evaluation
    res = spark.read.csv(
        'wasb://5003@network5003.blob.core.windows.net/result.csv', header=True, inferSchema=True)
    cnt = 0
    lgt = 0
    for row in res.rdd.collect():
        id = row['user_id']
        print id
        con = "a.id='" + id + "'"
        con = str(con)
        print con
        top = g.find(
            "(a)-[]->(b);(b)-[]->(c)").filter(con).select("c.id").distinct().count()
        print top
        test = v.rdd.takeSample(False, 1, seed=cnt)
        for t in test:
            random = t['id']
            con1 = "a.id='" + random + "'"
            con1 = str(con1)
            random = g.find(
                "(a)-[]->(b);(b)-[]->(c)").filter(con1).select("c.id").distinct().count()
            print random
        if top > random:
            lgt = lgt + 1
        cnt = cnt + 1
    # ratio: 96.7%, means it's meaningful to use this system to recommend
    # users for business
