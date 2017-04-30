# 结果记录

## Kaggle 最高分是1.21251

## TODO

- [x] 实验随机森林，无`cross-validation`，无`真实的测试集`，对应第一波

- [x] 实现`cross-validation`和`真实的测试集`，对应第二波

- [x] 实现GBT，对应第三波

- [x] 在大数据集上跑上面的内容，对应第四波

- [ ] 写从`mongodb`读取数据的方法

- [ ] 合并所有方法

## 第四波（在大数据集上面跑）

1. `maxIter=5, maxDepth=2, seed=42`

```python
gbt = GBTRegressor(maxIter=5, maxDepth=2, seed=42)

RMSE = 1.14729976
```

2. `maxIter=50, maxDepth=6, seed=42`

```python
gbt = GBTRegressor(maxIter=50, maxDepth=6, seed=42)

RMSE = 1.05118689
```


## 第三波

运行 `traingbt.py`

1. `maxIter=5, maxDepth=2, seed=42`

```python
gbt = GBTRegressor(maxIter=5, maxDepth=2, seed=42)

RMSE = 1.03398484
```

2. `maxIter=50, maxDepth=6, seed=42`

```python
gbt = GBTRegressor(maxIter=50, maxDepth=6, seed=42)

RMSE = 0.96054627
```

## 第二波

运行 `trainrf-cv.py`

1. 第一次

    参数：

    ```python
    (data, test_data) = data.randomSplit([0.8, 0.2])
    num_cv = 5
    test_rate = 0.2
    model.append(RandomForest.trainRegressor(train, categoricalFeaturesInfo={},
                                             numTrees=10, featureSubsetStrategy="auto",
                                             impurity='variance', maxDepth=10, maxBins=32))
    ```

    结果：

    ```
    Round 0: Test Root Mean Squared Error = 1.105953
    Round 1: Test Root Mean Squared Error = 1.111686
    Round 2: Test Root Mean Squared Error = 1.115976
    Round 3: Test Root Mean Squared Error = 1.104856
    Round 4: Test Root Mean Squared Error = 1.102599
    Using model 4: 1.102599

    Final test Root Mean Squared Error = 1.106821
    ```

2. 第二次

    参数：

    ```python
    (data, test_data) = data.randomSplit([0.8, 0.2])
    num_cv = 5
    test_rate = 0.2
    model.append(RandomForest.trainRegressor(train, categoricalFeaturesInfo={},
                                             numTrees=30, featureSubsetStrategy="auto",
                                             impurity='variance', maxDepth=5, maxBins=32))
    ```

    结果：

    ```
    Round 0: Test Root Mean Squared Error = 1.084453
    Round 1: Test Root Mean Squared Error = 1.083273
    Round 2: Test Root Mean Squared Error = 1.084983
    Round 3: Test Root Mean Squared Error = 1.081591
    Round 4: Test Root Mean Squared Error = 1.075452
    Using model 4: 1.075452

    Final test Root Mean Squared Error = 1.079033
    ```

3. 第三次

    参数：

    ```python
    (data, test_data) = data.randomSplit([0.8, 0.2])
    num_cv = 5
    test_rate = 0.2
    model.append(RandomForest.trainRegressor(train, categoricalFeaturesInfo={},
                                             numTrees=30, featureSubsetStrategy="auto",
                                             impurity='variance', maxDepth=10, maxBins=32))
    ```

    结果：

    ```
    Round 0: Test Root Mean Squared Error = 1.104115
    Round 1: Test Root Mean Squared Error = 1.090919
    Round 2: Test Root Mean Squared Error = 1.097171
    Round 3: Test Root Mean Squared Error = 1.099824
    Round 4: Test Root Mean Squared Error = 1.102316
    Using model 1: 1.090919

    Final test Root Mean Squared Error = 1.097824
    ```

**总结：第二次的参数效果较好**

## 第一波 

下面的结果仅仅针对训练集，没有用测试集

1. model = 

    ```
    RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                numTrees=3, featureSubsetStrategy="auto",
                                impurity='variance', maxDepth=4, maxBins=32)

    Test Mean Squared Error = 1.16816262811
    ```

2. model = 

    ```
    RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                numTrees=30, featureSubsetStrategy="auto",
                                impurity='variance', maxDepth=5, maxBins=32)
                                      
    Test Mean Squared Error = 1.15278999013
    ```
    
3. model = 

    ```
    RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                numTrees=100, featureSubsetStrategy="auto",
                                impurity='variance', maxDepth=20, maxBins=32)
                                       
    Test Mean Squared Error = 1.15461927248
    ```

**总结：调参没用了，训不动，等第二波来**
