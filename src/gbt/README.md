# Files and Functions

- `dataprocessing.py`: For data cleaning.

- `traingbt.py`: For doing the rating prediction and recommendation.

# Results

## Kaggle: 1.21251

## Fourth (Big dataset)

|maxIter|maxDepth|RMSE|
|:---------:|:----------:|:---------:|
|5|2|1.14729976|
|10|2|1.12052114|
|10|4|1.06971257|
|10|6|1.05618326|
|10|8|1.05188215|
|10|10|1.05102746|
|10|12|1.05331973|
|10|14|1.06188889|
|50|6|1.05118689|
|50|10|1.04932254|


## Third (Small dataset)

`python traingbt.py`

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

## Second

`python trainrf-cv.py`

1. parameters

    ```python
    (data, test_data) = data.randomSplit([0.8, 0.2])
    num_cv = 5
    test_rate = 0.2
    model.append(RandomForest.trainRegressor(train, categoricalFeaturesInfo={},
                                             numTrees=10, featureSubsetStrategy="auto",
                                             impurity='variance', maxDepth=10, maxBins=32))
    ```

    result：

    ```
    Round 0: Test Root Mean Squared Error = 1.105953
    Round 1: Test Root Mean Squared Error = 1.111686
    Round 2: Test Root Mean Squared Error = 1.115976
    Round 3: Test Root Mean Squared Error = 1.104856
    Round 4: Test Root Mean Squared Error = 1.102599
    Using model 4: 1.102599

    Final test Root Mean Squared Error = 1.106821
    ```

2. parameters

    ```python
    (data, test_data) = data.randomSplit([0.8, 0.2])
    num_cv = 5
    test_rate = 0.2
    model.append(RandomForest.trainRegressor(train, categoricalFeaturesInfo={},
                                             numTrees=30, featureSubsetStrategy="auto",
                                             impurity='variance', maxDepth=5, maxBins=32))
    ```

    result：

    ```
    Round 0: Test Root Mean Squared Error = 1.084453
    Round 1: Test Root Mean Squared Error = 1.083273
    Round 2: Test Root Mean Squared Error = 1.084983
    Round 3: Test Root Mean Squared Error = 1.081591
    Round 4: Test Root Mean Squared Error = 1.075452
    Using model 4: 1.075452

    Final test Root Mean Squared Error = 1.079033
    ```

3. parameters：

    ```python
    (data, test_data) = data.randomSplit([0.8, 0.2])
    num_cv = 5
    test_rate = 0.2
    model.append(RandomForest.trainRegressor(train, categoricalFeaturesInfo={},
                                             numTrees=30, featureSubsetStrategy="auto",
                                             impurity='variance', maxDepth=10, maxBins=32))
    ```

    result：

    ```
    Round 0: Test Root Mean Squared Error = 1.104115
    Round 1: Test Root Mean Squared Error = 1.090919
    Round 2: Test Root Mean Squared Error = 1.097171
    Round 3: Test Root Mean Squared Error = 1.099824
    Round 4: Test Root Mean Squared Error = 1.102316
    Using model 1: 1.090919

    Final test Root Mean Squared Error = 1.097824
    ```

## First

no testset

1. model = 

    ```python
    RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                numTrees=3, featureSubsetStrategy="auto",
                                impurity='variance', maxDepth=4, maxBins=32)

    Test Mean Squared Error = 1.16816262811
    ```

2. model = 

    ```python
    RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                numTrees=30, featureSubsetStrategy="auto",
                                impurity='variance', maxDepth=5, maxBins=32)
                                      
    Test Mean Squared Error = 1.15278999013
    ```
    
3. model = 

    ```python
    RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                numTrees=100, featureSubsetStrategy="auto",
                                impurity='variance', maxDepth=20, maxBins=32)
                                       
    Test Mean Squared Error = 1.15461927248
    ```
