# 结果记录

## Kaggle 最高分是1.21251

## TODO

- [x] 实验随机森林，无`cross-validation`，无`真实的测试集`，对应第一波

- []  实现`cross-validation`和`真实的测试集`，对应第二波

- []  手写GBT（待定）

### 第一波 

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

总结：调参没用了，训不动，等第二波来


    