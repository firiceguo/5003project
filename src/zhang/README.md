# README
Final files:

 - baseline.py
 - recommedation.py
---

## baseline.py

 - what you must import 
```python
from pyspark.sql.functions import col
from pyspark.sql import Row
import json
import math
```
- Functions
 - `loadDataJson()` in this file
 - `calculateRMSE()` in this file
 
- USE
1. give 3 data file paths and a fraction to divide the training and test dataset
2. call the function `loadDataJson()` to return 3 data DF
2. call the function `calculateRMSE()`to calculate RMES
3. print the rmes

----------
## recommedation.py ##
- what you must import 
```python
from pyspark.sql.functions import col
from pyspark.sql import Row
import json
import math
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
from pyspark.sql import functions as F
```
- Functions
 - `GetRecomList()` is copied in this file, src is from lyy
 - `covtDataFormat()` in this file and used before recommendation
 - `recommsys()` in this file
- USE
1. give the review data file paths and a fraction to divide the training and test dataset
2. divide the dataset into training and testing
3. load the review data by 2 lines (just write here not as a function)
4. call the function `covtDataFormat()`to convert the original data format
5. call the function `recommsys()`
6. print the rmes
