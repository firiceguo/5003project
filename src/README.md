## Before run:

1. Open the `run.py` and set your python environment.

```python
spark_home = '/SPARK_HOME/'
os.environ['SPARK_HOME'] = spark_home
mongouri = "mongodb://127.0.0.1/users"
mongo_spark_connector_path = '/MONGO_SPARK/mongo-spark-connector_2.10-2.0.0.jar'

```

2. About our dataset:

   We use the data from [Yelp Dataset Challenge](https://www.yelp.com/dataset_challenge/dataset) (1.8 GB tar file). Note that the data online is not the same as our data we used (it changed a few days ago), so we are not sure we can get the same result using this data.

## Run:

Only need to run the `run.py`.

```bash
python2 run.py
```

### Note:

- You have to run the `network/network.ipynb` on the Azure HDInsight.

- You have to install the NLTK. Besides, some corpses are required as well. You may need to use `nltk.download()` to download them:
    ​                              
    1. SentimentIntensityAnalyzer

    2. Tokenizer

- The baseline can run on the test dataset. About test dataset, you can run the `yelp/gettest.sh` to get the test dataset. (Maybe you have to do the `chmod 755 gettest.sh` first.)

- The GBT part have to run on the whole dataset.

### About Azure HDInsight

All of our code have run on Azure HDInsight, but we only provide a test version on physical machine to do the test. If you want to run our code on Azure, you have to do some configuration.

- Install NLTK for Azure;

- Run the `gbt/dataprocessing.py` to get the data for GBT model and import them to your MongoDB.

- Make sure that you have copied and pasted all of our support functions for Jupyter-Notebook on HDInsight.

- Configure your MongoDB and import all of the data.

- For our network part, make sure you have set up your `graframes`, then do the configuration following:

    ```python
    # If you do it on the Jupyter Notebook, do the following config.
    %%configure -f
    { "conf": {"spark.jars.packages": "graphframes:graphframes:0.3.0-spark2.0-s_2.11" }}

    sc.addPyFile('wasb://5003@network5003.blob.core.windows.net/graphframes-0.3.0-spark2.0-s_2.11.jar')
    ```

## Directory Architecture

Following is the directory architecture of our code.

Note that the `yelp` directory is our data path.

```
src
├── baseline
│   ├── baseline.py
│   ├── baseline.pyc
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── README.md
│   ├── recommedation.py
│   └── recommedation.pyc
├── gbt
│   ├── dataprocessing.py
│   ├── dataprocessing.pyc
│   ├── getdata-output.log
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── README.md
│   ├── traingbt.py
│   └── traingbt.pyc
├── graphframes-0.3.0-spark2.0-s_2.11.jar
├── graphframes.zip
├── mongo-spark-connector_2.10-2.0.0.jar
├── network
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── network.py
│   ├── network.pyc
│   └── README.md
├── nlp
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── README.md
│   ├── sentiment_based_model.py
│   ├── sentiment_based_model.pyc
│   ├── time_efficiency
│   │   ├── READMD.md
│   │   ├── Test_Efficiency_MongoDBvsJson.log
│   │   └── TestTimeConsume.py
│   ├── topic_based_model.py
│   └── topic_based_model.pyc
├── README.md
├── run.py
└── yelp
    ├── big_dataset
    │   ├── gbt_businesses.json
    │   ├── gbt_businesses_test.json
    │   ├── gbt_users.json
    │   ├── gbt_users_test.json
    │   └── gbt_user_test.json
    ├── business.json
    ├── business_test.json
    ├── checkin.json
    ├── checkin_test.json
    ├── gettest.sh
    ├── README.md
    ├── review.json
    ├── review_test.json
    ├── user.json
    └── user_test.json
```
