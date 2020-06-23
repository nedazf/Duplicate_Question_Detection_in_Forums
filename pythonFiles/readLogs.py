import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import  math, json
from pyspark.sql import SparkSession, functions as F
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import scipy
import pandas as pd
cluster_seeds = ['127.0.0.1']
spark = SparkSession.builder.appName('Correlate Logs Spark Cassandra').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#API
from flask import Flask, request
from flask_restful import Resource, Api
from flask_jsonpify import jsonify

from flask_cors import CORS

app = Flask(__name__)
# api = Api(app)
api = CORS(app)

def customFunction(row):
   return {'id' : row.id, 'ques': row.question}

@app.route("/datasets")
def Datasets():
    	# get questions from dataset 1
        df1 = spark.read.format("org.apache.spark.sql.cassandra").options(table='dataset1', keyspace='duplicates').load()
        ques_df1 = df1.rdd.map(customFunction)
        dataset1 = ques_df1.collect()

        # get questions from dataset 2
        df2 = spark.read.format("org.apache.spark.sql.cassandra").options(table='dataset2', keyspace='duplicates').load()
        ques_df2 = df2.rdd.map(customFunction)
        dataset2 = ques_df2.collect()
        data = {'dataset1': dataset1, 'dataset2': dataset2}
        return jsonify(data)

@app.route("/modelresult/<id1>&<id2>")
def ModelImplementation(id1,id2):
    # get question 1 from dataset 1
    df1 = spark.read.format("org.apache.spark.sql.cassandra").options(table='dataset1', keyspace='duplicates').load()
    df1 = df1.toPandas()
    ques1 = df1.loc[df1['id'] == int(id1), 'question'].iloc[0]
    ans1 = df1.loc[df1['id'] == int(id1), 'answer'].iloc[0]
    tag1 = df1.loc[df1['id'] == int(id1), 'tag'].iloc[0]
    similar1 = df1.loc[df1['id'] == int(id1), 'similar'].iloc[0]

    # get question 2 from dataset 2
    df2 = spark.read.format("org.apache.spark.sql.cassandra").options(table='dataset2', keyspace='duplicates').load()
    df2 = df2.toPandas()
    ques2 = df2.loc[df2['id'] == int(id2), 'question'].iloc[0]
    ans2 = df2.loc[df2['id'] == int(id2), 'answer'].iloc[0]
    tag2 = df2.loc[df2['id'] == int(id2), 'tag'].iloc[0]
    
    answer = ''
    if ans1 != 'no answer':
        answer = ans1
    else:
        answer = ans2


    test_df = pd.DataFrame()
    test_df['question1'] = [ques1]
    test_df['question2'] = [ques2]

    quora_df = pd.read_csv('preprocessed_Quora.csv', sep='\t', )
    askubuntu_df = pd.read_csv('preprocessed_AskUbuntu.csv', sep='\t')
    final_df = pd.concat([quora_df, askubuntu_df], ignore_index=True)
    df = final_df.sample(frac=1)

    # convert selected questions to feature vectors
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    concatQues = pd.concat((df['question1'],df['question2'])).unique()
    concatQues = pd.Series(concatQues)
    count_vect.fit(concatQues.values.astype('U'))
    train_q1 = count_vect.transform(test_df['question1'].values.astype('U'))
    train_q2 = count_vect.transform(test_df['question2'].values.astype('U'))
    X = scipy.sparse.hstack((train_q1,train_q2))

    # load the model from disk
    loaded_model = pickle.load(open('bow_xgboost_model.sav', 'rb'))
    xgb_prediction = loaded_model.predict(X)
    data = {'prediction': str(xgb_prediction[0]), 'ques1': ques1, 'ques2':ques2, 'answer': answer, 'tag1': tag1, 'tag2': tag2, 'similar': similar1}
    # print(xgb_prediction[0])
    return jsonify(data)


if __name__ == '__main__':
    app.run(port='5006')
