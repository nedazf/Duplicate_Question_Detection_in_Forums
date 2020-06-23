import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import re, datetime
from uuid import uuid4
from pyspark.sql import SparkSession, functions, types, Row
cluster_seeds = ['127.0.0.1']
spark = SparkSession.builder.appName('Load Logs Spark Cassandra').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
import pandas as pd

def main(input1, input2, keyspace):
    # read dataset1.csv
    dataset1 = pd.read_csv(input1, sep='\t')
    print(dataset1.head())
    df1 = spark.createDataFrame(dataset1)

    # read dataset2.csv
    dataset2 =  pd.read_csv(input2, sep='\t')
    print(dataset2.head())
    df2 = spark.createDataFrame(dataset2)

    # load data in cassandra
    df1.write.format("org.apache.spark.sql.cassandra").options(table='dataset1', keyspace=keyspace).save()
    df2.write.format("org.apache.spark.sql.cassandra").options(table='dataset2', keyspace=keyspace).save()
    
if __name__ == '__main__':
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    keyspace = sys.argv[3]
    main(input1, input2, keyspace)
