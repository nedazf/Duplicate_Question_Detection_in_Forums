# Duplicate Questions across multiple Question-Answering Forums

The project aims at identifying duplicate questions across various question-answering platforms using NLP and machine learning. This helps in finding answers to various unanswered questions on one question-answering platform by referring to questions having similar intent across other platforms. 


## Dataset
* Quora (https://www.kaggle.com/c/quora-question-pairs/data)
    * Contains: question pairs, their ids and label is_duplicate

* AskUbuntu (https://archive.org/download/stackexchange)
    * Contains: XML files

* Apple,Askubuntu, Quora, Android, Sprint, Superuser (https://github.com/darsh10/qra_code)
    * Adversarial Domain Adaptation for Duplicate Question Detection
  
## Setup
Download and install the following based on your operating system:
- Spark (https://spark.apache.org/downloads.html)
- Cassandra (http://cassandra.apache.org/download/) 
- Flask (http://flask.pocoo.org/)
- React (https://react-cn.github.io/react/downloads.html)


## Project Components
Our projects is divided into following parts:

### Structuring Ask Ubuntu dataset
- First we begin by setting up the environment variable so that we can use it easily later. Example:

export SPARK_HOME = /usr/local/spark

- Since, the ask ubuntu dataset consists of 7 XML files, we have parsed two of them namely, posts.xml and postlinks.xml files. File posts.xml consists of questions and other information related to questions while the file postlinks.xml contains id of questions marked as duplicates by the users. So we parse both these files, extract question and question_id from posts.xml and combine it with the ids extracted from postlinks.xml to form duplicate question pairs for AskUbuntu dataset.

Run the following command to parse AskUbuntu dataset and store duplicate question pairs from askUbuntu dataset in auDuplicateQues.csv file.
`${SPARK_HOME}/bin/spark-submit pythonFiles/structureUbuntuXML.py `

### Pre-processing Data 
- In pre-processing, the script reads the csv file into a dataframe. Then HTML tags, punctuation, symbols, stop-words and non-ASCII characters are removed from the question pairs. Further, the text in questions is stemmed and lemmatized based on parameters set in the function call.

- As different datasets in csv files have different structures, there are different scripts to preprocess each dataset. Use similar command to preprocess datasets:

`${SPARK_HOME}/bin/spark-submit pythonFiles/preprocessQuora.py `

### Analysing Data
After creating the preprocessed data, we did some general analysis using matplotlib and seaborn to plot visualizations for summarizing the characteristics of data in each forum.

Run the following command to get plots for all datasets:
`${SPARK_HOME}/bin/spark-submit visualization/visualization.py`


### Implement and evaluate models
we used models such as Logistic Regression, Random Forest, XGBoost with different levels of TFIDF and BOW vectors and also a MaLSTM neural network with word2vec input vectors to predict the labels.
First, we conducted experiments on individual datasets (Quora, Apple, Android, Sprint, Superuser, and AskUbuntu). Second, we conduct experiments on our integrated dataset 
Further, we used the best performing model in terms of the FScore measure to build a small scale working system (web app interface).

Run the following command to train and save baseline models on integrated datasets:
`${SPARK_HOME}/bin/spark-submit pythonFiles/baselineModels.py `


## Siamese_LSTM

Using MaLSTM model(Siamese networks + LSTM with Manhattan distance) to detect semantic similarity between question pairs. Training dataset used is a subset of the original Quora Question Pairs Dataset(~363K pairs used).

It is Keras implementation based on [Original Paper(PDF)](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) and [Excellent Medium Article](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07).

![](https://cloud.githubusercontent.com/assets/9861437/20479493/6ea8ad12-b004-11e6-89e4-53d4d354d32e.png)

### Prerequisite

- Paper, Articles
    - [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)
    - [How to predict Quora Question Pairs using Siamese Manhattan LSTM](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)
- Data
    - [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
    - [Kaggle's Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/data)
- References
    - [aditya1503/Siamese-LSTM](https://github.com/aditya1503/Siamese-LSTM) Original author's GitHub
    - [dhwajraj/deep-siamese-text-similarity](https://github.com/dhwajraj/deep-siamese-text-similarity) TensorFlow based implementation

Kaggle's `test.csv` is too big, so I had extracted only the top 20 questions and created a file called `test-20.csv` and It is used in the `predict.py`.

You should put all data files to `./data` directory.

### How to Run
### Training
```
$ python3 train.py
```

### Predicting
It uses `test-20.csv` file mentioned above.
```
$ python3 predict.py
```

### The Results
I have tried with various parameters such as number of hidden states of LSTM cell, activation function of LSTM cell and repeated count of epochs.
I have used NVIDIA Tesla P40 GPU x 2 for training and 10% data was used as the validation set(batch size=1024*2).
As a result, I have reached about 82.29% accuracy after 50 epochs about 10 mins later.

```
Epoch 50/50
363861/363861 [==============================] - 12s 33us/step - loss: 0.1172 - acc: 0.8486 - val_loss: 0.1315 - val_acc: 0.8229
Training time finished.
50 epochs in       601.24
```


## Demo
### Loading dataset in Cassandra
Use the following command to upload data in cassandra tables: 

`${SPARK_HOME}/bin/spark-submit --packages datastax:spark-cassandra-connector:2.3.1-s_2.11 pythonFiles/loadData.py data/dataset1.csv data/dataset2.csv duplicates`

Here loadData.py is the file used to connect to cassandra and load data. Files dataset1.csv(questions from askUbuntu dataset) and dataset2.csv(questions from other datasets) include datasets that we want to upload in cassandra for demo. The last input in the above command 'duplicates' is the name of the keyspace that we are using in Cassandra.

### Set up Python API using flask
Use the following commands to set up python API using flask. 

`${SPARK_HOME}/bin/spark-submit --packages datastax:spark-cassandra-connector:2.3.1-s_2.11 pythonFiles/readLogs.py`

The above command sets up a flask app named readlogs which connects to cassandra and fetches data for the web app interface in react. The app is served on url http://127.0.0.1:5006/

To connect to cassandra, go to cassandra directory and run the command: `bin/cassandra -f`

There are two main APIs:
- API that fetches questions data from cassandra table: http://127.0.0.1:5006/datasets
- API that fetches questions, answers to questions, tags and similar question suggestions from cassandra: http://127.0.0.1:5006/modelresult/1&2 (here inputs 1 and 2 denote question ids of questions selected from dropdowns in the web app interface)

### Set up and run react app
To set up and run react app, go to path /frontend/templates/static, run the command mentioned below which loads all the node modules needed for the react app and bundles all the jsx files together in bundle.js :

`node_modules/.bin/webpack js/index.jsx public/bundle.js`

Next run the following command:

`npm run watch`

To set up the flask server, go to path /frontend and run the command mentioned below in a new terminal window:

`python run.py`

This sets up a flask app. Now the web app interface is up and running on http://127.0.0.1:5000/.







