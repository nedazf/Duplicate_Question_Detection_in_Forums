import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report, accuracy_score



quora_df = pd.read_csv('../data/preprocessed_Quora.csv', sep='\t', )
askubuntu_df = pd.read_csv('../data/preprocessed_AskUbuntu.csv', sep='\t')
final_df = pd.concat([quora_df, askubuntu_df], ignore_index=True)
df = final_df.sample(frac=1)
print(df.head())

# BOW + Xgboost Model
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
concatQues = pd.concat((df['question1'],df['question2'])).unique()
concatQues = pd.Series(concatQues)
count_vect.fit(concatQues.values.astype('str'))
trainq1_trans = count_vect.transform(df['question1'].values.astype('str'))
trainq2_trans = count_vect.transform(df['question2'].values.astype('str'))
labels = df['is_duplicate'].values
X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))
Y = labels
X_train,X_valid,Y_train,Y_valid = train_test_split(X,Y, test_size = 0.33, random_state = 42)

# train and save model
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train) 
xgb_prediction = xgb_model.predict(X_valid)

# predict and show result
print('training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))
print('validation score:', f1_score(y_valid, xgb_model.predict(X_valid), average='macro'))
print(classification_report(y_valid, xgb_prediction))



# Word level Tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
concatQues = pd.concat((df['question1'],df['question2'])).unique()
concatQues = pd.Series(concatQues)
tfidf_vect.fit(concatQues.values.astype('U'))
trainq1_trans = tfidf_vect.transform(df['question1'].values.astype('U'))
trainq2_trans = tfidf_vect.transform(df['question2'].values.astype('U'))
labels = df['is_duplicate'].values
X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))
y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

# train and save model
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train) 
filename = 'word_tfidf_model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))

# predict and show result
xgb_prediction = xgb_model.predict(X_valid)
print('word level tf-idf training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))
print('word level tf-idf validation score:', f1_score(y_valid, xgb_model.predict(X_valid), average='macro'))
print(classification_report(y_valid, xgb_prediction))



# N-gram Level TF-IDF
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
concatQues = pd.concat((df['question1'],df['question2'])).unique()
concatQues = pd.Series(concatQues)
tfidf_vect_ngram.fit(concatQues.values.astype('U'))
train_q1 = tfidf_vect_ngram.transform(df['question1'].values.astype('U'))
train_q2 = tfidf_vect_ngram.transform(df['question2'].values.astype('U'))
labels = df['is_duplicate'].values
X = scipy.sparse.hstack((train_q1,train_q2))
y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

# train and save model
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train) 
filename = 'ngram_tfidf_model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))

# predict and show result
xgb_prediction = xgb_model.predict(X_valid)
print('n-gram level tf-idf training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))
print('n-gram level tf-idf validation score:', f1_score(y_valid, xgb_model.predict(X_valid), average='macro'))
print(classification_report(y_valid, xgb_prediction))



# Character Level TF-IDF
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
concatQues = pd.concat((df['question1'],df['question2'])).unique()
concatQues = pd.Series(concatQues)
tfidf_vect_ngram_chars.fit(concatQues.values.astype('U'))
train_q1 = tfidf_vect_ngram_chars.transform(df['question1'].values.astype('U'))
train_q2 = tfidf_vect_ngram_chars.transform(df['question2'].values.astype('U'))
labels = df['is_duplicate'].values
X = scipy.sparse.hstack((train_q1,train_q2))
y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

# train and save model
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train) 
filename = 'char_tfidf_model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))

# predict and show result
xgb_prediction = xgb_model.predict(X_valid)
print('character level tf-idf training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))
print('character level tf-idf validation score:', f1_score(y_valid, xgb_model.predict(X_valid), average='macro'))
print(classification_report(y_valid, xgb_prediction))

