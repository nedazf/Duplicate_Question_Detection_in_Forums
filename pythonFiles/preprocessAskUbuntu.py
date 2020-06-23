import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re
import spacy


df = pd.read_csv('auDuplicateQues.csv', sep='\t')
df = df[['ques1', 'ques2', 'is_duplicate']]
print(df.isnull().sum())
df = df.dropna()
print("\nAfter preprocessing \n-------------------")
print(df.isnull().sum())


# add spacing at the beginning and end of string
def addPadding(string):
    return ' '+string+' '
    
def removeStopWords(text):
    stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']
    
    text = text.split()
    text = [w for w in text if not w in stop_words]
    text = " ".join(text)
    return text

def runStemmer(text):
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

def runLemmatizer(text):
    sp = spacy.load('en')
    text = sp(text)
    text = " ".join([token.lemma_ for token in text])
    return text

# normalize text
def normalizeText(text, remStopWords=True, stemWords=False):

    text = str(text)
    
    # Handle known symbols like $, %, &
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    
    # remove remaining punctuation from the text
    text = re.sub(r'[^\w\s]','', text)
    
    # replace nonascii characters by word 'nonascii'
    text = re.sub('[^\x00-\x7F]+', addPadding('nonascii'), text)
    
    if remStopWords:
        text = removeStopWords(text)
    
    if stemWords:
        text = runStemmer(text)
        
    text = runLemmatizer(text)
    text = re.sub(r'[^\w\s]','', text)
    
    # finally, convert all words in text to lower case
    result = text.lower()
    return result

df['ques1'] = df['ques1'].apply(normalizeText, args=(True, False))
print('normalized question 1')
df['ques2'] = df['ques2'].apply(normalizeText, args=(True, False))
print('normalized question 2')

a = 0 
for i in range(a,a+10):
#     print(df.question1[i])
    print(df.ques1[i])
#     print(df.question2[i])
    print(df.ques2[i])
    print()


askUbuntu_df = pd.DataFrame({'question1':df['ques1'], 'question2':df['ques2'], 'is_duplicate': df['is_duplicate']})
askUbuntu_df.to_csv('../data/preprocessed_AskUbuntu.csv', sep='\t', index=False)
