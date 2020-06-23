import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from os import path



pal = sns.color_palette()

print('# File sizes')
data_names=['quora','android','apple','askubuntu','sprint','superuser']
cloudshapes= ['quora.png','android.png','apple-logo.png','circle.jpeg','circle.jpeg','circle.jpeg']
for data_name,cloudshape in zip(data_names,cloudshapes):
    print(cloudshape+'***',data_name)

    data_path = "../../data/data/"+data_name+"/combined"
    visualization_path = '../../visualization/'+data_name+"/"
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    for f in os.listdir(data_path):
        if 'zip' not in f:
            print(f.ljust(30) + str(round(os.path.getsize(data_path+"/" + f) / 1000000, 2)) + 'MB')
    df_train = pd.read_csv(data_path +"/"+'full_converted.csv')
    df_train.head(20)


    print(data_name)
    print('Total number of question pairs for training: {}'.format(len(df_train)))
    print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
    qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
    print('Total number of questions in the training data: {}'.format(len(
        np.unique(qids))))
    print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

    plt.figure(figsize=(12, 5))
    plt.hist(qids.value_counts(), bins=50)
    plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of question appearance counts',fontsize=22)
    plt.xlabel('Number of occurences of question',fontsize=22)
    plt.ylabel('Number of questions',fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig(visualization_path+'normalised_histogram',bbox_inches='tight')

    print()

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    # test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

    dist_train = train_qs.apply(len)
    # dist_test = test_qs.apply(len)
    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], density=True, label='train')
    # plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], density=True, alpha=0.5, label='test')
    plt.title('Normalised histogram of character count in questions', fontsize=22)
    plt.legend()
    plt.xlabel('Number of characters', fontsize=22)
    plt.ylabel('Probability', fontsize=22)
    plt.savefig(visualization_path+'histogram')



    print('mean-fulldata {:.2f} std-fulldata {:.2f}  max-fulldata {:.2f}'.format(dist_train.mean(),
                              dist_train.std(), dist_train.max()))

    dist_train = train_qs.apply(lambda x: len(x.split(' ')))
    # dist_test = test_qs.apply(lambda x: len(x.split(' ')))

    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], density=True, label='train')
    # plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], density=True, alpha=0.5, label='test')
    plt.title('Normalised histogram of word count in questions', fontsize=22)
    plt.legend()
    plt.xlabel('Number of words', fontsize=22)
    plt.ylabel('Probability', fontsize=22)
    plt.savefig(visualization_path+'character_count')

    print('mean-train {:.2f} std-train {:.2f}  max-train {:.2f} '.format(dist_train.mean(),
                              dist_train.std(), dist_train.max()))

    from wordcloud import WordCloud

    alice_mask = np.array(Image.open(cloudshape))
    cloud = WordCloud(width=1440, height=1080, mask=alice_mask,background_color="white",contour_color='steelblue').generate(" ".join(train_qs.astype(str)))
    plt.figure(figsize=(20, 15))
    plt.imshow(cloud)
    plt.axis('off')

    plt.savefig(visualization_path+cloudshape)

    qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
    math = np.mean(train_qs.apply(lambda x: '[math]' in x))
    fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
    capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
    capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
    numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))

    print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
    print('Questions with [math] tags: {:.2f}%'.format(math * 100))
    print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
    print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))
    print('Questions with capital letters: {:.2f}%'.format(capitals * 100))
    print('Questions with numbers: {:.2f}%'.format(numbers * 100))

    from nltk.corpus import stopwords

    stops = set(stopwords.words("english"))

    def word_match_share(row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
        return R

    plt.figure(figsize=(15, 5))
    train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
    plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, density=True, label='Not Duplicate')
    plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, density=True, alpha=0.7, label='Duplicate')
    plt.legend()
    plt.title('Label distribution over word_match_share', fontsize=22)
    plt.xlabel('word_match_share', fontsize=22)
    plt.savefig(visualization_path+'label_distribution')
