
# coding: utf-8

# In[1]:

########################## Plot text sentiment+score script ########################################

import MySQLdb as mdb
import json
import re
import pandas as pd
import numpy as np
import sys
import time
import uuid
import semantria
import unicodedata
from time import time
from string import punctuation
from collections import Counter
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
from nltk import *
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import TextBlob 
from textblob.taggers import NLTKTagger
from textblob.classifiers import NaiveBayesClassifier

pd.set_option('display.float_format', lambda x: '%.3f' % x)

t0 = time()

db = mdb.connect(host="filmfundr.com", user="ffuser", passwd="FF*123!", db="filmfundrdb")
cur = db.cursor()

cur.execute("SHOW tables")
result = cur.fetchall()
##or print result
#for (table_name,) in cur:
 #   print(table_name)
if result[0] == []: 
	print "Connection Error!"
else:
	print "Connected successfully! continue your job..."
db.autocommit(True)
print("Reading collection from file...")

fmovies = pd.DataFrame(pd.read_sql('select plot from flat_movies WHERE plot IS NOT NULL limit 10000 ;', con=db))
fmovies.rename(columns={'plot': 'plot_text'}, inplace=True)
p1 = []
for word in fmovies.plot_text.tolist():
    p1.append(unicode(str(word), errors='ignore'))
import unicodedata
class unicode(unicode):
    def __repr__(self):
        return __builtins__.unicode.__repr__(self).lstrip("u")
#word1 = [str(item) for item in word1]
#word2 = [str(item) for item in word2]
plot1 = []
for i in p1:
    plot1.append(unicodedata.normalize('NFKD',i).encode('ascii','ignore'))
plottext= pd.DataFrame(plot1, columns=["text"]) 

stops = set(stopwords.words('english'))
low =[]
for l in plot1:
    low.append([w.lower() for w in l.split() if w.isalpha()] )

newlist = []
for line in low:
    newlist.append([w for w in line if w not in stops])


wordlist = []
for items in newlist: 
    wordlist.append(Counter(items).most_common(2))    

df = pd.DataFrame(wordlist, columns = ['ImportantWord1','ImportantWord2'])
df['tmp'] = 1
plottext['tmp'] = 1

df_final = pd.merge(plottext, df, on = ['tmp'], left_index=True, right_index=True,how = 'inner')
df_final.drop(['tmp'],axis = 1,  inplace = True)
nltk_tagger = NLTKTagger()
df_final['polarity'] = df_final.apply(lambda x: TextBlob(x['text']).sentiment.polarity, axis=1)
df_final['subjectivity'] = df_final.apply(lambda x: TextBlob(x['text']).sentiment.subjectivity, axis=1)
df_final['Phrases'] = df_final.apply(lambda x: TextBlob(x['text']).noun_phrases, axis=1)
df_final['Ngram'] = df_final.apply(lambda x: TextBlob(x['text']).ngrams(n=3), axis=1)
#df_final['pos'] = df_final.apply(lambda x: TextBlob(x['text'], analyzer=NaiveBayesAnalyzer()).sentiment.p_pos, axis=1)
train = [('I love this sandwich.', 'pos'),
         ('this is an amazing place!', 'pos'),
     ('I feel very good about these beers.', 'pos'),
     ('this is my best work.', 'pos'),
    ("what an awesome view", 'pos'),
     ('I do not like this restaurant', 'neg'),
     ('I am tired of this stuff.', 'neg'),
     ("I can't deal with this", 'neg'),
     ('he is my sworn enemy!', 'neg'),
     ('my boss is horrible.', 'neg'),
     ('the beer was good.', 'pos'),
     ('I do not enjoy my job', 'neg'),
     ("I ain't feeling dandy today.", 'neg'),
     ("I feel amazing!", 'pos'),
     ('Gary is a friend of mine.', 'pos'),
     ("I can't believe I'm doing this.", 'neg')]

cl = NaiveBayesClassifier(train)
df_final['Classification'] = df_final.apply(lambda x: TextBlob(x['text'], classifier=cl).classify(), axis=1)
df_final
print("done collecting and then performing scoring on data in %0.3fs" % (time() - t0))
######################################### END ###########################################




