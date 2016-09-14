
# coding: utf-8

# In[1]:

################### Valuation___script #################################
##################################################################
##################################################################

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
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.cross_validation import StratifiedKFold
from sklearn import ensemble, preprocessing, cross_validation
from sklearn import metrics 
from sklearn import grid_search
from sklearn.ensemble import AdaBoostRegressor #For Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
from nltk import *
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import TextBlob 
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###############################
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

fm = pd.DataFrame(pd.read_sql
                  ('select title, production_year,genre,budget,rating,plot, keywords, votes, gross from flat_movies WHERE gross IS NOT NULL AND votes IS NOT NULL AND keywords IS NOT NULL AND plot IS NOT NULL AND rating IS NOT NULL AND budget IS NOT NULL AND genre IS NOT NULL AND production_year IS NOT NULL AND title IS NOT NULL;', con=db))
cu = pd.DataFrame(pd.read_sql('select user_id, ref, video_text, model, genre,investment,s1_score,s2_score,s3_score,s4_score,s5_score from user_process WHERE video_text IS NOT NULL AND model IS NOT NULL AND genre IS NOT NULL AND investment IS NOT NULL AND s1_score IS NOT NULL AND s2_score IS NOT NULL AND s3_score IS NOT NULL AND s4_score IS NOT NULL AND s5_score IS NOT NULL;', con=db))
userref = sys.argv[1]
cd = pd.DataFrame(pd.read_sql('select * from user_process WHERE ref=%(userref)s;', con=db,params={"userref":userref}))
print("done collecting data in %0.3fs" % (time() - t0))


cu.rename(columns={'investment': 'budget'}, inplace=True)


##join the two sets as com.
com = pd.concat([cu,fm], ignore_index=True)
com['combined'] = com["video_text"].map(str) + ","+ com["plot"].map(str) + "," + com["keywords"].map(str) + "," + com["title"].map(str)


###putting timer to keep check 
t0 = time()
def coder(x):
    p1 = []
    for word in x.tolist():
        p1.append(unicode(str(word), errors='ignore'))
    plot1 = []    
    for i in p1:
        plot1.append(unicodedata.normalize('NFKD',i).encode('ascii','ignore'))
    return plot1

plottext= pd.DataFrame(coder(com['plot']), columns=["plot"]) 
videotext = pd.DataFrame(coder(com['video_text']), columns=["video_text"])
titletext = pd.DataFrame(coder(com['title']), columns=["title"])
keywordstext = pd.DataFrame(coder(com['keywords']), columns=["keywords"])
combinetext = pd.DataFrame(coder(com['combined']), columns=["combined"])
import warnings
warnings.filterwarnings("ignore")
from nltk import *
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import TextBlob 
plottext['polarity_plot'] = plottext.apply(lambda x: TextBlob(x['plot']).sentiment.polarity, axis=1)
plottext['subjectivity_plot'] = plottext.apply(lambda x: TextBlob(x['plot']).sentiment.subjectivity, axis=1)
videotext['polarity_video'] = videotext.apply(lambda x: TextBlob(x['video_text']).sentiment.polarity, axis=1)
videotext['subjectivity_video'] = videotext.apply(lambda x: TextBlob(x['video_text']).sentiment.subjectivity, axis=1)
keywordstext['polarity_key'] = keywordstext.apply(lambda x: TextBlob(x['keywords']).sentiment.polarity, axis=1)
keywordstext['subjectivity_key'] = keywordstext.apply(lambda x: TextBlob(x['keywords']).sentiment.subjectivity, axis=1)
titletext['polarity_title'] = titletext.apply(lambda x: TextBlob(x['title']).sentiment.polarity, axis=1)
titletext['subjectivity_title'] = titletext.apply(lambda x: TextBlob(x['title']).sentiment.subjectivity, axis=1)
combinetext['polarity_combine'] = combinetext.apply(lambda x: TextBlob(x['combined']).sentiment.polarity, axis=1)
combinetext['subjectivity_combine'] = combinetext.apply(lambda x: TextBlob(x['combined']).sentiment.subjectivity, axis=1)

print("done scoring in %0.3fs" % (time() - t0))



#cu = pd.merge(cu, videotext,on = ['video_text'], left_index=True, right_index=True,how = 'inner' )
#cu.rename(columns={'polarity_y': 'sentiment'}, inplace=True)
com1 = pd.merge(com,videotext,on = ['video_text'], left_index=True, right_index=True,how = 'inner')
com2 = pd.merge(com1,plottext,on = ['plot'], left_index=True, right_index=True,how = 'inner')
com3 = pd.merge(com2,keywordstext,on = ['keywords'], left_index=True, right_index=True,how = 'inner')
com4 = pd.merge(com3,titletext,on = ['title'], left_index=True, right_index=True,how = 'inner')
com_5 = pd.merge(com4,combinetext,on = ['combined'], left_index=True, right_index=True,how = 'inner')
#cu.drop(['video_text'],axis=1,inplace=True)


#################################### very important ####################################
t0 = time()
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer        
from nltk.stem.porter import PorterStemmer

#######
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems
######## 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 50)

##apply functions.
vlist = videotext['video_text'].tolist()
plist = plottext['plot'].tolist()
klist = keywordstext['keywords'].tolist()
tlist = titletext['title'].tolist()
clist = combinetext['combined'].tolist()

from sklearn.feature_extraction.text import TfidfTransformer

#transform to vector
tfidf_transformer = TfidfTransformer(use_idf = True)
vtfidf = tfidf_transformer.fit_transform(vectorizer.fit_transform(vlist).toarray()).toarray()
ptfidf = tfidf_transformer.fit_transform(vectorizer.fit_transform(plist).toarray()).toarray()
ktfidf = tfidf_transformer.fit_transform(vectorizer.fit_transform(klist).toarray()).toarray()
ttfidf = tfidf_transformer.fit_transform(vectorizer.fit_transform(tlist).toarray()).toarray()
ctfidf = tfidf_transformer.fit_transform(vectorizer.fit_transform(clist).toarray()).toarray()

def changetodf(x):
    df = []
    for i in x:
        df.append(i)
        up =np.vstack(df)
    return pd.DataFrame(up)
        
vdf = changetodf(vtfidf)
pdf = changetodf(ptfidf)
kdf = changetodf(ktfidf)
tdf = changetodf(ttfidf)
cdf = changetodf(ctfidf)
print("done scoring in %0.3fs" % (time() - t0))



com_5['tmp'] = 1
vdf['tmp'] = 1
pdf['tmp'] = 1
kdf['tmp'] = 1
tdf['tmp'] = 1
cdf['tmp'] = 1
df1 = com_5.merge(vdf, on=['tmp'],left_index=True, right_index=True,how = 'inner')
df2 = df1.merge(pdf, on=['tmp'],left_index=True, right_index=True,how = 'inner')
df3 = df2.merge(kdf, on=['tmp'],left_index=True, right_index=True,how = 'inner')
df4= df3.merge(tdf, on=['tmp'],left_index=True, right_index=True,how = 'inner')
df_all = df4.merge(cdf, on=['tmp'],left_index=True, right_index=True,how = 'inner')


#list(df_all.columns.values)
df_all.drop(['keywords','title','plot','video_text', 'combined', 'tmp'], axis=1, inplace=True)


list = ['s1_score','s2_score','s3_score','s4_score','s5_score', 'user_id', 'votes','rating']
for l in list:
    df_all[l] = df_all.groupby(['budget','genre'])[l].transform(lambda grp: grp.fillna(method='ffill'))


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(df_all['rating'].reshape(-1,1))
df_all['rating']= imp.transform(df_all['rating'].reshape(-1,1))
imp.fit(df_all['votes'].reshape(-1,1))
df_all['votes']= imp.transform(df_all['votes'].reshape(-1,1))



from sklearn import preprocessing
from sklearn.pipeline import Pipeline
class MultiColumn:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode
    def fit(self,X,y=None):
        return self # not relevant here
    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
 



df_all =MultiColumn(columns = ['model', 'user_id', 'genre']).fit_transform(df_all)
#cu =MultiColumn(columns = ['model', 'genre', 'user_id']).fit_transform(cu)



#################### I Ran it once to check the importance of our features. Interestingly all the features out of score on 0 to 1, obtained ~1!! We have great feeatures in hand##############
################# No need to run #########################
#from sklearn.linear_model import RandomizedLasso ## lasso rocks for regression problem such as ours. 
#rlasso = RandomizedLasso(alpha=0.8) ## elasticnet (l1 + l2=== kill the overfitting!)
#prefeatures = {'budget' ,'genre','investment','model','s1_score','rating','votes','sentiment', 's2_score','s3_score','s4_score','s5_score','meanbudget'}
#X = com_new[list(prefeatures)].values
#Y = list(com_new['gross'].values)
#rlasso.fit(X, Y)
#names = list(prefeatures)
#print "Features sorted by their score:"
#print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True)
###output:
#Features sorted by their score:
#[(1.0, 'votes'), (1.0, 'sentiment'), (1.0, 's5_score'), (1.0, 's4_score'), (1.0, 's2_score'), 
#(1.0, 's1_score'), (1.0, 'rating'), (1.0, 'model'), (1.0, 'meanbudget'), 
#(1.0, 'investment'), (1.0, 'genre'), (1.0, 'budget'), (0.995, 's3_score')]
df_all = df_all.fillna(-999)
train = df_all[df_all.ref == -999]
train.reset_index()
train.drop(['ref'], axis=1, inplace=True)
test = df_all[df_all.ref != -999]

train['gross'] = np.log(train['gross'].astype('float64'))
train['budget'] = np.log(train['budget'].astype('float64'))
train['votes'] = np.log(train['votes'].astype('float64'))

test['gross'] = np.log(test['gross'].astype('float64'))
test['budget'] = np.log(test['budget'].astype('float64'))
test['votes'] = np.log(test['votes'].astype('float64'))



from sklearn.cross_validation import train_test_split
# Generate the training set.  Set random_state to be able to replicate results.
#xtrain = train.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
#xtest = train.loc[~train.index.isin(xtrain.index)]
# Print the shapes of both sets.
#print(xtrain.shape)
#print(xtest.shape)



##### @train the model ###############
##### @grid searching === God ---saves time as well ############
features = train.columns.tolist()
# Filter the columns to remove ones we don't want.
features = [c for c in features if c not in [ "gross"]]

#features = ['budget' ,'genre','s1_score','rating','votes','sentiment', 's2_score','s3_score','s4_score','s5_score','avg_score','budgetbygenre','budgetbygenrerating','scorebygenre','votegenre','voterate' ]
x = train[features]
y = train['gross'].values
xtest = test[features]


param = { "n_estimators":[100],"learning_rate":[0.1], "loss":['linear']} ##exponential because budegt and gross are in different scale then rest of data.
t0 = time()
gs = grid_search.GridSearchCV(AdaBoostRegressor(base_estimator = DecisionTreeRegressor()),param_grid=param,verbose=1, n_jobs = 3)
gs.fit(x, y)
print ("Starting to predict")
pred= pd.DataFrame(gs.predict(xtest))
print ("Prediction Completed")
print("done model training & prediction in %0.3fs" % (time() - t0))


test['gross'] = pred

###final output
df_final = test[['ref', 'gross']]



#to take only first row. need to  make sure that new user ref always is at the top.
ff = df_final.ix[0,]



# === DUG ADDITION === #
# userref
import json
outputjson=json.dumps({"gross":ff['gross'].tolist()}, separators=(',',':'))
print (outputjson)

sql = "UPDATE user_process SET processed_at=NOW(), output_json='%s' WHERE ref='%s';" % (outputjson, userref)

cur.execute(sql)
db.commit


# In[ ]:

################################## END. ###########################################

