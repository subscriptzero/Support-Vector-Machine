
# coding: utf-8

# In[1]:
################# audience score script #################
######### import libraries.
import sys
import requests
import urllib
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import MySQLdb as mdb
import numpy as np
import time
import uuid
import semantria


# In[2]:

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

text1 = pd.DataFrame(pd.read_sql('select plot from flat_movies;', con=db))
text2 = pd.DataFrame(pd.read_sql('select video_text from user_process;', con=db))
text3 = pd.DataFrame(pd.read_sql('select overview from movies;', con=db))

####
text1.rename(columns={'plot': 'plot_text'}, inplace=True)
p1 = []
for word in text1.plot_text.tolist():
    p1.append(unicode(str(word), errors='ignore'))
v1 = []
for words in text2.video_text.tolist():
    v1.append(unicode(str(words), errors='ignore'))
o1 = []
for wordz in text3.overview.tolist():
    o1.append(unicode(str(wordz), errors='ignore'))
import unicodedata
class unicode(unicode):
    def __repr__(self):
        return __builtins__.unicode.__repr__(self).lstrip("u")
#word1 = [str(item) for item in word1]
#word2 = [str(item) for item in word2]
plot1 = []
for i in p1:
    plot1.append(unicodedata.normalize('NFKD',i).encode('ascii','ignore'))
video1 = []
for i in v1:
    video1.append(unicodedata.normalize('NFKD',i).encode('ascii','ignore'))
ovs1 = []
for i in o1:
    ovs1.append(unicodedata.normalize('NFKD',i).encode('ascii','ignore'))


values = plot1 + video1 + ovs1


# In[3]:

from string import punctuation
from collections import Counter
#import obo
full = filter(None, values)
full = [x for x in full if x != 'None']


# In[4]:

import nltk
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))
low =[]
for l in full:
    low.append([w.lower() for w in l.split() if w.isalpha()] )


# In[5]:

newlist = []
for line in low:
    newlist.append([w for w in line if w not in stops])


# In[6]:

###magic
wordlist = []
for items in newlist: 
    wordlist.append(Counter(items).most_common(2))    


# In[7]:

df = pd.DataFrame(wordlist)
cols = ['words1', 'words2']
df.columns = cols

df[['words1x', 'words1y']] = df['words1'].apply(pd.Series)
df[['words2x', 'words2y']] = df['words2'].apply(pd.Series)    
df.drop(['words1','words2','words1y','words2y'], axis=1, inplace=True)
words1 = df['words1x'].tolist()
words2 = df['words2x'].tolist()


# In[157]:

complete_url1 = []
base_url = 'https://www.30db.com/opinions/'
for i in (words1):
    u = base_url +  str(i) 
    complete_url1.append(u)

complete_url2 = []
for i in (words2):
    u = base_url +  str(i) 
    complete_url2.append(u)
    
#######################################scraping starts here!###############################################    
data = []
for l in complete_url1:
    web = urllib.urlopen(l)
    soup = BeautifulSoup(web.read(), 'lxml')
    #soup.prettify
    ##digging goldd
    data.append(soup.find_all("script")[0].string)
ans = []
for each in data:
    ##cleaning
    p = re.search(r'request.*}', each)
    #m = p.match(data)
    ans.append(p.group(0))
jsd = []
for more in ans:
    ans2 = re.search(r'{.*}',more)
    jsd.append(json.dumps(ans2.group(0),ensure_ascii=False))
my_list = []
for struc in jsd:
    #structuring
    jsl= str(json.loads(struc))
    #create list
    my_list.append(jsl.split(","))
Pos = []
Neg = []
PercentPos = []
for elem in my_list:
    #extract elements
    Pos.append(elem[7])
    Neg.append(elem[8])
    PercentPos.append(elem[9])
Pos1 = []
Neg1 = []
PercentPos1 = []
for i in Pos:
    #clean moreeeeeee
    Pos1.append(i.replace('"queryPCount":', ''))
for j in Neg:
    Neg1.append(j.replace('"queryNCount":',''))
for k in PercentPos:
    PercentPos1.append(k.replace('"queryPCTPos":',''))
Posdf = pd.DataFrame(Pos1, columns=["PositiveAudience"]) 
Negdf = pd.DataFrame(Neg1, columns=["NegativeAudience"])
PercentPosdf = pd.DataFrame(PercentPos1, columns=["PercentPositive"]) 
df1 = pd.concat([Posdf, Negdf, PercentPosdf], axis=1)
df1['PercentPositive'] = pd.to_numeric(df1['PercentPositive'], errors='ignore')
df1['PercentNegative'] = 100 - df1['PercentPositive']
df1            
            
data1 = []
for l1 in complete_url2:
    web1 = urllib.urlopen(l1)
    soup1 = BeautifulSoup(web1.read(), 'lxml')
    #soup.prettify
    ##digging goldd
    data1.append(soup1.find_all("script")[0].string)
ans1 = []
for each1 in data1:
    ##cleaning
    p1 = re.search(r'request.*}', each1)
    #m = p.match(data)
    ans1.append(p1.group(0))
jsd1 = []
for more1 in ans1:
    ans12 = re.search(r'{.*}',more1)
    jsd1.append(json.dumps(ans12.group(0),ensure_ascii=False))
my_list1 = []
for struc1 in jsd1:
    #structuring
    jsl1= str(json.loads(struc1))
    #create list
    my_list1.append(jsl1.split(","))
Pos1 = []
Neg1 = []
PercentPos1 = []
for elem1 in my_list1:
    #extract elements
    Pos1.append(elem1[7])
    Neg1.append(elem1[8])
    PercentPos1.append(elem1[9])
Pos2 = []
Neg2 = []
PercentPos2 = []
for i1 in Pos1:
    #clean moreeeeeee
    Pos2.append(i1.replace('"queryPCount":', ''))
for j1 in Neg1:
    Neg2.append(j1.replace('"queryNCount":',''))
for k1 in PercentPos1:
    PercentPos2.append(k1.replace('"queryPCTPos":',''))
Posdf1 = pd.DataFrame(Pos2, columns=["PositiveAudience"]) 
Negdf1 = pd.DataFrame(Neg2, columns=["NegativeAudience"])
PercentPosdf1 = pd.DataFrame(PercentPos2, columns=["PercentPositive"]) 
df2 = pd.concat([Posdf1, Negdf1, PercentPosdf1], axis=1)
df2['PercentPositive'] = pd.to_numeric(df2['PercentPositive'], errors='ignore')
df2['PercentNegative'] = 100 - df2['PercentPositive']
df2            
            
###combined audience score of the "two top words" of each text piece.
final = df1.add(df2, fill_value=0)
### average percentage audience while taking about the combined words.
final['PercentPositive'] = final['PercentPositive']/2
final['PercentNegative'] = final['PercentNegative']/2
final        

