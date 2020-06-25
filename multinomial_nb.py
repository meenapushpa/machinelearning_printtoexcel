#import the key libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score
import lightgbm as lgb
from pandas import crosstab
import xlsxwriter
# import data
train = pd.read_csv("D:\\Test\\cleaned_hm.csv")

# get the unique predicted category
labels=train.predicted_category.unique()
dic={}
for i,labels in enumerate(labels):
    dic[labels]=i
labels=train.predicted_category.apply(lambda x:dic[x])

# using train_test_split method to split training sets and test sets
x_train, x_test, y_train, y_test  = train_test_split(train.cleaned_hm, labels, test_size=0.20,random_state=1)

# Another train_test_split for getting hmid values for test sets
x1_train, x1_test, y1_train, y1_test  = train_test_split(train.hmid, labels, test_size=0.20,random_state=1)

# print the hmid,cleaned_hm values for test sets in excel files
dftest=pd.DataFrame({'hmid': x1_test,'cleaned_hm':x_test})
dftest.to_excel("test_set.xlsx",sheet_name='sheet1',index=False)

# 1000 Since our theme is a thousand words
vectorizer = CountVectorizer(max_features=1000)
x_train = vectorizer.fit_transform(x_train)

# Apply the vectoriser on test data using the previous vocabulary set
feature_names = vectorizer.get_feature_names()
cvec_t = CountVectorizer(vocabulary=feature_names)
x_test = cvec_t.fit_transform(x_test).toarray()

# MultinomialNB method to get accuracy_score
nb = MultinomialNB()
nb.fit(x_train, y_train)
nb.score(x_test, y_test) # the test dataset is

# predict calculations for Training and Test sets
test_predict=nb.predict(x_test)

# yhat_probs is used to get the accuracy,precison,recall,F1 score
yhat_probs = nb.predict(x_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_probs)
print('[INFO] Naive Bayes Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_probs,labels=labels,pos_label=1, average='micro')
print('[INFO] Naive Bayes Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_probs,labels=labels,pos_label=1, average='micro')
print('[INFO] Naive Bayes Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_probs,labels=labels,pos_label=1,average='micro')
print('[INFO] Naive Bayes F1 score: %f' % f1)

# printing the MultinomialNB predicted category to 'test_predict.xlsx' for test sets
dfF = pd.DataFrame({'NB_Predict':test_predict})
dfF = dfF.fillna(1)
dfF = dfF.astype(int)
dfF.replace({0:'affection', 1:'exercise', 2:'bonding', 3:'leisure', 4:'achievement' , 5:'enjoy_the_moment', 6:'nature'}, inplace=True)
dfF.to_excel("test_predict.xlsx",sheet_name='sheet1',index=False)

# concat the hmid,cleaned_hm,predicted_category values to 'test_consolidate.xlsx' for test sets
df1=pd.read_excel('test_set.xlsx')
df2=pd.read_excel('test_predict.xlsx')
False_data = pd.DataFrame()
False_data=pd.concat([df1,df2],axis=1)
False_data.to_excel("test_consolidate_naive_bayes.xlsx",index=False)

print(" [SUCCESS] Multinomial NB final data stored in test_consolidate_naive_bayes.xlsx! ")
