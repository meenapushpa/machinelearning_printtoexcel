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

# print the hmid,cleaned_hm values for training and test sets in excel files
dftrain=pd.DataFrame(x_train)
dftrain.columns = ['cleaned_hm']
dftrain.to_excel("training_set.xlsx",sheet_name='sheet1')
dftest = pd.DataFrame(x_test)
dftest.columns = ['cleaned_hm']
dftest.to_excel("test_set.xlsx",sheet_name='sheet1')

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
training_predict=nb.predict(x_train)

# yhat_probs is used to get the accuracy,precison,recall,F1 score
yhat_probs = nb.predict(x_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_probs)
print('[INFO] Naive Bayes Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_probs, average='micro')
print('[INFO] Naive Bayes Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_probs, average='micro')
print('[INFO] Naive Bayes Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_probs, average='micro')
print('[INFO] Naive Bayes F1 score: %f' % f1)

# printing the MultinomialNB predicted category to 'test_predict.xlsx' for test sets
dfF = pd.DataFrame({'NB_Predict':test_predict})
dfF = dfF.fillna(1)
dfF = dfF.astype(int)
dfF.replace({0:'affection', 1:'exercise', 2:'bonding', 3:'leisure', 4:'achievement' , 5:'enjoy_the_moment', 6:'nature'}, inplace=True)
dfF.to_excel("test_predict.xlsx",sheet_name='sheet1')

#Printing the MultinomialNB predicted category to 'training_predict.xlsx' for training sets
dfT = pd.DataFrame({'NB_Predict':training_predict})
dfT = dfT.fillna(1)
dfT = dfT.astype(int)
dfT.replace({0:'affection', 1:'exercise', 2:'bonding', 3:'leisure', 4:'achievement' , 5:'enjoy_the_moment', 6:'nature'}, inplace=True)
dfT.to_excel("training_predict.xlsx",sheet_name='sheet1')

# concat the hmid,cleaned_hm,predicted_category values to 'test_consolidate.xlsx' for test sets
df1=pd.read_excel('test_set.xlsx')
df2=pd.read_excel('test_predict.xlsx')
df1.rename({"Unnamed: 0":"hmid"}, axis="columns", inplace=True)
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
False_data = pd.DataFrame()
False_data=pd.concat([df1,df2],axis=1)
False_data.to_excel("test_consolidate.xlsx")

#concat the hmid,cleaned_hm,predicted_category values to 'training_consolidate.xlsx' for training sets
df1=pd.read_excel('training_set.xlsx')
df2=pd.read_excel('training_predict.xlsx')
df1.rename({"Unnamed: 0":"hmid"}, axis="columns", inplace=True)
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
True_data = pd.DataFrame()
True_data=pd.concat([df1,df2],axis=1)
True_data.to_excel("training_consolidate.xlsx")

# Append the data from the 'test_consolidate.xlsx and training_consolidate.xlsx' files one by one and exporting naive_bayes_final_predict.xlsx spreadsheet
Files=['test_consolidate.xlsx','training_consolidate.xlsx']
all_data = pd.DataFrame()
for f in Files:
    df = pd.read_excel(f)
    all_data = all_data.append(df,ignore_index=True)
    all_data = all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')]
all_data.to_excel("naive_bayes_final_predict.xlsx",index=False)
print(" [SUCCESS] Multinomial NB final data stored in naive_bayes_final_predict.xlsx! ")
