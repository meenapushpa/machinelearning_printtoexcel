#import the key libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
x_train, x_test, y_train, y_test = train_test_split(train.cleaned_hm, labels, test_size=0.20,random_state=1)

# Another train_test_split for getting hmid values for test sets
x1_train, x1_test, y1_train, y1_test  = train_test_split(train.hmid, labels, test_size=0.20,random_state=1)

# print the hmid,cleaned_hm values for test sets in excel files
dftest=pd.DataFrame({'hm_id': x1_test,'cleaned_hm':x_test})
dftest.to_excel("test_set.xlsx",sheet_name='sheet1',index=False)

#1000 Since our theme is a thousand words
vectorizer = CountVectorizer(max_features=1000)
x_train = vectorizer.fit_transform(x_train)

#Apply the vectoriser on test data using the previous vocabulary set
feature_names = vectorizer.get_feature_names()
cvec_t = CountVectorizer(vocabulary=feature_names)
x_test = cvec_t.fit_transform(x_test).toarray()

# create dataset for lightgbm
lgb_train = lgb.Dataset(x_train.astype(np.float64), y_train)
lgb_eval = lgb.Dataset(x_test.astype(np.float64), y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': {'multi_logloss', 'multi_error'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
    ,'num_class': 7
}
# Gradient boosting method to get the accuracy
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)
pred = gbm.predict(x_test.astype(np.float64), num_iteration=gbm.best_iteration)
#pred is used to get the accuracy,precison,recall,F1 score
pred = pd.DataFrame(pred).idxmax(axis=1)
#predict calculations for Training and Test sets
pred1 = gbm.predict(x_test.astype(np.float64), num_iteration=gbm.best_iteration)
test_predict = pd.DataFrame(pred1).idxmax(axis=1)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(pred, y_test)
print('[INFO] Gradient Boosting Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(pred , y_test,labels=labels,pos_label=1, average='micro')
print('[INFO] Gradient Boosting Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(pred , y_test,labels=labels,pos_label=1,average='micro')
print('[INFO] Gradient Boosting Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(pred , y_test,labels=labels,pos_label=1, average='micro')
print('[INFO] Gradient Boosting F1 score: %f' % f1)

# printing the MultinomialNB predicted category to 'test_predict.xlsx' for test sets
dfF = pd.DataFrame({'Boosting_Predict':test_predict})
dfF = dfF.fillna(1)
dfF = dfF.astype(int)
dfF.replace({0:'affection', 1:'exercise', 2:'bonding', 3:'leisure', 4:'achievement' , 5:'enjoy_the_moment', 6:'nature'}, inplace=True)
dfF.to_excel("test_predict.xlsx",sheet_name='sheet1',index=False)

# concat the hmid,cleaned_hm,predicted_category values to 'test_consolidate.xlsx' for test sets
df1=pd.read_excel('test_set.xlsx')
df2=pd.read_excel('test_predict.xlsx')
False_data = pd.DataFrame()
False_data=pd.concat([df1,df2],axis=1)
False_data.to_excel("test_consolidate_gradient_boosting.xlsx",index=False)

print(" [SUCCESS] Gradient Boosting final data stored in test_consolidate_gradient_boosting.xlsx! ")
