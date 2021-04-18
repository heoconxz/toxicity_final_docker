#Author: Tung Tran
#Student ID: x2020fjw
#Student number: 202005431
#Email: x2020fjw@stfx.ca
import numpy as np
import pandas as pd 
import sklearn as sk 
import matplotlib.pyplot as plt

from collections import Counter


#---------------------------------------------------------------------------
#-------Preprocessing-------------------------------------------------------
#---------------------------------------------------------------------------


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
feamat = pd.read_csv('feamat.csv')

train[['chemId','assayId']] = train.Id.str.split(';',1,expand=True)
test[['chemId','assayId']] = test.x.str.split(';',1,expand=True)

test_id = test.x
train = train.drop(['Id'],1)
test = test.drop(['x'],1)

feamat = feamat.drop('V2',1)


#Merge feamat with train and test

def merge_feamat(original, feamat):
    original = pd.merge(original,feamat,how='left',left_on='chemId',right_on='V1')
    original = original.drop(['chemId','V1'],1) #Remove Id
    #original = original.drop(['chemId'],1) #Remove Id    
    return original

train = merge_feamat(train,feamat)
test = merge_feamat(test,feamat)


#remove columns have 1 uniquene value
#print(train.nunique())

for col in train.columns:
    if (len((train[col].unique()))==1):
        train.drop(col,inplace=True,axis=1)
        test.drop(col,inplace=True,axis=1)


#Try finding correlation using Pearson --- not effective --- remove
# import seaborn as sb

# corr = train.corrwith(train.Expected).abs()
# #sb.heatmap(corr.to_frame())
# train.drop(corr[corr < 0.07].index,inplace=True,axis=1)
# test.drop(corr[corr < 0.07].index,inplace=True,axis=1)


#x.index[np.isinf(x).any(1)]
#test.index[np.isinf(test).any(1)]
#Drop because contain inf value
#print(train.columns.to_series()[np.isinf(train).any()])
#print(test.columns.to_series()[np.isinf(test).any()])
#train.drop('V15',1,inplace=True)
#test.drop('V15',1,inplace=True)

#replace inf value with 0
train.replace(np.inf,0,inplace=True)
test.replace(np.inf,0,inplace=True)


#split data
y = train.Expected
x = train.drop('Expected',1)


#normalize
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

x_scaled = pd.DataFrame(scaler.fit_transform(x),index = x.index,columns = x.columns)
test_scaled = pd.DataFrame(scaler.transform(test),index=test.index,columns = test.columns)


#Select K best - using ANOVA or Chi^2
from sklearn.feature_selection import SelectKBest,chi2,f_classif

selectedFeatures = SelectKBest(score_func=f_classif,k=50)
sel = selectedFeatures.fit(x_scaled,y)


#Visualize the score:

# dfscores = pd.DataFrame(sel.scores_)
# dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.nlargest(50,'Score'))


x_fs=sel.transform(x_scaled)
test_fs=sel.transform(test_scaled)


#Split for validation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_fs,y,test_size=0.20,random_state=42)


# instantiating the random over sampler 

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy=0.5,random_state=0)
# resampling X, y
x_train, y_train = ros.fit_resample(x_train, y_train)
# new class distribution 
#print(Counter(y_train))


# instantiating the random under sampler --- not effective --- remove

# from imblearn.under_sampling import RandomUnderSampler

# rus = RandomUnderSampler()
# # resampling X, y
# x_train, y_train = rus.fit_resample(x_train, y_train)
# # new class distribution 
# print(Counter(y_train))


# tried over sampler with SMOTE
# from imblearn.over_sampling import SMOTE

# smote = SMOTE()
# # resampling X, y
# x_train, y_train = smote.fit_resample(x_train, y_train)
# # new class distribution 
# print(Counter(y_train))


# print(y_train.value_counts())
# print(y_test.value_counts())


#---------------------------------------------------------------------------
#------Model training/predicting--------------------------------------------
#---------------------------------------------------------------------------
# Comment out all model with low result

from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import regularizers
from keras import metrics as kmetrics

from sklearn import metrics

def getSubmission(model,name):
    result = model.predict(test_fs)
    df = pd.DataFrame({'Id':test_id,'Predicted':result})
    #df.Predicted +=1
    df.to_csv('result/'+name+'_submission.csv',index=False)


# #Try CNN
# x_partial_train,x_validation,y_partial_train,y_validation= train_test_split(x_train,y_train,test_size=0.2,random_state=42)

# model=models.Sequential()
# model.add(layers.Dense(20,activation='relu',input_shape=(20,)))
# model.add(layers.Dense(1,activation='sigmoid'))
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# model.fit(x_partial_train,y_partial_train,epochs=20,batch_size=5,validation_data=(x_validation,y_validation))
# print("score on test: " + str(model.evaluate(x_test,y_test)[1]))
# print("score on train: "+ str(model.evaluate(x_train,y_train)[1]))
# print(metrics.classification_report(model.predict(x_test),y_test)) #85%
# print(metrics.classification_report(model.predict(x_train),y_train)) #85%






#Try with random forrest
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFECV
# #rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=200,oob_score=True,criterion='gini',max_depth=10)
# rf = RandomForestClassifier(random_state=42,n_estimators=80,oob_score=True,max_depth=30,class_weight="balanced")

# rf.fit(x_train,y_train)
# print(metrics.classification_report(rf.predict(x_test),y_test)) #85%
# print(metrics.classification_report(rf.predict(x_train),y_train)) #85%


# getSubmission(rf,'randomforest_classweight_balanced') #49%


#Try with decision tree
# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier(max_depth=30)
# dtc.fit(x_train,y_train)
# print(metrics.classification_report(dtc.predict(x_test),y_test)) #85%
# print(metrics.classification_report(dtc.predict(x_train),y_train)) #85%

# getSubmission(dtc,'decision_tree') #49%


#Try AdaBoost
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.datasets import make_classification
# ada = AdaBoostClassifier(n_estimators=500, random_state=42,algorithm='SAMME')
# ada.fit(x_train, y_train)

# print(metrics.classification_report(ada.predict(x_test),y_test)) #55%
# print(metrics.classification_report(ada.predict(x_train),y_train)) #56%

#Try sklearn gradientboosting
# from sklearn.ensemble import GradientBoostingClassifier
# gbc = GradientBoostingClassifier(n_estimators=400, learning_rate=0.01,
# max_depth=12, random_state=42).fit(x_train, y_train)

# print(metrics.classification_report(gbc.predict(x_test),y_test)) #78%
# print(metrics.classification_report(gbc.predict(x_train),y_train)) #97%

# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold

# params = {'n_estimators':[200,300],
# 'learning_rate':[0.08,0.1,0.12],
# 'max_depth':[8,9,10]}

# rds_gbc= RandomizedSearchCV(estimator=GradientBoostingClassifier(),param_distributions=params,n_iter=20,scoring='f1_macro',n_jobs=3,cv=StratifiedKFold(4),verbose=2, random_state=42)
# rds_gbc.fit(x_train,y_train)
# print(metrics.classification_report(rds_gbc.predict(x_test),y_test)) #77%
# print(metrics.classification_report(rds_gbc.predict(x_train),y_train)) #94%


# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold

# params = {'n_estimators':[200,300],
# 'learning_rate':[0.08,0.1,0.12],
# 'max_depth':[8,9,10]}

# grd_gbc= GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=params,scoring='f1_macro',n_jobs=3,cv=StratifiedKFold(5),verbose=3)
# grd_gbc.fit(x_train,y_train)
# print(metrics.classification_report(grd_gbc.predict(x_test),y_test)) #79%
# print(metrics.classification_report(grd_gbc.predict(x_train),y_train)) #95%


# print(rds_gbc.best_params_)


# getSubmission(gbc,'gradientboosting_79') #49%


# Try xgboost - selected model
import xgboost as xgb
xgbc = xgb.XGBClassifier(n_estimators=10000,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method='gpu_hist',
    eval_metric=['error'],
    num_parallel_tree=4,
    max_depth=14,objective='binary:logistic',random_state=42)

print('------Training Start-------')
xgbc.fit(x_train,y_train,
            verbose=True,
            early_stopping_rounds=50,
            eval_metric=['error'],
            eval_set=[(x_test,y_test)])
print('result on test set:')
print(metrics.classification_report(xgbc.predict(x_test),y_test,digits=5)) #80.29%
print('result on train set:')
print(metrics.classification_report(xgbc.predict(x_train),y_train,digits=5)) #95%


#---------------------------------------------------------
#Randomize search for xgbc. 
#Comment out because I finded the best hyperparameter and fixed it above.
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv--------

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import StratifiedKFold

# params = {'n_estimators':[1000],
#     'learning_rate':[0.1],
#     'subsample':[0.7,0.8,0.9,1.0],
#     'colsample_bytree':[0.7,0.8,0.9,1.0]}
# fit_params={"early_stopping_rounds":50, 
#             "eval_metric" : "logloss", 
#             "eval_set" : [(x_test,y_test)]}


# #grd_xgbc= GridSearchCV(estimator=xgbc,param_grid=params,scoring='f1_macro',n_jobs=3,verbose=3)
# #grd_xgbc.fit(x_train,y_train)

# rs_xgbc= RandomizedSearchCV(xgbc,params,n_iter=15,cv=5,scoring='f1_macro',n_jobs=4,verbose=3)
# rs_xgbc.fit(x_train,y_train,**fit_params)


# rs_xgbc.cv_results_


# pd.concat([pd.DataFrame(grd_xgbc.cv_results_["params"]),pd.DataFrame(grd_xgbc.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
# pd.concat([pd.DataFrame(rs_xgbc.cv_results_["params"]),pd.DataFrame(rs_xgbc.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)


# print(metrics.classification_report(grd_xgbc.predict(x_test),y_test,digits=5)) #79%
# print(metrics.classification_report(grd_xgbc.predict(x_train),y_train,digits=5)) #95%


getSubmission(xgbc,'FinalResult_xgbc_rs_8050')
print('----Result prediction saved to result/FinalResult_xgbc_rs_8029.csv----')



#try with KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
# knn.fit(x_train, y_train)


# print(metrics.classification_report(knn.predict(x_test),y_test,digits=5)) #85%
# print(metrics.classification_report(knn.predict(x_train),y_train,digits=5)) #85%

#try with LinearSVC
# from sklearn.svm import LinearSVC
# svm=LinearSVC(C=0.0001)
# svm.fit(x_train, y_train)

# print(metrics.classification_report(svm.predict(x_test),y_test))
# print(metrics.classification_report(svm.predict(x_train),y_train))

#Try recursive features elimination
# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import StratifiedKFold
# #rfecv = RFECV(estimator=RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=80,oob_score=True,criterion='gini',max_depth=5),step=1,n_jobs=3, cv=StratifiedKFold(10), scoring='f1_macro',verbose=10)
# rfecv = RFECV(estimator=DecisionTreeClassifier(max_depth=20),step=1,n_jobs=3, cv=StratifiedKFold(5), scoring='f1_macro',verbose=10)
# rfecv.fit(x_train,y_train)


# print(metrics.classification_report(rfecv.predict(x_test),y_test)) #85%
# print(metrics.classification_report(rfecv.predict(x_train),y_train)) #85%

# getSubmission(rfecv,'feature_elimination_balanced') #47%


# print('Optimal number of features: {}'.format(rfecv.n_features_))





