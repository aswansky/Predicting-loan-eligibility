# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:21:45 2022

@author: abdal
"""

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import stattools
from statsmodels.tools.tools import categorical
import scipy
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import dill
import statistics as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
data_source=pd.read_csv(r'D:\Documents\python\advanced data analysis using python\ML\test coding\projects\loan data\Loan_Data.csv',on_bad_lines='skip')

#Data load

data=pd.read_csv(r'D:\Documents\python\advanced data analysis using python\ML\test coding\projects\loan data\Loan_Data.csv',on_bad_lines='skip')
desc=pd.DataFrame(data.describe())

#data coding 
# for each variable apply categorical function. for ex...
(Loan_Status,Loan_Status_dict)=categorical(data,["Loan_Status"],dictnames=True,drop=True)


data=data.select_dtypes(include=("number"))
data=pd.concat([data,Education_cat,Gender_cat,Loan_Status,Married_cat,Property_Area],axis=1)

#data missing fill
llist=list(data.columns)
 
for i,var in enumerate(llist):
   code=compile("data","["+llist[i]+"]","eval")
   m=(st.mode(eval(code).iloc[:,i]))
   data.iloc[:,i]=data.iloc[:,i].fillna(m)
    

#dependent and independent variable selection

dataX=data.drop(['Y','N'],axis=1)
dataY=pd.DataFrame(data.Y)
dataX_train,dataX_test,dataY_train,dataY_test=train_test_split(dataX,dataY,test_size=.2)


#Naive Bayes with bootstrap
#Boostrap
data_train=pd.concat([dataX_train,dataY_train],axis=1)
data_train_select=data_train[data_train['Y']==0]
data_train_selected=data_train_select.sample(n=189,replace=True)

dataY_train.reset_index(inplace=True,drop=True)

dataX_train=pd.concat([dataX_train,data_train_selected.drop('Y',axis=1)],axis=0)
dataY_train=pd.concat([dataY_train,pd.DataFrame(data_train_selected['Y'])],axis=0)

bayes=MultinomialNB()
bayes.fit(dataX_train,dataY_train)
prediction=bayes.predict(dataX_test)
print('accuracy score of naive bayes is',format(accuracy_score(dataY_test, prediction)) )
print('precision score of naive bayes is',format(precision_score(dataY_test, prediction)))
print('F1 score of naive bayes is',format(f1_score(dataY_test, prediction)))
#accuracy score of naive bayes is 0.43902439024390244
#precision score of naive bayes is 0.43902439024390244
#F1 score of naive bayes is 0.5174825174825175

#naive bayes with standardized X
scaler=MinMaxScaler()

zdataX_train=scaler.fit_transform(dataX_train)
zdataX_test=scaler.fit_transform(dataX_test)
bayesz=MultinomialNB()
bayesz.fit(zdataX_train,dataY_train)
predictionz=bayesz.predict(zdataX_test)
print('accuracy score of naive bayes is',format(accuracy_score(dataY_test, predictionz)) )
print('precision score of naive bayes is',format(precision_score(dataY_test, predictionz)))
print('recall score of naive bayes is',format(recall_score(dataY_test, predictionz)))
print('F1 score of naive bayes is',format(f1_score(dataY_test, predictionz)))
#recommendations. use transfromation and bootstrap enhance the results
#accuracy score of naive bayes is 0.7073170731707317
#precision score of naive bayes is 0.7203389830508474
#recall score of naive bayes is 0.9659090909090909
#F1 score of naive bayes is 0.825242718446602

#naive bayes with standardized X and bootstrap
#Boostrap
data_train=pd.concat([dataX_train,dataY_train],axis=1)
data_train_select=data_train[data_train['Y']==0]
data_train_selected=data_train_select.sample(n=189,replace=True)

dataY_train.reset_index(inplace=True,drop=True)

dataX_train=pd.concat([dataX_train,data_train_selected.drop('Y',axis=1)],axis=0)
dataY_train=pd.concat([dataY_train,pd.DataFrame(data_train_selected['Y'])],axis=0)


scaler=MinMaxScaler()

zdataX_train=scaler.fit_transform(dataX_train)
zdataX_test=scaler.fit_transform(dataX_test)
bayesz=MultinomialNB()
bayesz.fit(zdataX_train,dataY_train)
predictionz=bayesz.predict(zdataX_test)
print('accuracy score of naive bayes is',format(accuracy_score(dataY_test, predictionz)) )
print('precision score of naive bayes is',format(precision_score(dataY_test, predictionz)))
print('recall score of naive bayes is',format(recall_score(dataY_test, predictionz)))
print('F1 score of naive bayes is',format(f1_score(dataY_test, predictionz)))
#accuracy score of naive bayes is 0.6016260162601627
#precision score of naive bayes is 0.7936507936507936
#recall score of naive bayes is 0.5813953488372093
#F1 score of naive bayes is 0.6711409395973155

#Decision tree classifier with both bootstrap and transform
#Boostrap
data_train=pd.concat([dataX_train,dataY_train],axis=1)
data_train_select=data_train[data_train['Y']==0]
data_train_selected=data_train_select.sample(n=189,replace=True)

dataY_train.reset_index(inplace=True,drop=True)

dataX_train=pd.concat([dataX_train,data_train_selected.drop('Y',axis=1)],axis=0)
dataY_train=pd.concat([dataY_train,pd.DataFrame(data_train_selected['Y'])],axis=0)


scaler=MinMaxScaler()

zdataX_train=scaler.fit_transform(dataX_train)
zdataX_test=scaler.fit_transform(dataX_test)
tree1=DecisionTreeClassifier(max_depth=15,min_samples_split=10,min_samples_leaf=10)
tree1.fit(zdataX_train,dataY_train)
zpredictions2=tree1.predict(zdataX_test)
print('accuracy score of deision tree is',format(accuracy_score(dataY_test, zpredictions2)) )
print('precision score of deision tree  is',format(precision_score(dataY_test, zpredictions2)))
print('recall score of deision tree  is',format(recall_score(dataY_test, zpredictions2)))
print('F1 score of deision tree  is',format(f1_score(dataY_test, zpredictions2)))
#accuracy score of deision tree is 0.6910569105691057
#precision score of deision tree is 0.7681159420289855
#recall score of deision tree is 0.7066666666666667
#F1 score of deision tree is 0.736111111111111

#Decision tree classifier with no bootstrap no transform
tree2=DecisionTreeClassifier(max_depth=10,min_samples_split=15,min_samples_leaf=3)
tree2.fit(dataX_train,dataY_train)
predictions3=tree2.predict(dataX_test)
print('accuracy score of deision tree is',format(accuracy_score(dataY_test, predictions3)) )
print('precision score of deision tree  is',format(precision_score(dataY_test, predictions3)))
print('recall score of deision tree  is',format(recall_score(dataY_test, predictions3)))
print('F1 score of deision tree  is',format(f1_score(dataY_test, predictions3)))
#accuracy score of deision tree is 0.7154471544715447
#precision score of deision tree is 0.810126582278481
#recall score of deision tree is 0.7619047619047619
#F1 score of deision tree is 0.7852760736196319

#Decision tree classifier with  bootstrap no transform
#Boostrap
data_train=pd.concat([dataX_train,dataY_train],axis=1)
data_train_select=data_train[data_train['Y']==0]
data_train_selected=data_train_select.sample(n=189,replace=True)

dataY_train.reset_index(inplace=True,drop=True)

dataX_train=pd.concat([dataX_train,data_train_selected.drop('Y',axis=1)],axis=0)
dataY_train=pd.concat([dataY_train,pd.DataFrame(data_train_selected['Y'])],axis=0)

tree3=DecisionTreeClassifier(max_depth=10,min_samples_split=15,min_samples_leaf=3)
tree3.fit(dataX_train,dataY_train)
predictions4=tree3.predict(dataX_test)
print('accuracy score of deision tree is',format(accuracy_score(dataY_test, predictions4)) )
print('precision score of deision tree  is',format(precision_score(dataY_test, predictions4)))
print('recall score of deision tree  is',format(recall_score(dataY_test, predictions4)))
print('F1 score of deision tree  is',format(f1_score(dataY_test, predictions4)))
#accuracy score of deision tree is 0.7154471544715447
#precision score of deision tree is 0.7976190476190477
#recall score of deision tree is 0.788235294117647
#F1 score of deision tree is 0.7928994082840236

#Decision tree classifier with  no bootstrap but transform only
scaler1=MinMaxScaler()

zdataX_train=scaler1.fit_transform(dataX_train)
zdataX_test=scaler1.fit_transform(dataX_test)

tree4=DecisionTreeClassifier(max_depth=5,min_samples_split=10,min_samples_leaf=15)
tree4.fit(zdataX_train,dataY_train)
predictions5=tree4.predict(zdataX_test)
print('accuracy score of deision tree is',format(accuracy_score(dataY_test, predictions5)) )
print('precision score of deision tree  is',format(precision_score(dataY_test, predictions5)))
print('recall score of deision tree  is',format(recall_score(dataY_test, predictions5)))
print('F1 score of deision tree  is',format(f1_score(dataY_test, predictions5)))
#accuracy score of deision tree is 0.8048780487804879
#precision score of deision tree is 0.7777777777777778
#recall score of deision tree is 0.9746835443037974
#F1 score of deision tree is 0.8651685393258427


#SVM with no transform no bootstrap
svc1=SVC()
svc1.fit(dataX_train,dataY_train)
svcpredictions=svc1.predict(dataX_test)
print('accuracy score of svc is',format(accuracy_score(dataY_test, svcpredictions)) )
print('precision score of svc  is',format(precision_score(dataY_test, svcpredictions)))
print('recall score of svc  is',format(recall_score(dataY_test, svcpredictions)))
print('F1 score of svc  is',format(f1_score(dataY_test, svcpredictions)))
#accuracy score of svc is 0.6910569105691057
#precision score of svc is 0.6910569105691057
#recall score of svc is 1.0
#F1 score of svc is 0.8173076923076924

#SVM with    bootstrap only
#Boostrap
data_train=pd.concat([dataX_train,dataY_train],axis=1)
data_train_select=data_train[data_train['Y']==0]
data_train_selected=data_train_select.sample(n=189,replace=True)

dataY_train.reset_index(inplace=True,drop=True)

dataX_train=pd.concat([dataX_train,data_train_selected.drop('Y',axis=1)],axis=0)
dataY_train=pd.concat([dataY_train,pd.DataFrame(data_train_selected['Y'])],axis=0)

svc2=SVC()
svc2.fit(dataX_train,dataY_train)
svcpredictions=svc2.predict(dataX_test)
print('accuracy score of svc is',format(accuracy_score(dataY_test, svcpredictions)) )
print('precision score of svc  is',format(precision_score(dataY_test, svcpredictions)))
print('recall score of svc  is',format(recall_score(dataY_test, svcpredictions)))
print('F1 score of svc  is',format(f1_score(dataY_test, svcpredictions)))
#accuracy score of svc is 0.22764227642276422
#precision score of svc is .5
#recall score of svc is 0.031578947368421054
#F1 score of svc is 0.05940594059405941

#SVM with   transform only

scaler1=MinMaxScaler()

zdataX_train=scaler1.fit_transform(dataX_train)
zdataX_test=scaler1.fit_transform(dataX_test)

svc3=SVC()
svc3.fit(zdataX_train,dataY_train)
svcpredictions=svc3.predict(zdataX_test)

print('accuracy score of svc is',format(accuracy_score(dataY_test, svcpredictions)) )
print('precision score of svc  is',format(precision_score(dataY_test, svcpredictions)))
print('recall score of svc  is',format(recall_score(dataY_test, svcpredictions)))
print('F1 score of svc  is',format(f1_score(dataY_test, svcpredictions)))
#accuracy score of svc is 0.8211382113821138
#precision score of svc is 0.7941176470588235
#recall score of svc is 0.9878048780487805
#F1 score of svc is 0.8804347826086957

#SVM with both bootstrap   transform only
#Boostrap
data_train=pd.concat([dataX_train,dataY_train],axis=1)
data_train_select=data_train[data_train['Y']==0]
data_train_selected=data_train_select.sample(n=189,replace=True)

dataY_train.reset_index(inplace=True,drop=True)

dataX_train=pd.concat([dataX_train,data_train_selected.drop('Y',axis=1)],axis=0)
dataY_train=pd.concat([dataY_train,pd.DataFrame(data_train_selected['Y'])],axis=0)


scaler1=MinMaxScaler()

zdataX_train=scaler1.fit_transform(dataX_train)
zdataX_test=scaler1.fit_transform(dataX_test)

svc4=SVC()
svc4.fit(zdataX_train,dataY_train)
svcpredictions=svc4.predict(zdataX_test)

print('accuracy score of svc is',format(accuracy_score(dataY_test, svcpredictions)) )
print('precision score of svc  is',format(precision_score(dataY_test, svcpredictions)))
print('recall score of svc  is',format(recall_score(dataY_test, svcpredictions)))
print('F1 score of svc  is',format(f1_score(dataY_test, svcpredictions)))
#accuracy score of svc is 0.7317073170731707
#precision score of svc is 0.8172043010752689
#recall score of svc is  0.8260869565217391
#F1 score of svc is 0.8260869565217391


#Random forest with no bootstrap no transform

rf01 = RandomForestClassifier(n_estimators = 100,
criterion="gini").fit(dataX_train,dataY_train)
rf01predictions=rf01.predict(dataX_test)

print('accuracy score of scv is',format(accuracy_score(dataY_test, rf01predictions)) )
print('precision score of scv  is',format(precision_score(dataY_test, rf01predictions)))
print('recall score of scv  is',format(recall_score(dataY_test, rf01predictions)))
print('F1 score of scv  is',format(f1_score(dataY_test, rf01predictions)))
#accuracy score of svc is 0.7073170731707317
#precision score of svc is 0.7473684210526316
#recall score of svc is  0.8554216867469879
#F1 score of svc is 0.797752808988764

#Random forest with  bootstrap
#Boostrap
data_train=pd.concat([dataX_train,dataY_train],axis=1)
data_train_select=data_train[data_train['Y']==0]
data_train_selected=data_train_select.sample(n=189,replace=True)

dataY_train.reset_index(inplace=True,drop=True)

dataX_train=pd.concat([dataX_train,data_train_selected.drop('Y',axis=1)],axis=0)
dataY_train=pd.concat([dataY_train,pd.DataFrame(data_train_selected['Y'])],axis=0)


rf02 = RandomForestClassifier(n_estimators = 100,
criterion="gini").fit(dataX_train,dataY_train)
rf02predictions=rf02.predict(dataX_test)

print('accuracy score of scv is',format(accuracy_score(dataY_test, rf02predictions)) )
print('precision score of scv  is',format(precision_score(dataY_test, rf02predictions)))
print('recall score of scv  is',format(recall_score(dataY_test, rf02predictions)))
print('F1 score of scv  is',format(f1_score(dataY_test, rf02predictions)))
#accuracy score of svc is 0.7398373983739838
#precision score of svc is 0.7934782608695652
#recall score of svc is  0.8488372093023255
#F1 score of svc is 0.8202247191011235

#Random forest with  transform
scaler=MinMaxScaler()

zdataX_train=scaler.fit_transform(dataX_train)
zdataX_test=scaler.fit_transform(dataX_test)

rf03= RandomForestClassifier(n_estimators = 100,
criterion="gini").fit(zdataX_train,dataY_train)
rf03predictions=rf03.predict(zdataX_test)

print('accuracy score of svc is',format(accuracy_score(dataY_test, rf03predictions)) )
print('precision score of svc  is',format(precision_score(dataY_test, rf03predictions)))
print('recall score of svc  is',format(recall_score(dataY_test, rf03predictions)))
print('F1 score of svc  is',format(f1_score(dataY_test, rf03predictions)))
#accuracy score of svc is 0.7642276422764228
#precision score of svc is 0.8105263157894737
#recall score of svc is  0.875
#F1 score of svc is 0.8415300546448087

#Random forest with  transform and bootstrap
#Boostrap
data_train=pd.concat([dataX_train,dataY_train],axis=1)
data_train_select=data_train[data_train['Y']==0]
data_train_selected=data_train_select.sample(n=189,replace=True)

dataY_train.reset_index(inplace=True,drop=True)

dataX_train=pd.concat([dataX_train,data_train_selected.drop('Y',axis=1)],axis=0)
dataY_train=pd.concat([dataY_train,pd.DataFrame(data_train_selected['Y'])],axis=0)


scaler=MinMaxScaler()

zdataX_train=scaler.fit_transform(dataX_train)
zdataX_test=scaler.fit_transform(dataX_test)

rf04= RandomForestClassifier(n_estimators = 100,
criterion="gini").fit(zdataX_train,dataY_train)
rf04predictions=rf04.predict(zdataX_test)

print('accuracy score of Random forest is',format(accuracy_score(dataY_test, rf04predictions)) )
print('precision score of Random forest  is',format(precision_score(dataY_test, rf04predictions)))
print('recall score of Random forest  is',format(recall_score(dataY_test, rf04predictions)))
print('F1 score of Random forest  is',format(f1_score(dataY_test, rf04predictions)))
#accuracy score of Random forest is 0.8211382113821138
#precision score of Random forest is 0.8651685393258427
#recall score of Random forest is  0.8850574712643678
#F1 score of Random forest is 0.875



########### TUNING RANDOM FOREST WITH FEATURE TRANSFORMATION
""" tuned paramters 
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
"""
n_estimators=[int(x) for x in np.linspace(200,2000,10)]
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(10, 100,11)]
min_samples_split=[2,5,7,10] 
min_samples_leaf=[1, 2, 4]
bootstrap = [True, False]
criterion=['gini','entropy']

#create random grid
random_grid={'n_estimators':n_estimators,'max_features':max_features,
             'max_depth':max_depth,'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf,'bootstrap':bootstrap,'criterion':criterion}

rf=RandomForestClassifier()
rf_random=RandomizedSearchCV(rf, random_grid,n_iter=100,cv=3,verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(zdataX_train,dataY_train)
tf=rf_random.best_estimator_
#tf=RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_leaf=4,
#                       min_samples_split=5, n_estimators=1000)
rfpredictions=tf.predict(zdataX_test)

print('accuracy score of Random forest is',format(accuracy_score(dataY_test, rfpredictions)) )
print('precision score of Random forest  is',format(precision_score(dataY_test, rfpredictions)))
print('recall score of Random forest  is',format(recall_score(dataY_test, rfpredictions)))
print('F1 score of Random forest  is',format(f1_score(dataY_test, rfpredictions)))

#accuracy score of svc is 0.7886178861788617
#precision score of svc is 0.7788461538461539
#recall score of svc is  0.9642857142857143
#F1 score of svc is 0.8617021276595744

