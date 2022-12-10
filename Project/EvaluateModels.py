#To prevent Kernel issues, the models had to be run in the terminal. Therefore, I am creating this .py file to allow myself to run and evaluate
#the models. This file will define model parameters and run the models to identify the best model for predicting the winning team in an NFL game.

#### importing libraries and important functions
import boto3
import pandas as pd; pd.set_option('display.max_column', 100)
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,  GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from itertools import product




#### reading csv file of final data (refer to VariableImportance.ipynb)
games = pd.read_csv('games_Final.csv')

#split into input and output
x = games.drop(columns = 'home_win') 
y = games['home_win']




#### defining dataframes to hold parameters for models
def expand_grid(dictionary1):
    return pd.DataFrame([row for row in product(*dictionary1.values())], 
                        columns = dictionary1.keys())
#decision tree
dt_dict = {'depth':[3,5,7,9]}
dt_params = expand_grid(dt_dict)

#random forest
rf_dict = {'n_tree': [100, 300, 500, 1000], 'depth': [3, 5, 7, 9]}
rf_params = expand_grid(rf_dict)

#ada and gradient boost 
ag_dict = {'n_tree': [100, 300, 500, 1000], 'depth': [3, 5, 7, 9], 'learning_rate': [0.1, 0.01, 0.001]}
ag_params = expand_grid(ag_dict)

#support vector machine
svm_dict = {'kernal':['rbf','poly']}
svm_params = expand_grid(svm_dict)




#### evaluating models and storing results
#list for results of each model
lr_results = []
svm_results = []
dt_results = []
rf_results = []
ada_results = []
gb_results = []

#list of cut off values to evaluate
cutoff = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

for j in range(0,100):
    #splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)

    ##scale the data for models that require scaling
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    
    ## logistic regression model
    #build model
    lr_md=LogisticRegression().fit(x_train_scaled, y_train)
    #predicting on test set
    lr_preds = lr_md.predict_proba(x_test_scaled)[:,1]
    #loop to evaluate cutoff:
    for k in cutoff:
        #change likelihoods to labels with k cutoff
        lr_label = np.where(lr_preds<k,0,1)
        #append results
        lr_results.append([k,accuracy_score(y_test,lr_label),recall_score(y_test, lr_label),f1_score(y_test, lr_label)])
    #store results in dataframe
    lr_df = pd.DataFrame(columns = ['cut_off','accuracy','recall','f1'], data = lr_results)
    #adding an overall score for model evaluation (will be the average of recall, accuracy, and f1)
    lr_df['overall_score'] = ((lr_df['accuracy']+lr_df['recall']+lr_df['f1'])/3)
    
    ## support vector machine model
    n = svm_params.shape[0]
    for i in range(0,n):
        #build model
        svm_md = SVC(kernel = svm_params.loc[i, 'kernal'], probability = True).fit(x_train_scaled, y_train)
        #predicting on test set
        svm_preds = svm_md.predict_proba(x_test_scaled)[:,1]
        #loop to evaluate cutoff:
        for k in cutoff:
            #change likelihoods to labels with k cutoff
            svm_label = np.where(svm_preds<k,0,1)
            #append results
            svm_results.append([svm_params.loc[i, 'kernal'],k,accuracy_score(y_test,svm_label),recall_score(y_test, svm_label),f1_score(y_test, svm_label)])
    #store results in dataframe
    svm_df = pd.DataFrame(columns = ['kernal','cut_off','accuracy','recall','f1'], data = svm_results)
    #adding an overall score for model evaluation (will be the average of recall, accuracy, and f1)
    svm_df['overall_score'] = ((svm_df['accuracy']+svm_df['recall']+svm_df['f1'])/3)
    
    ## decision tree
    n = dt_params.shape[0]
    for i in range(0,n):
        #build model
        dt_md = DecisionTreeClassifier(max_depth = dt_params.loc[i, 'depth']).fit(x_train,y_train)
        #predicting on test set
        dt_preds = dt_md.predict_proba(x_test)[:,1]
        #loop to evaluate cutoff:
        for k in cutoff:
            #change likelihoods to labels with k cutoff
            dt_label = np.where(dt_preds<k,0,1)
            #append results
            dt_results.append([dt_params.loc[i, 'depth'],k,accuracy_score(y_test,dt_label),recall_score(y_test, dt_label),f1_score(y_test, dt_label)])
    #store results in dataframe
    dt_df = pd.DataFrame(columns = ['depth','cut_off','accuracy','recall','f1'], data = dt_results)
    #adding an overall score for model evaluation (will be the average of recall, accuracy, and f1)
    dt_df['overall_score'] = ((dt_df['accuracy']+dt_df['recall']+dt_df['f1'])/3)
    
    ## random forest
    n = rf_params.shape[0]
    for i in range(0,n):
        #build model
        rf_md = RandomForestClassifier(max_depth = rf_params.loc[i, 'depth'], n_estimators =  rf_params.loc[i, 'n_tree']).fit(x_train,y_train)
        #predicting on test set
        rf_preds = rf_md.predict_proba(x_test)[:, 1]
        #loop to evaluate cutoff:
        for k in cutoff:
            #change likelihoods to labels with k cutoff
            rf_label = np.where(rf_preds<k,0,1)
            #append results
            rf_results.append([rf_params.loc[i, 'depth'],rf_params.loc[i, 'n_tree'],k,accuracy_score(y_test,rf_label),recall_score(y_test, rf_label),f1_score(y_test, rf_label)])
    #store results in dataframe
    rf_df = pd.DataFrame(columns = ['depth','n_tree','cut_off','accuracy','recall','f1'], data = rf_results)
    #adding an overall score for model evaluation (will be the average of recall, accuracy, and f1)
    rf_df['overall_score'] = ((rf_df['accuracy']+rf_df['recall']+rf_df['f1'])/3)
    
    ##ada boost model
    n = ag_params.shape[0]
    for i in range(0,n):
        #build model
        ada_md = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = ag_params.loc[i, 'depth']), n_estimators =  ag_params.loc[i, 'n_tree'], learning_rate =ag_params.loc[i, 'learning_rate']).fit(x_train,y_train)
        #predicting on test set
        ada_preds = ada_md.predict_proba(x_test)[:, 1]
        #loop to evaluate cutoff:
        for k in cutoff:
            #change likelihoods to labels with k cutoff
            ada_label = np.where(ada_preds<k,0,1)
            #append results
            ada_results.append([ag_params.loc[i, 'depth'],ag_params.loc[i, 'n_tree'],ag_params.loc[i, 'learning_rate'], k,accuracy_score(y_test,ada_label),recall_score(y_test, ada_label),f1_score(y_test, ada_label)])
    #store results in dataframe
    ada_df = pd.DataFrame(columns = ['depth','n_tree','learning_rate','cut_off','accuracy','recall','f1'], data = ada_results)
    #adding an overall score for model evaluation (will be the average of recall, accuracy, and f1)
    ada_df['overall_score'] = ((ada_df['accuracy']+ada_df['recall']+ada_df['f1'])/3)
    
    ##gradient boost model
    n = ag_params.shape[0]
    for i in range(0,n):
        #build model
        gb_md = GradientBoostingClassifier(max_depth = ag_params.loc[i, 'depth'], n_estimators =  ag_params.loc[i, 'n_tree'], learning_rate =ag_params.loc[i, 'learning_rate']).fit(x_train,y_train)
        #predicting on test set
        gb_preds = gb_md.predict_proba(x_test)[:, 1]
        #loop to evaluate cutoff:
        for k in cutoff:
            #change likelihoods to labels with k cutoff
            gb_label = np.where(gb_preds<k,0,1)
            #append results
            gb_results.append([ag_params.loc[i, 'depth'],ag_params.loc[i, 'n_tree'],ag_params.loc[i, 'learning_rate'], k,accuracy_score(y_test,gb_label),recall_score(y_test, gb_label),f1_score(y_test, gb_label)])
    #store results in dataframe
    gb_df = pd.DataFrame(columns = ['depth','n_tree','learning_rate','cut_off','accuracy','recall','f1'], data = gb_results)
    #adding an overall score for model evaluation (will be the average of recall, accuracy, and f1)
    gb_df['overall_score'] = ((gb_df['accuracy']+gb_df['recall']+gb_df['f1'])/3)



#extract results for all models   
lr_df.to_csv('lr_results.csv', index = False)
svm_df.to_csv('svm_results.csv', index = False)
dt_df.to_csv('dt_results.csv', index = False)
rf_df.to_csv('rf_results.csv', index = False)
ada_df.to_csv('ada_results.csv', index = False)
gb_df.to_csv('gb_results.csv', index = False)