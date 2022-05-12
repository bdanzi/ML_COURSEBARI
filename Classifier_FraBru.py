# ClassifierTest.py

## KFJ 4MAY2022 @ BARI - EXAMPLE CODE FOR MLP CLASSIFIER TESTS
## Import training dataset, randomize, normalize & store in array.
## For training have another array available into which a subset of
##  training data is written for use by MLP.  A non-intersecting portion 
##  is stored to be the "Test" set.
## Train MLP sequentially with 50, 100, 150, ... events... and save results.
## then a check on the mlp prediction (mlp score) and 
#  on the mean square error loss function for train & test data depending on training set size.

import os, sys          # Need this to send messages to OS
import numpy as np      # numpy is numerical python
import pandas as pd
from pandas import read_csv  ## convenient to read in a dataset
from array import*
from matplotlib import pyplot as plt

## Following are libraries for ML:
from sklearn.preprocessing import StandardScaler # Re-scales inputs to mean 0 std 1.
from sklearn.neural_network import MLPClassifier # This is the code to make a neural net.
from sklearn.model_selection import train_test_split  # Splits the dataset randomly into train & test
from sklearn.metrics import plot_roc_curve,roc_auc_score,roc_curve, auc  ## To show results
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

## Read in dataset
infile = 'ClassTrainingData5SigFigs.txt'    ## 5-digit float;14980 sigs,  15040  baks
targCol = 0                   # In this dataset the "targets" or "labels" are in column 0.
lst = list(range(24))

# Read dataset to pandas dataframe and turn it into a 2-D array.
Hdata = read_csv( infile, delimiter=' ', header=None ,  usecols=lst ) # Use panda's read_csv to read in the data from txt file...
## 'usecols' must either be list-like of all strings, all unicode, all integers or a callable.
Hdata = Hdata.values         # ... but need to do this to use arithmetic
X  = Hdata[:, 1:24]          # vector of variables (features) used for MLML.
y  = Hdata[:,targCol]        # target data...
y  = y.astype('int')         #    .... must be ints.

LoopFlag = 1 ## set this to not 1 to skip looping over training subsets
testFraction = 0.2    # used by "train_test_split" to set aside data for validation
seed=7  
X_hold, X_test, y_hold, y_test = train_test_split(X, y, test_size=testFraction, random_state=seed, shuffle=True) 

## Normalise train&test datasets with same transformation.
scaler = StandardScaler()
scaler.fit(X_hold)
X_hold = scaler.transform(X_hold)
X_test = scaler.transform(X_test)

#cnt=0  ##  just checking...
#for i in range(len(y_hold)):
#    if y_hold[i] == 1 : cnt=cnt+1
#print("Data: ", cnt, X_hold.shape, X_test.shape, y_hold, y_test)

# Defaults ==>  mlp = MLPClassifier(hidden_layer_sizes=(100), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

# Create an instance of MLP
mlp = MLPClassifier(hidden_layer_sizes=(50) , activation='relu',solver='sgd', alpha=0.0001,validation_fraction=0.1, max_iter=1000,early_stopping=False ,verbose=False)
bdt = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=700
)



bdt.fit(X_hold, y_hold)

#
Chunk=50         # how many events per chunk
NumChuncks=10  #     .... times this determines number of training events used.
Scores_train=[]
mse_train=[]
mse_test=[]
Scores_test=[]
Scores_trainsize=[]


ar=["\n"]  ## jeezus, what a hack to get np.savetxt to write 1 line per array!
if LoopFlag == 1:
    for iter in range(1, NumChuncks):
        TrainSize= Chunk * iter
        y_train = np.empty((TrainSize))  ## Reserve memory for trainng data...
        y_train = y_hold[0:TrainSize]
        X_train = np.zeros((TrainSize, 23))
        for row in range(TrainSize):
            X_train[row] = X_hold[row]
        mlpfitted = mlp.fit(X_train,y_train) # Fitting procedure which determines the MLP weights by using a fraction TrainSize of training data
        y_score_test = mlpfitted.predict_proba(X_test)
        y_score_train = mlpfitted.predict_proba(X_train)
        TrainScore = mlp.score(X_train, y_train)
        TestScore  = mlp.score( X_test, y_test)
        Mse_train = mean_squared_error(y_train,y_score_train[:,1])
        Mse_test = mean_squared_error(y_test,y_score_test[:,1])
        Scores_trainsize.append(iter*1.0*Chunk)
        Scores_train.append(TrainScore)
        Scores_test.append(TestScore)
        mse_train.append(Mse_train)
        mse_test.append(Mse_test)
        print("ITER: ", iter, "TrainScore: ", "%.5f" % TrainScore, "TestScore: ", "%.5f" %  TestScore  )

#######################################################################################################
## Plotting the MLP score vs Trainsize, increasing this number in steps of Chunk number##
#######################################################################################################
plt.plot(Scores_trainsize,Scores_train,"o-",label = "Training data set")
plt.plot(Scores_trainsize,Scores_test,"o-",label = "Test data set")
## naming the x axis
plt.xlabel('Number of training samples')
## naming the y axis
plt.ylabel('MLP score')
## giving a title to my graph
plt.title('Training and test score on Training and Test samples')
## show a legend on the plot
plt.legend(['Training data set','Validation data set'])
plt.savefig('numbertrainingsamples_MLP_score.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#############################################################################################################################
## Plotting the Mean square error loss function vs size of training sample, increasing this number in steps of Chunk number##
#############################################################################################################################
plt.clf() # clean the figure
plt.plot(Scores_trainsize,mse_train,"o-",label = "Training data set")
plt.plot(Scores_trainsize,mse_test,"o-",label = "Test data set")
## naming the x axis
plt.xlabel('Number of training samples')
## naming the y axis
plt.ylabel('Mean square error')
## giving a title to my graph
plt.title('Training and test loss function on Training and Test samples')
## show a legend on the plot
plt.legend(['Training data set','Validation data set'])
plt.savefig('numbertrainingsamples_loss_mse.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# Fitting procedure which determines the MLP weights by using the whole Training dataset X_old, y_old
mlpfitted_tot = mlp.fit(X_hold,y_hold)
y_score_test = mlpfitted_tot.predict_proba(X_test) # MLP prediction on test data
y_score_training = mlpfitted_tot.predict_proba(X_hold)# MLP prediction on training data
print("FULL DATASET (MLP result): ", "TrainScore:", mlp.score(X_hold, y_hold), "TestScore: ", mlp.score(X_test, y_test))

# Compute ROC curve and ROC area for each model
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()

# Compute ROC curve for the MLP algorithm
fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test[:,1])
roc_auc_test = auc(fpr_test, tpr_test)
fpr_train, tpr_train, _ = roc_curve(y_hold, y_score_training[:,1])
roc_auc_train = auc(fpr_train, tpr_train)


##################################################################################
#					Plotting ML algorithms' ROCs for test and train data    	 #
##################################################################################
lw = 2
# second method for computing and plotting the BDT performance by plotting automatically a ROC curve for testing data
fig = plot_roc_curve(bdt, X_test, y_test,name='Test Bosteed Decision Tree',color="green") 
plot_roc_curve(bdt, X_hold, y_hold,name='Training Bosteed Decision Tree',ax = fig.ax_,color="blue")
# manual process for plotting the MLP performance
plt.plot(fpr_test,tpr_test,color="darkorange",lw=lw,label="Test MLP ROC curve (area = %0.2f)" % roc_auc_test)
plt.plot(fpr_train,tpr_train,color="purple",lw=lw,label="Train MLP ROC curve (area = %0.2f)" % roc_auc_train)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.grid()
plt.savefig('ROC_results.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


