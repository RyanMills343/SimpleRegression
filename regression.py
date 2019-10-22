import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pandas import plotting as plt

#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plot

import numpy as np

from sklearn import preprocessing as pp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics



####################################################################################################################
###### read in the data to pandas dataframe ########################################################################
####################################################################################################################
analysis = pd.read_excel('HealthInsuranceClean.xlsx',
                       header=0,                      # header is now in the first row
                       na_values='null',              # na values are "null" in excel
                       index_col=None)                # don't set an index column
analysis.drop(columns=['Fips'], inplace=True)
# look at the original data
#print("Original:", analysis.head())

# look at the distributions and scatterplot matrix of the data
#plt.scatter_matrix(analysis)
#plot.show()


####################################################################################################################
###### Prep the Data for Modeling ##################################################################################
####################################################################################################################

######################################################################################
##### 1) Normalization  ##############################################################
######################################################################################
''' http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py '''

#### A) Look at the distributions of the data #################
# Distribution of year Before Scaling
#plot.figure('original')
#plot.hist(analysis['Year'])

#### B) Scale the Data ########################################
# Create a scale object
scale = pp.MinMaxScaler()

# choose columns we want to scale
columns = ['Percent', 'MOE', 'Number']

# fit the scale object to the columns & set values
analysis[columns] = scale.fit_transform(analysis[columns])

# look at the scaled dataframe
#print('\nScaled:\n', analysis.head())

#### C) Look at the Distributions after scaling ###############
#Distribution of year After Scaling
#plot.figure('scaled')
#plot.hist(analysis['Year'])
#plot.show()

analysis[columns] = scale.inverse_transform(analysis[columns])

print('\nReturned to Original:\n', analysis.head())

'''You can get k-1 dummy levels by specifying drop_first = True, but we want to include all of them so we can pivot back 
correctly using idxmax. This means we need to remove the last dummy when we generate our x inputs (see below) '''

# create dummies on state column and don't add any prefix or sep (to get good state names when we pivot back)
analysis = pd.get_dummies(analysis, columns = ['Location', 'Family Income', 'Age','Coverage Type'], prefix = '', prefix_sep = '')

# look at the resulting dataframe
print('\nDummy Variable Encoding:\n', analysis.head())


###################################################################################################################
###### Evaluate the Accuracy of Linear Regression #################################################################
###################################################################################################################

''' Sklearn uses the numpy seed so we need to set the seed using numpy (np)'''
np.random.seed(100)

# only keep values where variables are not null (otherwise will mess up regression)
no_null = analysis.dropna(how='any')

# y = dependent = divorce rate
y1 = no_null['Number']
y2 = no_null['Percent']
y3 = no_null['MOE']

# x = all other variables except for last dummy variable (so we have k-1): could specify by location as well
# note: if using multiple dummy cols you will need to drop one value for each
x = no_null.drop(['United States','Uninsured','$75,000 or more','19-25'], axis=1)


########################################################################
#### A) Hold out Method for evaluating accuracy metrics ################
########################################################################
''' You want to split the data into a training and testing set. You may also see "validation set', but
you do not need to worry about evaluation for this assignment. 
Training: the data used to build the model
Validation: the data used to tune parameters
Testing: the data used to get accuracy/performance measures for data outside of our initial sample'''

### 1) split the data into 80% train and 20% test ################
x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2)
# x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.2)
# x_train, x_test, y_train, y_test = train_test_split(x, y3, test_size=0.2)




lm_test = LinearRegression().fit(x_train, y_train)

''' R2 = percent of variance of y explained by model '''
r2 = lm_test.score(x_test, y_test)
print('\nHold Out R2:', round(r2, 4))

pred = lm_test.predict(x_test)

# Get RMSE (average difference of predicted vs. actual in Y units) from predictions & actual values (y)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))  # RMSE is the square root of MSE
print('Hold Out RMSE:', round(rmse, 4))

########################################################################
#### B) K-Fold Cross Validation ########################################
########################################################################

### 1) create an empty linear regression object #######################################################
lm_k = LinearRegression()

scores_r2 = list(np.round(cross_val_score(lm_k, x, y1, cv=10), 4))

# set scoring parameter to get neg mse (use 10 folds for everything to keep consistency)
scores_mse = cross_val_score(lm_k, x, y1, cv=10, scoring="neg_mean_squared_error")

scores_rmse = list(np.round(np.sqrt(np.abs(scores_mse)), 4))

print('\nCross-validated R2 By Fold:\n', scores_r2, "\nCross-Validated MSE By Fold:\n", scores_rmse)

# A) generate a prediction list
predictions = cross_val_predict(lm_k, x, y1, cv=10)

# get the r2
r2 = metrics.r2_score(y1, predictions)

rmse = np.sqrt(metrics.mean_squared_error(y1, predictions))

print('Cross-Validated R2 For All Folds:', round(r2, 4), '\nCross-Validation RMSE For All Folds:', round(rmse,4) )


####################################################################################################################
###### Build Final Model  ##########################################################################################
####################################################################################################################
lm_final = LinearRegression().fit(x, y1)


####################################################################################################################
###### Prep Data for Output ########################################################################################
####################################################################################################################

######################################################################################
#### 1) Predict With Model to Fill Null Values #######################################
######################################################################################

# get the row number (index #) of null values in divorce rate
div_null_list = analysis.index[analysis['Divorce_Rate'].isnull()].tolist()

# get the row number (index #) of null values in marriage rate
mar_null_list = analysis.index[analysis['Marriage_Rate'].isnull()].tolist()

null_list = np.setdiff1d(div_null_list,mar_null_list)

pred_vals = analysis.drop(['Divorce_Rate', 'Wyoming'], axis=1)

for x in null_list:                 # x is the position of the null value row
    analysis.ix[x, 'Divorce_Rate'] = lm_final.predict([pred_vals.iloc[x].tolist()])

#####################################################################################
#### 2) Reshape to Remove Dummies ###################################################
#####################################################################################
state = analysis.drop(analysis.columns[:3], axis=1)

# remove all dummy columns from analysis dataframe
analysis.drop(analysis.columns[3:], axis=1, inplace=True)

analysis['State'] = state.idxmax(axis=1)   # axis = 1 tells it to get the column name
print('\nFinal Dataset:\n',analysis.head())



####################################################################################################################
###### Export to Excel #############################################################################################
####################################################################################################################
# index = false because we don't need the row numbering provided by pandas
analysis.to_excel('data/rates_imputation.xls', index=False, na_rep='null')