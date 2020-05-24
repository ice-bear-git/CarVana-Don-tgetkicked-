#First of all, we need to import/install all the packages needed for this session

import numpy as np
import pandas as pd
import time

#This task was far from easy: we had a lot of categorical values in the dataset

#We initialized the training set -> you can get it here: https://www.kaggle.com/c/DontGetKicked/data

train = pd.read_csv('training.csv')

#We start to clean the dataset

#1. We eliminate some columns we found not useful to our task
train.drop(['RefId', 'PurchDate', 'VehYear', 'Color', 'WheelTypeID', 'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice', 'MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO'], axis=1, inplace=True)

#2. We search for NaN values
print('These are our NaN values at the start...')
train_for_nan = train.isnull().sum()
sum_null = train.isnull().sum()
sum_null.index
for name, num_na in zip(sum_null.index, train_for_nan):
    if num_na > 0:
        print("{}: {}".format(name, num_na))
time.sleep(2)
print('starting the cleaning process in 3...2...1...')

#3. We search for patterns to eliminate all the NaN values

#3.1 For Nationality

print('cleaning nan variables for Nationality variable...')
make_nationality_null=train[train["Nationality"].isnull()][['Make']]
pre_cross_df=train.loc[train['Make'].isin(make_nationality_null['Make'])]
crosstab_american_asian=pd.crosstab(pre_cross_df.Make, pre_cross_df.Nationality)
crosstab_american_asian['Nationality'] = np.where(crosstab_american_asian['AMERICAN']>crosstab_american_asian['OTHER ASIAN'], 'AMERICAN', 'OTHER ASIAN')
train.loc[train['Nationality'].isnull(),'Nationality'] = train['Make'].map(crosstab_american_asian.Nationality)
print('clean!')

#We use the informations given by Make to find the NaN values of Nationality

#3.2 For Trim and other variables we need to change approach

train['Trim'].groupby(
    [train['Model'],train['SubModel'], train['Trim']]).count()

#we will use variables as this in the whole cleaning process
#To be understood, this variables need to read as this example:
#a dataframe with model, submodel, make where trim is nan
model_submodel_make_nan_trim=train[train["Trim"].isnull()][['Model','SubModel','Make']]

#We fix the first group of NaN values grouping by Model
dataframe_for_groupby_model=train.loc[train['SubModel'].isin(model_submodel_make_nan_trim['SubModel'])]
dataframe_for_groupby_model['Trim'].groupby([dataframe_for_groupby_model['SubModel']]).unique()
print('cleaning NaN values for trim... phase 1')
crosstab_model_trim=pd.crosstab(dataframe_for_groupby_model.Model, dataframe_for_groupby_model.Trim)
maxValueIndex_Model = crosstab_model_trim.idxmax(axis=1)
Model_lookup = maxValueIndex_Model.to_frame(name='Trim')
train.loc[train['Trim'].isnull(),'Trim'] = train['Model'].map(Model_lookup.Trim)
print('clean!')


#We fix the second group of NaN values grouping by SubModel
dataframe_for_groupby_submodel=train.loc[train['SubModel'].isin(model_submodel_make_nan_trim['SubModel'])]
dataframe_for_groupby_submodel['Trim'].groupby(
    [dataframe_for_groupby_submodel['SubModel']]).unique()
print('cleaning NaN values for trim... phase 2')
crosstab_submodel_trim=pd.crosstab(dataframe_for_groupby_submodel.SubModel, dataframe_for_groupby_submodel.Trim)
maxValueIndex_SubModel = crosstab_submodel_trim.idxmax(axis=1)
SubModel_lookup = maxValueIndex_SubModel.to_frame(name='Trim')
train.loc[train['Trim'].isnull(),'Trim'] = train['SubModel'].map(SubModel_lookup.Trim)
print('clean!')

#We fix the third group of NaN values grouping by Make

dataframe_for_groupby_make=train.loc[train['Make'].isin(model_submodel_make_nan_trim['Make'])]
dataframe_for_groupby_make['Trim'].groupby([dataframe_for_groupby_make['Make']]).unique()
print('cleaning NaN values for trim... phase 3')
crosstab_make_trim=pd.crosstab(dataframe_for_groupby_make.Make, dataframe_for_groupby_make.Trim)
maxValueIndex_Make = crosstab_make_trim.idxmax(axis=1)
make_trim_nan_elimination = maxValueIndex_Make.to_frame(name='Trim')
train.loc[train['Trim'].isnull(),'Trim'] = train['Make'].map(make_trim_nan_elimination.Trim)
print('clean!')


#3.3 We use the same grouping by approach for SubModel's NaN values
print('cleaning nan variables for SubModel variable...')

model_make_nan_submodel=train[train["SubModel"].isnull()][['Model','Make']]
dataframe_for_submodel_nan=train.loc[train['Model'].isin(model_make_nan_submodel['Model'])]
dataframe_for_submodel_nan['SubModel'].groupby(
    [dataframe_for_submodel_nan['Model']]).unique()
crosstab_for_submodel_nan=pd.crosstab(dataframe_for_submodel_nan.Model, dataframe_for_submodel_nan.SubModel)
maxValueIndex_for_submodel = crosstab_for_submodel_nan.idxmax(axis=1)
submodel_nan_elimination = maxValueIndex_for_submodel.to_frame(name='SubModel')
train.loc[train['SubModel'].isnull(),'SubModel'] = train['Model'].map(submodel_nan_elimination.SubModel)
print('clean!')

#3.4 We use the same grouping by approach for Transimission's NaN values
print('cleaning nan variables for Transmission variable...')

#first: we clean a dirt-written value (manual->Manual)
train.loc[train['Transmission']=='Manual','Transmission'] = train.loc[train['Transmission']=='Manual','Transmission'].str.replace('Manual', 'MANUAL')

#then, we do the same we did for other variables by model
nan_df=train[train["Transmission"].isnull()][['Model','SubModel']]
temp_df=train.loc[train['SubModel'].isin(nan_df['SubModel'])]
temp_df['Transmission'].groupby([temp_df['Model']]).unique()
c=pd.crosstab(temp_df.Model, temp_df.Transmission)
maxValueIndex = c.idxmax(axis=1)
lookup = maxValueIndex.to_frame(name='Transmission')
train.loc[train['Transmission'].isnull(),'Transmission'] = train['Model'].map(lookup.Transmission)


#3.5 We use the same grouping by approach for WheelType's NaN values
print('cleaning nan variables for WheelType variable...')

#We fix the first group of NaN values groupbing by Model
nan_df=train[train["WheelType"].isnull()][['Model','SubModel','Make','Trim']]
temp_df=train.loc[train['Model'].isin(nan_df['Model'])]
temp_df['WheelType'].groupby([temp_df['Model']]).unique()
c=pd.crosstab(temp_df.Model, temp_df.WheelType)
maxValueIndex= c.idxmax(axis=1)
lookup = maxValueIndex.to_frame(name='WheelType')
train.loc[train['WheelType'].isnull(),'WheelType'] = train['Model'].map(lookup.WheelType)

#We fix the second group of NaN values groupbing by SubModel
temp_df=train.loc[train['SubModel'].isin(nan_df['SubModel'])]
temp_df['WheelType'].groupby([temp_df['SubModel']]).unique()
c=pd.crosstab(temp_df.SubModel, temp_df.WheelType)
maxValueIndex = c.idxmax(axis=1)
lookup= maxValueIndex.to_frame(name='WheelType')
train.loc[train['WheelType'].isnull(),'WheelType'] = train['SubModel'].map(lookup.WheelType)

#We fix the third group of NaN values groupbing by Make
temp_df=train.loc[train['Make'].isin(nan_df['Make'])]
temp_df['WheelType'].groupby([temp_df['Make']]).unique()
c=pd.crosstab(temp_df.Make, temp_df.WheelType)
maxValueIndex = c.idxmax(axis=1)
lookup = maxValueIndex.to_frame(name='WheelType')
train.loc[train['WheelType'].isnull(),'WheelType'] = train['Make'].map(lookup.WheelType)
print('clean!')



#3.6 We use the same grouping by approach for Size's NaN values

print('cleaning nan variables for Size variable...')

#First, we use Model to group by
nan_df=train[train["Size"].isnull()][['Model','SubModel','Make']]
temp_df=train.loc[train['Model'].isin(nan_df['Model'])]
temp_df['Size'].groupby(
    [temp_df['Model']]).unique()
c=pd.crosstab(temp_df.Model, temp_df.Size)
maxValueIndex = c.idxmax(axis=1)
lookup = maxValueIndex.to_frame(name='Size')
train.loc[train['Size'].isnull(),'Size'] = train['Model'].map(lookup.Size)

#Secondly, we use Trim
nan_df=train[train["Size"].isnull()][['Model','SubModel','Make', 'Trim']]
temp_df=train.loc[train['Trim'].isin(nan_df['Trim'])]
temp_df['Size'].groupby(
    [temp_df['Trim']]).unique()
c=pd.crosstab(temp_df.Trim, temp_df.Size)
maxValueIndex = c.idxmax(axis=1)
lookup = maxValueIndex.to_frame(name='Size')
train.loc[train['Size'].isnull(),'Size'] = train['Trim'].map(lookup.Size)

print('clean!')

#3.7 We use the same grouping by approach for TopThreeAmericanName's NaN values

print('cleaning nan variables for TopThreeAmericanName variable...')
nan_df=train[train["TopThreeAmericanName"].isnull()][['Make']]
temp_df=train.loc[train['Make'].isin(nan_df['Make'])]
temp_df['TopThreeAmericanName'].groupby(
    [temp_df['Make']]).unique()
c=pd.crosstab(temp_df.Make, temp_df.TopThreeAmericanName)
maxValueIndex = c.idxmax(axis=1)
lookup = maxValueIndex.to_frame(name='TopThreeAmericanName')
train.loc[train['TopThreeAmericanName'].isnull(),'TopThreeAmericanName'] = train['Make'].map(lookup.TopThreeAmericanName)
print('clean!')


#3.8 For the MMRA variables, we can use a different method: we cover the NaN values with the mean of each stratification
print('cleaning nan variables for MMRA variables...')

#We define a function that cover up the NaN valeus with the mean of all the other values
def mean_nan_filling(x):
    return x.fillna(x.mean())

#first, we use the most granular level of grouping that we have
train['MMRAcquisitionAuctionAveragePrice']=train['MMRAcquisitionAuctionAveragePrice'].groupby(
    [train['Make'], train['SubModel'], train['Model']]).apply(mean_nan_filling)
train['MMRAcquisitionAuctionCleanPrice']=train['MMRAcquisitionAuctionCleanPrice'].groupby(
    [train['Make'], train['SubModel'], train['Model']]).apply(mean_nan_filling)
train['MMRAcquisitionRetailAveragePrice']=train['MMRAcquisitionRetailAveragePrice'].groupby(
    [train['Make'], train['SubModel'], train['Model']]).apply(mean_nan_filling)
train['MMRAcquisitonRetailCleanPrice']=train['MMRAcquisitonRetailCleanPrice'].groupby(
    [train['Make'], train['SubModel'], train['Model']]).apply(mean_nan_filling)

#we step back a level of granularity and we group by for make and submodel
train['MMRAcquisitionAuctionAveragePrice']=train['MMRAcquisitionAuctionAveragePrice'].groupby(
    [train['Make'], train['SubModel']]).apply(mean_nan_filling)
train['MMRAcquisitionAuctionCleanPrice']=train['MMRAcquisitionAuctionCleanPrice'].groupby(
    [train['Make'], train['SubModel']]).apply(mean_nan_filling)
train['MMRAcquisitionRetailAveragePrice']=train['MMRAcquisitionRetailAveragePrice'].groupby(
    [train['Make'], train['SubModel']]).apply(mean_nan_filling)
train['MMRAcquisitonRetailCleanPrice']=train['MMRAcquisitonRetailCleanPrice'].groupby(
    [train['Make'], train['SubModel']]).apply(mean_nan_filling)

#we end covering up NaN values grouping only by make
train['MMRAcquisitionAuctionAveragePrice']=train['MMRAcquisitionAuctionAveragePrice'].groupby(
    [train['Make']]).apply(mean_nan_filling)
train['MMRAcquisitionAuctionCleanPrice']=train['MMRAcquisitionAuctionCleanPrice'].groupby(
    [train['Make']]).apply(mean_nan_filling)
train['MMRAcquisitionRetailAveragePrice']=train['MMRAcquisitionRetailAveragePrice'].groupby(
    [train['Make']]).apply(mean_nan_filling)
train['MMRAcquisitonRetailCleanPrice']=train['MMRAcquisitonRetailCleanPrice'].groupby(
    [train['Make']]).apply(mean_nan_filling)

print('clean!')

print('These are our NaN values at the end...')
train_for_nan = train.isnull().sum()
sum_null = train.isnull().sum()
sum_null.index
for name, num_na in zip(sum_null.index, train_for_nan):
    if num_na > 0:
        print("{}: {}".format(name, num_na))
time.sleep(7)
print('creating the clean document...')