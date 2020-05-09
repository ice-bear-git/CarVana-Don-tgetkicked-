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
time.sleep(7)
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
model_submodel_nan_transmission=train[train["Transmission"].isnull()][['Model','SubModel']]
dataframe_for_tranmission_nan=train.loc[train['Model'].isin(model_submodel_nan_transmission['Model'])]
dataframe_for_tranmission_nan['Transmission'].groupby([dataframe_for_tranmission_nan['Model']]).unique()
crosstab_for_tranmission_nan=pd.crosstab(dataframe_for_tranmission_nan.Model, dataframe_for_tranmission_nan.Transmission)
maxValueIndex_for_transmission = crosstab_for_tranmission_nan.idxmax(axis=1)
transmission_nan_elimination = maxValueIndex_Model.to_frame(name='Transmission')
train.loc[train['Transmission'].isnull(),'Transmission'] = train['Model'].map(transmission_nan_elimination.Transmission)

#by submodel
model_submodel_nan_transmission=train[train["Transmission"].isnull()][['Model','SubModel']]
dataframe_for_tranmission_nan_SubModel=train.loc[train['SubModel'].isin(model_submodel_nan_transmission['SubModel'])]
dataframe_for_tranmission_nan_SubModel['Transmission'].groupby([dataframe_for_tranmission_nan_SubModel['SubModel']]).unique()
crosstab_for_tranmission_nan_submodel=pd.crosstab(dataframe_for_tranmission_nan_SubModel.Model, dataframe_for_tranmission_nan_SubModel.Transmission)
maxValueIndex_for_transmission_submodel = crosstab_for_tranmission_nan_submodel.idxmax(axis=1)
transmission_nan_elimination_submodel = maxValueIndex_for_transmission_submodel.to_frame(name='Transmission')
train.loc[train['Transmission'].isnull(),'Transmission'] = train['SubModel'].map(transmission_nan_elimination_submodel.Transmission)
print('clean!')

#3.5 We use the same grouping by approach for WheelType's NaN values
print('cleaning nan variables for WheelType variable...')

#We fix the first group of NaN values groupbing by Model
model_submodel_make_trim_nan_wheeltype=train[train["WheelType"].isnull()][['Model','SubModel','Make','Trim']]
dataframe_for_wheeltype_nan=train.loc[train['Model'].isin(model_submodel_make_trim_nan_wheeltype['Model'])]
dataframe_for_wheeltype_nan['WheelType'].groupby([dataframe_for_wheeltype_nan['Model']]).unique()
crosstab_for_wheeltype_nan=pd.crosstab(dataframe_for_wheeltype_nan.Model, dataframe_for_wheeltype_nan.WheelType)
maxValueIndex_for_wheeltype= crosstab_for_wheeltype_nan.idxmax(axis=1)
wheeltype_nan_elimination = maxValueIndex_for_wheeltype.to_frame(name='WheelType')
train.loc[train['WheelType'].isnull(),'WheelType'] = train['Model'].map(wheeltype_nan_elimination.WheelType)

#We fix the second group of NaN values groupbing by SubModel
dataframe_for_groupby_submodel_wheeltype=train.loc[train['SubModel'].isin(model_submodel_make_trim_nan_wheeltype['SubModel'])]
dataframe_for_groupby_submodel_wheeltype['WheelType'].groupby([dataframe_for_groupby_submodel_wheeltype['SubModel']]).unique()
crosstab_for_wheeltype_nan_submodel=pd.crosstab(dataframe_for_groupby_submodel_wheeltype.SubModel, dataframe_for_groupby_submodel_wheeltype.WheelType)
maxValueIndex_for_wheeltype_submodel = crosstab_for_wheeltype_nan_submodel.idxmax(axis=1)
wheeltype_nan_elimination_submodel = maxValueIndex_for_wheeltype_submodel.to_frame(name='WheelType')
train.loc[train['WheelType'].isnull(),'WheelType'] = train['SubModel'].map(wheeltype_nan_elimination_submodel.WheelType)

#We fix the third group of NaN values groupbing by Make
dataframe_for_groupby_make_wheeltype=train.loc[train['Make'].isin(model_submodel_make_trim_nan_wheeltype['Make'])]
dataframe_for_groupby_make_wheeltype['WheelType'].groupby([dataframe_for_groupby_make_wheeltype['Make']]).unique()
crosstab_for_wheeltype_nan_make=pd.crosstab(dataframe_for_groupby_make_wheeltype.Make, dataframe_for_groupby_make_wheeltype.WheelType)
maxValueIndex_for_wheeltype_make = crosstab_for_wheeltype_nan_make.idxmax(axis=1)
wheeltype_nan_elimination_make = maxValueIndex_for_wheeltype_make.to_frame(name='WheelType')
train.loc[train['WheelType'].isnull(),'WheelType'] = train['Make'].map(wheeltype_nan_elimination_make.WheelType)
print('clean!')



#3.6 We use the same grouping by approach for Size's NaN values

print('cleaning nan variables for Size variable...')

#First, we use Model to group by
model_submodel_make_nan_size=train[train["Size"].isnull()][['Model','SubModel','Make']]
dataframe_for_size_nan=train.loc[train['Model'].isin(model_submodel_make_nan_size['Model'])]
dataframe_for_size_nan['Size'].groupby(
    [dataframe_for_size_nan['Model']]).unique()
crosstab_for_size_nan_model=pd.crosstab(dataframe_for_size_nan.Model, dataframe_for_size_nan.Size)
maxValueIndex_for_size = crosstab_for_size_nan_model.idxmax(axis=1)
size_nan_elimination_make = maxValueIndex_for_size.to_frame(name='Size')
train.loc[train['Size'].isnull(),'Size'] = train['Model'].map(size_nan_elimination_make.Size)

#Secondly, we use Trim
model_submodel_make_trim_nan_size=train[train["Size"].isnull()][['Model','SubModel','Make', 'Trim']]
dataframe_groupby_trim_size=train.loc[train['Trim'].isin(model_submodel_make_trim_nan_size['Trim'])]
dataframe_groupby_trim_size['Size'].groupby(
    [dataframe_groupby_trim_size['Trim']]).unique()
crosstab_trim_size=pd.crosstab(dataframe_groupby_trim_size.Trim, dataframe_groupby_trim_size.Size)
maxValueIndex_for_size_trim = crosstab_trim_size.idxmax(axis=1)
size_nan_elimination_trim = maxValueIndex_for_size_trim.to_frame(name='Size')
train.loc[train['Size'].isnull(),'Size'] = train['Trim'].map(size_nan_elimination_trim.Size)

print('clean!')

#3.7 We use the same grouping by approach for TopThreeAmericanName's NaN values

print('cleaning nan variables for TopThreeAmericanName variable...')
make_nan_topthree=train[train["TopThreeAmericanName"].isnull()][['Make']]
dataframe_groupby_topthree=train.loc[train['Make'].isin(make_nan_topthree['Make'])]
dataframe_groupby_topthree['TopThreeAmericanName'].groupby(
    [dataframe_groupby_topthree['Make']]).unique()
crosstab_topthree_make=pd.crosstab(dataframe_groupby_topthree.Make, dataframe_groupby_topthree.TopThreeAmericanName)
maxValueIndex_for_topthree = crosstab_topthree_make.idxmax(axis=1)
topthree_nan_elimination_make = maxValueIndex_for_topthree.to_frame(name='TopThreeAmericanName')
train.loc[train['TopThreeAmericanName'].isnull(),'TopThreeAmericanName'] = train['Make'].map(topthree_nan_elimination_make.TopThreeAmericanName)
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
print('creating the cleaned document...')