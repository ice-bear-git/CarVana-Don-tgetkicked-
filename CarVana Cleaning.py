#First of all, we need to import/install all the packages needed for this session

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#We initialized some RGB combinations for good visualization -> credit: http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


#We initialized the training set -> you can get it here: https://www.kaggle.com/c/DontGetKicked/data

train = pd.read_csv('training.csv')

#We start to clean the dataset

#1. We eliminate some columns we found not useful to our task
train.drop(['RefId', 'PurchDate', 'VehYear', 'Color', 'WheelTypeID', 'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice', 'MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO'], axis=1, inplace=True)

#2. We search for NaN values
train_for_nan = train.isnull().sum()
sum_null = train.isnull().sum()
sum_null.index
for name, num_na in zip(sum_null.index, train_for_nan):
    if num_na > 0:
        print("{}: {}".format(name, num_na))

#3. We search for patterns to eliminate all the NaN values

#3.1 For Nationality

make_nationality_null=train[train["Nationality"].isnull()][['Make']]
pre_cross_df=train.loc[train['Make'].isin(make_nationality_null['Make'])]
crosstab_american_asian=pd.crosstab(pre_cross_df.Make, pre_cross_df.Nationality)
crosstab_american_asian['Nationality'] = np.where(crosstab_american_asian['AMERICAN']>crosstab_american_asian['OTHER ASIAN'], 'AMERICAN', 'OTHER ASIAN')
train.loc[train['Nationality'].isnull(),'Nationality'] = train['Make'].map(crosstab_american_asian.Nationality)

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


#We fix the second group of NaN values grouping by SubModel
dataframe_for_groupby_submodel=train.loc[train['SubModel'].isin(model_submodel_make_nan_trim['SubModel'])]
dataframe_for_groupby_submodel['Trim'].groupby(
    [dataframe_for_groupby_submodel['SubModel']]).unique()
print('cleaning NaN values for trim... phase 2')
crosstab_submodel_trim=pd.crosstab(dataframe_for_groupby_submodel.SubModel, dataframe_for_groupby_submodel.Trim)
maxValueIndex_SubModel = crosstab_submodel_trim.idxmax(axis=1)
SubModel_lookup = maxValueIndex_SubModel.to_frame(name='Trim')
train.loc[train['Trim'].isnull(),'Trim'] = train['SubModel'].map(SubModel_lookup.Trim)

#We fix the third group of NaN values grouping by Make
dataframe_for_groupby_make=train.loc[train['Make'].isin(model_submodel_make_nan_trim['Make'])]
dataframe_for_groupby_make['Trim'].groupby([dataframe_for_groupby_make['Make']]).unique()
print('cleaning NaN values for trim... phase 3')
crosstab_make_trim=pd.crosstab(dataframe_for_groupby_make.Make, dataframe_for_groupby_make.Trim)
maxValueIndex_Make = crosstab_make_trim.idxmax(axis=1)
make_trim_nan_elimination = maxValueIndex_Make.to_frame(name='Trim')
train.loc[train['Trim'].isnull(),'Trim'] = train['Make'].map(make_trim_nan_elimination.Trim)

#3.3 We use the same grouping by approach for SubModel's NaN values

model_make_nan_submodel=train[train["SubModel"].isnull()][['Model','Make']]
dataframe_for_submodel_nan=train.loc[train['Model'].isin(model_make_nan_submodel['Model'])]
dataframe_for_submodel_nan['SubModel'].groupby(
    [dataframe_for_submodel_nan['Model']]).unique()
crosstab_for_submodel_nan=pd.crosstab(dataframe_for_submodel_nan.Model, dataframe_for_submodel_nan.SubModel)
maxValueIndex_for_submodel = crosstab_for_submodel_nan.idxmax(axis=1)
submodel_nan_elimination = maxValueIndex_for_submodel.to_frame(name='SubModel')
train.loc[train['SubModel'].isnull(),'SubModel'] = train['Model'].map(submodel_nan_elimination.SubModel)

#3.4 We use the same grouping by approach for Transimission's NaN values

#first: we clean a dirt-written value (manual->Manual)
train.loc[train['Transmission']=='Manual','Transmission'] = train.loc[train['Transmission']=='Manual','Transmission'].str.replace('Manual', 'MANUAL')

#then, we do the same we did for other variables
model_submodel_nan_transmission=train[train["Transmission"].isnull()][['Model','SubModel']]
dataframe_for_tranmission_nan=train.loc[train['Model'].isin(model_submodel_nan_transmission['Model'])]
dataframe_for_tranmission_nan['Transmission'].groupby([dataframe_for_tranmission_nan['Model']]).unique()
crosstab_for_tranmission_nan=pd.crosstab(dataframe_for_tranmission_nan.Model, dataframe_for_tranmission_nan.Transmission)
maxValueIndex_for_transmission = crosstab_for_tranmission_nan.idxmax(axis=1)
transmission_nan_elimination = maxValueIndex_Model.to_frame(name='Transmission')
train.loc[train['Transmission'].isnull(),'Transmission'] = train['Model'].map(transmission_nan_elimination.Transmission)

#3.5 We use the same grouping by approach for WheelType's NaN values

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

#We fix the third group of NaN values groupbing by SubModel
dataframe_for_groupby_make_wheeltype=train.loc[train['Make'].isin(model_submodel_make_trim_nan_wheeltype['Make'])]
dataframe_for_groupby_make_wheeltype['WheelType'].groupby([dataframe_for_groupby_make_wheeltype['SubModel']]).unique()
crosstab_for_wheeltype_nan_make=pd.crosstab(dataframe_for_groupby_make_wheeltype.Make, dataframe_for_groupby_make_wheeltype.WheelType)
maxValueIndex_for_wheeltype_make = crosstab_for_wheeltype_nan_make.idxmax(axis=1)
wheeltype_nan_elimination_make = maxValueIndex_for_wheeltype_make.to_frame(name='WheelType')
train.loc[train['WheelType'].isnull(),'WheelType'] = train['SubModel'].map(wheeltype_nan_elimination_make.WheelType)

#3.6 We use the same grouping by approach for Size's NaN values
model_submodel_make_nan_size=train[train["Size"].isnull()][['Model','SubModel','Make']]
dataframe_for_size_nan=train.loc[train['SubModel'].isin(model_submodel_make_nan_size['SubModel'])]
dataframe_for_size_nan['Size'].groupby([dataframe_for_size_nan['SubModel']]).unique()

