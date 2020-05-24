#First of all, we need to import/install all the packages needed for this session

import pandas as pd
import matplotlib.pyplot as plt




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

#Then, we start our explorative analysis on the main variables of the dataset

#1. BadBuy
bb = train['IsBadBuy'].value_counts(normalize=True).plot(kind='bar', title='Bad Buy percentage', color=tableau20[4],
                                                         alpha=0.5)
plt.xlabel('Is Bad Buy')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

#We notice that the dataset is really biased towards Good buys.

#2. Auction
count_auction = train["Auction"].value_counts().plot(kind="bar", title="Auctioner", color=tableau20[7],
                                                         alpha=0.5)
plt.xlabel('Auctioner')
plt.ylabel('N. of cars')
plt.xticks(rotation=0)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

#We notice the most important auction house is Manheim

#3. VehYear
vehyear = train['VehYear'].hist(color=tableau20[10], alpha=0.5)
plt.title('Vehicle count by year of production')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.grid()
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

#We notice the majority of vehicles is from 2006

#4. VehAge
vehage = train['VehicleAge'].hist(color=tableau20[10], alpha=0.5)
plt.title('Vehicle count by age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

#We notice the majority of vehicles is 4 years old

#5. VehOdo
train['VehOdo'].plot(kind="kde", color=tableau20[2], alpha=0.5)
plt.title('Odometer count by miles')
plt.xlabel('Age')
plt.ylabel('Frequency')
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

#We notice the majority of vehicles has made 90000 miles


