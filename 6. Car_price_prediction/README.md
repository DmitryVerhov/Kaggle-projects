# Car price prediction

## Goal

Our goal was to predict prices of cars wich were given us in the test.csv file. The complexity of this task is that we don't have a data to train our models. So, we've needed to get them from other sourses.

## 1. Getting data.

We have written two functions to parse site auto.ru and got a dataframe with 115477 entires. Also we've found a dataset on kaggle of almost the same period with 332676 entires, but usefull were only 125848 entires, because of brands, presentsd in test. After we've joined this datadets and dropped duplicates, we've got a train dataset with 198403 entires.  


## 2. Data preparing.

It was a hard goal to make all datasets simillar, because of different formats in each dataset. Finally we've created a single dataframe to work with. Then we splitted features in 3 lists: numeric, bins and categorical.

We've cleaned the data, unifyed feature values, and created new features, one of which, age, were highly significant.

## 3. Exploratory Data Analysis.

During the EDA we've cleared out the significance of features and checked their correlation and distribution. Also we've found the dependence between target variable and features.

## 4. Data preprocessing.

In this block we've encoded bin and cat features. Also  num features and target variable were normalized. After we've normalized the target variable the result on leaderboard increased on about 30%. 

## 5. Building models.

For a start point the Naive model were builded to compare with. Then we've tried to train Bagging on DecisionTrees, Randomforest, GradientBoost, Bagging on GradientBoost, XgBoost, Catboost and ensemble of last three. 

For RandomForest we've tried to find the best parameters with gridsearch. Thats took a lot of time, but didn't gave a result.   

The best result showed Catboost, then we've tuned it and got the submission. 

After all, we was to input the correction coefficient, because of the difference in time between test and train data.

The result, that we have reached on leaderboard is 16.37674.
Not so good, but it better than baseline in about two times. 

## 6. Conclusions.

The best result showed us Catboost, maybe becase we do not have enough experience to tune up other models.

But the main goal was to improve our skills in DataScience and we undoubtedly reached it.

We upgraded our skills in python, clearing data, investigation, analysys and machine learning. The parsing skill will be very usefull in future. This is a real art to tune models, and it the first step in a such a fascinating activity.   
