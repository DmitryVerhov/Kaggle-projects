# Car price prediction

## Goal

The goal was to predict prices of cars wich were given in the csv file. The complexity of this task is that there was no data to train models. So, I had to get them from other sourses.

## 1. Getting data.

I wrote two functions to parse the site auto.ru and received a dataframe with 115477 entries.Also I've found a dataset on kaggle of almost the same period with 332676 entires, but usefull were only 125848 entires, because of brands, needed to predict. After merging the datasets and dropping duplicates, I've got a dataset with 198403 entires.  


## 2. Data preparing.

It was a hard goal to make all datasets simillar, because of different formats in each dataset. Finally I've created a single dataframe to work with. Then I splitted features in 3 categories: numeric, bins and categorical.

I've cleaned the data, unifyed feature values, and created new features, one of which, age, were highly significant.

## 3. Exploratory Data Analysis.

During the EDA I've cleared out the significance of features and checked their correlation and distribution. Also I've found the dependence between target variable and features.

## 4. Data preprocessing.

In this block I encoded bin and cat features. Also num features and target variable were normalized. The result on leaderboard increased on about 30% after normalization.  

## 5. Building models.
I built the Naive model to compare with it at first. Then I've tried to train Bagging on DecisionTrees, Randomforest, GradientBoost, Bagging on GradientBoost, XgBoost, Catboost and ensemble of last three. 

For RandomForest I've tried to find the best parameters with gridsearch. Thats took a lot of time, but didn't gave a result. The best result showed Catboost.   

After all, I was to input the correction coefficient, because of the difference in time between test and train data.

The result, that I have reached on leaderboard is mean absolute percentage error 16.37674.
Not so good, but it's better than baseline in about two times. 

## 6. Conclusions.

The best result showed Catboost, maybe becase I do not have enough experience to tune up other models.

But the main goal was to improve my skills in DataScience and I undoubtedly reached it.

I've upgraded my skills in python, clearing data, investigation, analysys and machine learning. The parsing skill will be very usefull in future.
