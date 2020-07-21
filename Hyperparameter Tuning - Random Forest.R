#Purpose: Exploring hyperparameter tuning (grid search and random search) with a Random Forest model

#Cleansing/Manipulation Packages:
pacman::p_load("dplyr","tidyr","stringr","lubridate","janitor")
#Graphing Packages:
pacman::p_load("ggplot2","esquisse","gridExtra")
#ML Packages:
pacman::p_load("car","MLmetrics","estimatr","fpp","forecast","caret","ROCR","lift","randomForest","glmnet","MASS",
               "e1071","partykit","rpart","ROCR","lift","randomForest","xgboost","tensorflow", "keras")
#Other Packages:
pacman::p_load("sqldf","readxl","geosphere")

#import datasets
library(mlbench)
data(Sonar) #imports dataset from the library
dataset <- Sonar
x <- dataset[,1:60]
y <- dataset[,61] #note we have two levels - M and R

##creating the model - first with default hyperparameter settings
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 1024
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x)) #it is pretty standard to initially set mtry to the square root of #features
tunegrid <- expand.grid(.mtry=mtry)

#now the random forest model
rf_default <- train(Class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)

















