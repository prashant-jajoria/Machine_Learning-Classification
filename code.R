## FIT5149 A2 S1 2021 - Final claffifier
## Group 2
## date: 30/05/2021
## Authors:   Faezeh Fazel 
##            Ricardo Arias 
##            Prashant Jajoria 


library(tidyverse)
library(tsfeatures)
library(dplyr)
library(zoo)
library(class)
library(caret)
#install.packages('runner') # Install the first time you run the Notebook
library(runner)
#install.packages('gtools') # Install the first time you run the Notebook
library(gtools)
library(glmnet)
#install.packages('MLmetrics') # Install the first time you run the Notebook
library(MLmetrics)
library(reshape2)
library(splitstackshape)
library(splitTools)
library(ranger)
library(xgboost)
library(drat)
library(Matrix)


# -------- read the csv----------------
train.dataset <- read.csv("./train_data_withlabels.csv")
test.dataset <- read.csv("./test_data_nolabels.csv")

# --- conver tto factors---------------
train.dataset$hourofday = as.factor(train.dataset$hourofday)
train.dataset$dayofweek = as.factor(train.dataset$dayofweek)
train.dataset$ac = as.factor(train.dataset$ac)
train.dataset$ev = as.factor(train.dataset$ev)
train.dataset$oven = as.factor(train.dataset$oven)
train.dataset$wash = as.factor(train.dataset$wash)
train.dataset$dryer = as.factor(train.dataset$dryer)

# make 5 different datasets for each of the appliance
train.dataset.ac = train.dataset[c(8,9,2,10,11,12,13,14,15,16,3)]
train.dataset.ev = train.dataset[c(8,9,2,10,11,12,13,14,15,16,4)]
train.dataset.oven = train.dataset[c(8,9,2,10,11,12,13,14,15,16,5)]
train.dataset.wash = train.dataset[c(8,9,2,10,11,12,13,14,15,16,6)]
train.dataset.dryer = train.dataset[c(8,9,2,10,11,12,13,14,15,16,7)]

#---- stratified sampling-------------------
set.seed(123)
stratas.ac <- partition(train.dataset.ac$ac, p = c(train.ac= 0.7,test.ac = 0.3),seed=1)
stratas.ev <- partition(train.dataset.ev$ev, p = c(train.ev = 0.7,test.ev = 0.3),seed=1)
stratas.oven <- partition(train.dataset.oven$oven, p = c(train.oven = 0.7,test.oven = 0.3),seed=1)
stratas.wash <- partition(train.dataset.wash$wash, p = c(train.wash = 0.7,test.wash = 0.3),seed=2)
stratas.dryer <- partition(train.dataset.dryer$dryer, p = c(train.dryer = 0.7,test.dryer = 0.3),seed=1)

train.ac <- train.dataset.ac[stratas.ac$train.ac, ]
test.ac <- train.dataset.ac[stratas.ac$test.ac, ]

train.ev <- train.dataset.ev[stratas.ev $train.ev , ]
test.ev <- train.dataset.ev[stratas.ev $test.ev , ]

train.oven <- train.dataset.oven[stratas.oven$train.oven, ]
test.oven <- train.dataset.oven[stratas.oven$test.oven, ]

train.wash <- train.dataset.wash[stratas.wash$train.wash, ]
test.wash <- train.dataset.wash[stratas.wash$test.wash, ]

train.dryer <- train.dataset.dryer[stratas.dryer$train.dryer, ]
test.dryer <- train.dataset.dryer[stratas.dryer$test.dryer, ]

####################-------- Cross validation --------------
#############----XGBoost AC----------

# defining seed to get the same result in each run
set.seed(1) 

train.ac$ac <- as.numeric(as.character(train.ac$ac))

# train dataset with non_categorical variables
train_ac_1 <- (train.ac[,c(3:10)] ^ 2)

# train dataset with categorical variables
train_ac_2 <- train.ac[, c(1,2)]

# converting the training dataset to matrix with hot encoding of categorical variables
x_training_ac = data.matrix(cbind(train_ac_1,train_ac_2))


# target variable
y_training_ac <- as.numeric(as.character(train.ac[,11]))

# test dataset with non_categorical variable
test_ac_1 <- (test.ac[,c(3:10)] ^ 2)

# test dataset with categorical variables
test_ac_2 <- test.ac[, c(1,2)]


# converting the testing dataset to matrix with hot encoding of categorical variables
x_testing_ac = data.matrix(cbind(test_ac_1, test_ac_2))

# convert the training dataset to xgb matrix
training_xgb_ac = xgb.DMatrix(data = x_training_ac, label = y_training_ac)

# convert the testing dataset to xgb matrix
testing_xgb_ac = xgb.DMatrix(data = x_testing_ac)

y_testing_ac <- as.numeric(as.character(test.ac[,11]))


# defining the parameters
parameter_list = list(objective = "binary:logistic", "scale_pos_weight" = 3.12)

# finding the best n_rounds
xgboost_cv_ac <- xgb.cv(params = parameter_list, data = training_xgb_ac, nrounds = 1000, nfold = 5,
                        print_every_n = 15, early_stopping_rounds = 30, maximize = F)

#------------------ XG Boost EV------------------------

# defining seed to get the same result in each run
set.seed(1) 
# train.ac$ac <- as.logical(train.ac$ac )

train.ev$ev <- as.numeric(as.character(train.ev$ev))

# train dataset with non_categorical variables
train_ev_1 <- (train.ev[,c(3:10)] ^ 2)

# train dataset with categorical variables
train_ev_2 <- train.ev[, c(1,2)]

# converting the training dataset to matrix with hot encoding of categorical variables
x_training_ev = data.matrix(cbind(train_ev_1,train_ev_2))


# target variable
y_training_ev <- as.numeric(as.character(train.ev[,11]))

# test dataset with non_categorical variable
test_ev_1 <- (test.ev[,c(3:10)] ^ 2)

# test dataset with categorical variables
test_ev_2 <- test.ev[, c(1,2)]


# converting the testing dataset to matrix with hot encoding of categorical variables
x_testing_ev= data.matrix(cbind(test_ev_1, test_ev_2))

# convert the training dataset to xgb matrix
training_xgb_ev = xgb.DMatrix(data = x_training_ev, label = y_training_ev)

# convert the testing dataset to xgb matrix
testing_xgb_ev = xgb.DMatrix(data = x_testing_ev)

y_testing_ev <- as.numeric(as.character(test.ev[,11]))

# defining the parameters
parameter_list = list(objective = "binary:logistic", "scale_pos_weight" =177.89)

# finding the best n_rounds
xgboost_cv_ev <- xgb.cv(params = parameter_list, data = training_xgb_ev, nrounds = 1000, nfold = 5,
                        print_every_n = 15, early_stopping_rounds = 30, maximize = F)

#------------------XG Boost Oven ------------------------------- 

# defining seed to get the same result in each run
set.seed(1) 

train.oven$oven <- as.numeric(as.character(train.oven$oven))

# train dataset with non_categorical variables
train_oven_1 <- (train.oven[,c(3:10)] ^ 2)

# train dataset with categorical variables
train_oven_2 <- train.oven[, c(1,2)]

# converting the training dataset to matrix with hot encoding of categorical variables
x_training_oven = data.matrix(cbind(train_oven_1,train_oven_2))


# target variable
y_training_oven <- as.numeric(as.character(train.oven[,11]))

# test dataset with non_categorical variable
test_oven_1 <- (test.oven[,c(3:10)] ^ 2)

# test dataset with categorical variables
test_oven_2 <- test.oven[, c(1,2)]


# converting the testing dataset to matrix with hot encoding of categorical variables
x_testing_oven= data.matrix(cbind(test_oven_1, test_oven_2))

# convert the training dataset to xgb matrix
training_xgb_oven = xgb.DMatrix(data = x_training_oven, label = y_training_oven)

# convert the testing dataset to xgb matrix
testing_xgb_oven = xgb.DMatrix(data = x_testing_oven)

y_testing_oven <- as.numeric(as.character(test.oven[,11]))

# defining the parameters
parameter_list = list(objective = "binary:logistic", "scale_pos_weight" = 69.13)

# finding the best n_rounds
xgboost_cv_oven <- xgb.cv(params = parameter_list, data = training_xgb_oven, nrounds = 1000, nfold = 5,
                          print_every_n = 15, early_stopping_rounds = 30, maximize = F)

#------------------ XG Boost wash ----------- 

# defining seed to get the same result in each run
set.seed(1) 
# train.ac$ac <- as.logical(train.ac$ac )

train.wash$wash <- as.numeric(as.character(train.wash$wash))

# train dataset with non_categorical variables
train_wash_1 <- (train.wash[,c( 3: 10)] ^ 2)

# train dataset with categorical variables
train_wash_2 <- train.wash[, c(1,2)]

# converting the training dataset to matrix with hot encoding of categorical variables
x_training_wash = data.matrix(cbind(train_wash_1,train_wash_2))


# target variable
y_training_wash <- as.numeric(as.character(train.wash[,11]))

# test dataset with non_categorical variable
test_wash_1 <- (test.wash[,c( 3:10)] ^ 2)

# test dataset with categorical variables
test_wash_2 <- test.wash[, c(1,2)]


# converting the testing dataset to matrix with hot encoding of categorical variables
x_testing_wash = data.matrix(cbind(test_wash_1, test_wash_2))

# convert the training dataset to xgb matrix
training_xgb_wash = xgb.DMatrix(data = x_training_wash, label = y_training_wash)

# convert the testing dataset to xgb matrix
testing_xgb_wash = xgb.DMatrix(data = x_testing_wash)

y_testing_wash <- as.numeric(as.character(test.wash[,11]))

# defining the parameters
parameter_list = list(objective = "binary:logistic", "scale_pos_weight" = 5.23)

# finding the best n_rounds
xgboost_cv_wash <- xgb.cv(params = parameter_list, data = training_xgb_wash, nrounds = 1000, nfold = 5,
                          print_every_n = 15, early_stopping_rounds = 30, maximize = F)

#---------------------------- XG boost dryer --------------------------

# defining seed to get the same result in each run
set.seed(1) 
# train.ac$ac <- as.logical(train.ac$ac )

train.dryer$dryer <- as.numeric(as.character(train.dryer$dryer))

# train dataset with non_categorical variables
train_dryer_1 <- (train.dryer[,c(3:10)] ^ 2)

# train dataset with categorical variables
train_dryer_2 <- train.dryer[, c(1,2)]

# converting the training dataset to matrix with hot encoding of categorical variables
x_training_dryer = data.matrix(cbind(train_dryer_1,train_dryer_2))


# target variable
y_training_dryer <- as.numeric(as.character(train.dryer[,11]))

# test dataset with non_categorical variable
test_dryer_1 <- (test.dryer[,c(3:10)] ^ 2)

# test dataset with categorical variables
test_dryer_2 <- test.dryer[, c(1,2)]


# converting the testing dataset to matrix with hot encoding of categorical variables
x_testing_dryer = data.matrix(cbind(test_dryer_1, test_dryer_2))

# convert the training dataset to xgb matrix
training_xgb_dryer = xgb.DMatrix(data = x_training_dryer, label = y_training_dryer)

# convert the testing dataset to xgb matrix
testing_xgb_dryer = xgb.DMatrix(data = x_testing_dryer)

y_testing_dryer <- as.numeric(as.character(test.dryer[,11]))

# defining the parameters
parameter_list = list(objective = "binary:logistic", "scale_pos_weight" = 29.77)

# finding the best n_rounds
xgboost_cv_dryer <- xgb.cv(params = parameter_list, data = training_xgb_dryer, nrounds = 1000, nfold = 5,
                           print_every_n = 15, early_stopping_rounds = 30, maximize = F)

#---------------- Submission --------------
##### for test dataset

# test dataset with non_categorical variable
test.dataset_1 <- (test.dataset[,c(2,5:11)] ^ 2)

# test dataset with categorical variables
test.dataset_2 <- test.dataset[, c(3,4)]


# converting the testing dataset to matrix with hot encoding of categorical variables
x_testing = data.matrix(cbind(test.dataset_1, test.dataset_2))

testing_xgb = xgb.DMatrix(data = x_testing)


# ------final train xg boost -----------------
## AC

# defining the maximum number of iteration
n_round = 386
# training the model
xgb_model_ac= xgb.train(data = training_xgb_ac , params = parameter_list, nrounds = n_round)

## EV
# defining the maximum number of iteration
n_round = 208
# training the model
xgb_model_ev= xgb.train(data = training_xgb_ev , params = parameter_list, nrounds = n_round)

## oven
# defining the maximum number of iteration
n_round = 533
# training the model
xgb_model_oven= xgb.train(data = training_xgb_oven , params = parameter_list, nrounds = n_round)

## wash
# defining the maximum number of iteration
n_round = 775
# training the model
xgb_model_wash= xgb.train(data = training_xgb_wash , params = parameter_list, nrounds = n_round)

## dryer
# defining the maximum number of iteration
n_round = 855

# training the model
xgb_model_dryer= xgb.train(data = training_xgb_dryer , params = parameter_list, nrounds = n_round)


#------------------ 6 submissions df -----------
#### 6 submission
ac = ifelse (predict(xgb_model_ac, newdata = testing_xgb ) > 1/64,1,0)
ev = ifelse (predict(xgb_model_ev, newdata = testing_xgb ) > 1/64,1,0)
oven = ifelse (predict(xgb_model_oven, newdata = testing_xgb ) > 1/64,1,0)
wash = ifelse (predict(xgb_model_wash, newdata = testing_xgb ) > 1/64,1,0)
dryer = ifelse (predict(xgb_model_dryer, newdata = testing_xgb ) > 1/64,1,0)

#-------------- final CSV------------
col_index = test.dataset[,1]
submission <- function(ac, ev, oven, wash, dryer){
  
  submission <- data.frame(col_index,ac, ev, oven, wash, dryer)
  return(submission)
}


submit = submission(ac, ev, oven, wash, dryer)

# writing the result to csv
write.csv(submit, "pred_labels.csv", row.names = F)

