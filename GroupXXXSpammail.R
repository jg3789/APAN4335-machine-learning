# Group XXX midterm project
# Team Members: Junmei Gu/ Li Wan/ Zhejun Zhang
# Due Date: March 12th, 2017
# Project Topic: 3. Spam Email Detection
# ---------------------------------------------------------------------------------
# Installed packages
install.packages("caTools") 
library(caTools) 
install.packages("e1071")
library(e1071)
install.packages("caret")
library(caret)
library(ROCR)
# Load data
# Dataset is stored in the same work directory 
spam_dataset = read.csv("spambase.csv")

###################################################################################
########################## 1.DATA CLEAN AND EXPLORATION ###########################
###################################################################################

########################## 1.1 Check Data Type ####################################

# Our dataset has 57 colunms of continuous variables and 1 column of nominal variable
# First, let's check if all variables have the correct type
check_numeric_type = sapply(spam_dataset, is.numeric)
check_numeric_type

# From the results, we can tell all 58 variables having continuous type
# We have to convert the last column from continuous to nominal
spam_dataset$spam = as.factor(spam_dataset$spam)
is.factor(spam_dataset$spam)

########################## 1.2 Check Missing Values ###############################
sum(is.na(spam_dataset))
# The result is 0. Therefore, no missing data exits in the spam email dataset

########################## 1.3 Check Duplicate Tuples #############################
sum(duplicated(spam_dataset))
# We found 391 duplicated rows

# Update dataset without duplicated tuples
spam_dataset = spam_dataset[!duplicated(spam_dataset),]
# Now we have 4210 rows instead of 4601 rows

########################## 1.4 Check Outliers #####################################
predictors = spam_dataset[,-58]
sapply(predictors,function(x) sum(length(boxplot.stats(x)$out)))
# By looking at the number of the total outliers for each variable, no human error is detected

########################## 1.5 Statistics Exploration #############################
summary(predictors)
sapply(predictors,function (x) var(x))

###################################################################################
########################## 2.CLASSIFICATION METHODS ###############################
###################################################################################
# Split dataset into train and test using ratio 0.8: 0.2
tuples = spam_dataset[,1]
msk = sample.split(tuples, SplitRatio=0.8)
trainset = spam_dataset[msk,]
testset = spam_dataset[!msk,]
########################## 2.1 logistic Regression ################################
log_model = glm( spam ~., data = trainset, family = binomial(logit))
pred = predict(log_model,trainset[,1:57])
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
pred = as.factor(pred)

confusionMatrix(pred,trainset$spam)
# Train Set Acuracy = 0.9184

pred = predict(log_model,testset[,1:57])
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
pred = as.factor(pred)

confusionMatrix(pred,testset$spam)
# Test Set Accuracy = 0.9158
# The following plots the ROC curves for logistic regression
pred1 <- prediction(predict(log_model), trainset$spam)
perf1 <- performance(pred1,"tpr","fpr")

auc = performance(pred1, measure = "auc")  
auc@y.values[[1]] 
title = paste("Logistic ROC curve AUC:", round(auc@y.values[[1]],digits = 4))
plot(perf1,sub = title)
########################## 2.2 KNN Clustering #####################################
library(class)
predknn = knn(trainset[,-58],trainset[,-58],trainset$spam,k=2)
predknn = as.factor(predknn)
confusionMatrix(predknn,trainset$spam)

predknn = knn(trainset[,-58],testset[,-58],trainset$spam,k=2)
predknn = as.factor(predknn)

confusionMatrix(predknn,testset$spam)
# Test Set Accuracy = 0.755
# The following plots the ROC curves for KNN
trainset_=trainset
library(plyr)
revalue(trainset_$spam, c("0"="-1"))

testset_=testset
revalue(testset_$spam, c("0"="-1"))

predknn = knn(trainset[,-58],testset[,-58],trainset$spam,k=2,prob = TRUE)
probknn <- attr(predknn, "prob")
#prob <- 2*ifelse(pred == "-1", 1-prob, prob) - 1
pred_knn <- prediction(probknn,testset_[,58])
pref_knn <- performance(pred_knn, "tpr", "fpr")
auc = performance(pred_knn, measure = "auc")  
auc@y.values[[1]] 

title = paste("KNN ROC curve AUC:", round(auc@y.values[[1]] ,digits = 4))
plot(pref_knn,sub = title)
########################## 2.3 RBF Supported Vector Machine #######################
svm_model <- svm(spam ~ ., data=trainset)
# By running the general svm model, we found it is using radial SVM
# Now we can start to find the best cost and gamma values

library(doMC)
registerDoMC()
svm_tune = tune(svm, spam ~., data = trainset, kernel="radial", ranges=list(cost=2^(2:4), gamma=10^(-3:-1)))
print(svm_tune)
# We found best parameters by tune: cost = 8, gamme = 0.01 

# Now lets build the RBF SVM model
svm_model_after_tune = svm(spam ~ ., data=trainset, kernel="radial", cost=8, gamma=0.01, prob=TRUE)

summary(svm_model_after_tune)

# Evaluation of the svm model using confusion matrix
pred = predict(svm_model_after_tune,testset[,1:57])
confusionMatrix(pred,testset$spam)

# Test Set Accuracy = 0.9265
pred = predict(svm_model_after_tune,trainset[,1:57])
confusionMatrix(pred,trainset$spam)

# The following plots the ROC curves for SVM
svm.preds<-predict(svm_model_after_tune, testset[,1:57], probability=TRUE)
svm.rocr<-prediction(attr(svm.preds,"probabilities")[,2], testset[,58] == "0")
svm.perf<-performance(svm.rocr, measure = "tpr", x.measure = "fpr")

svm.auc<-as.numeric(performance(svm.rocr, measure = "auc", x.measure
                                      = "cutoff")@ y.values)

title = paste("SVM ROC curve AUC:", round(svm.auc,digits = 4))
plot(svm.perf,sub = title)






