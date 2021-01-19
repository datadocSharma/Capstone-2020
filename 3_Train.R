############## Step 3: Initial Train ##################################
#- this will run on the original nfhs_train data 
# So it seems that caret-recipe combination is hard to run
# I needed to drop rows with any missing values from nfhs_train
# and validation for things to work
###############################################################
#require("devtools")
#install_github("tidymodels/themis")

library(tidyverse)
library(caret)
library(vip) 
library(recipes)
#library(pdp)
library(themis) # needed for step_smote
library(DMwR)
library(pROC)
library(microbenchmark)
library(doParallel)
library(doMC)


all_cores <- parallel::detectCores(logical = FALSE)
#registerDoParallel(cores = all_cores)
doMC::registerDoMC(all_cores-1)
setwd("~/Dropbox/Capstone")
#setwd("C:/Users/Gunjan/Desktop/Harvard Data Science/Capstone")
load("nfhs_train.Rdata")
load("validation.Rdata")

set.seed(888, sample.kind = "Rounding")
tr_ctrl <- trainControl(summaryFunction = twoClassSummary, 
                        verboseIter = TRUE, 
                        savePredictions =  TRUE, 
                        sampling = "smote", 
                        method = "repeatedcv", 
                        number= 5, 
                        repeats = 2,
                        classProbs = TRUE, 
                        allowParallel = TRUE)

# Logit model 
#install.packages("stats")
library(stats)
logit_bench <- microbenchmark(
  "logit" = {logit <- train(orig_recipe, data = nfhs_train,
               method = "glm",
               family = 'binomial',
               metric = "ROC",
               trControl = tr_ctrl
                )}
          , times = 1)
saveRDS(logit, "./logit.rds")
logit_time <- data.frame(summary(logit_bench))
save(logit_time, file="logit_time.Rdata")
logit_vi <- vi(logit$finalModel)
save(logit_vi, file = "logit_vi.Rdata")

# Ridge/LASSO
#install.packages("glmnet")
library(glmnet)
enet_bench2 <- microbenchmark(
"enet2" = {enet2 <- train(orig_recipe, data = nfhs_train,
                 method = "glmnet",
                 family = "binomial",
                 metric = "ROC",
                 tuneGrid = expand.grid(alpha = seq(0, 1, length.out = 5),
                                        lambda = seq(.5, 2, length.out = 5)),
                 trControl = tr_ctrl)}, 
              times = 1)
saveRDS(enet2, "./enet.rds")
enet_time <- data.frame(summary(enet_bench2))
save(enet_time, file="enet_time.Rdata")
enet_vi <- vi(enet2$finalModel)
save(enet_vi, file = "enet_vi.Rdata")

# MARS - 2 tuning parameters (max degree of interactions, and # terms to retain in final model)
#install.packages("earth")
library(earth)
hyper_grid <- expand.grid(degree = 1:3, nprune = seq(2, 100, length.out = 10) %>% floor())
mars_bench <- microbenchmark(
  "mars" = {mars <- train(orig_recipe, data = nfhs_train,
                          method = "earth",
                          metric = "ROC",
                          tuneGrid = hyper_grid,
                          trControl = tr_ctrl)}, 
            times = 1)
saveRDS(mars, "./mars.rds")
mars_time<- data.frame(summary(mars_bench))
save(mars_time, file="mars_time.Rdata")
mars_vi <- vi(mars$finalModel)
save(mars_vi, file = "mars_vi.Rdata")

## Non-linear Models - use ONE-HOT encoding of categorical variables

#### TREE BASED ALGOs 
# C5.0 - 3 parameters : trials, model, winnow
#install.packages("c50")
library(c50)
c50_bench <- microbenchmark(
"c50" = {c50 <- train(nodum_recipe, data = nfhs_train,
               method = "C5.0",
               importance = TRUE,
               n.trees = 100, 
               metric = "ROC",
               tuneLength = 8,
               trControl = tr_ctrl)}, 
        times = 1)
saveRDS(c50, "./c50.rds")
c50_time <- data.frame(summary(c50_bench))
save(c50_time, file="c50_time.Rdata")
c50_vi <- vi(c50$finalModel)
save(c50_vi, file = "c50_vi.Rdata")

# CART - one tuning parameter: cp
#install.packages("rpart")
library(rpart)
cart_bench <- microbenchmark(
  "cart" = {cart <- train(nodum_recipe, data = nfhs_train,
                        method = "rpart",
                        metric = "ROC",
                        tuneLength = 8,
                        trControl = tr_ctrl)}, 
  times = 1)
saveRDS(cart, "./cart.rds")
cart_time <- data.frame(summary(cart_bench))
save(cart_time, file="cart_time.Rdata")
cart_vi <- vi(cart$finalModel)
save(cart_vi, file = "cart_vi.Rdata")




# random forest performed consistently worse for datasets with high cardinality categorical variables.
# By one-hot encoding a categorical variable, we are inducing sparsity into the dataset which is undesirable.
#install.packages("ranger")
library(ranger)
rf_bench <- microbenchmark(
"rf" = {rf <- train(nodum_recipe, data = nfhs_train,
               method = "ranger",
               importance = "impurity",
               num.trees = 100,
               metric = "ROC",
               tuneLength = 8,
               trControl = tr_ctrl)
        }, times = 1)
saveRDS(rf, "./rf.rds")
rf_time <- data.frame(summary(rf_bench))
save(rf_time, file="rf_time.Rdata")
rf_vi <- vi(rf$finalModel)
save(rf_vi, file = "rf_vi.Rdata")

rf_class <- predict(rf, validation, models = rf$finalModel)
save(rf_class, file = "rf_class.Rdata")
rf_prob <- predict(rf, validation, models = rf$finalModel, type = 'prob')
save(rf_prob, file = "rf_prob.Rdata")
rf_cf <- confusionMatrix(rf_class, validation$husb_beat, positive="yes")

#ROCR package to calc and plot ROC curve and AUC
rf_p <- prediction(rf_prob[,2], validation$husb_beat) ##this is ROCR
rf_perf <- performance(rf_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
rf_roc <- plot(rf_perf, colorize=TRUE) + abline(a=0,b=1)
#savePlot("roc_lda.Rplot", type = "png")
rf_auc <- performance(rf_p, 'auc')
rf_auc.n<- rf_auc@y.values[[1]] # AUC number


####### ENSEMBLE MODELS 

## Bagged Cart (treebag): No params
#install.packages("ipred", "plyr")
library(ipred, plyr, e1071)
treebag_bench <- microbenchmark(
  "treebag" = {treebag <- train(orig_recipe, data = nfhs_train,
                                method = "treebag",
                                metric = "ROC",
                                tuneLength = 8,
                                trControl = tr_ctrl)}, 
  times = 1)
saveRDS(treebag, "./treebag.rds")
treebag_time <- data.frame(summary(treebag_bench))
save(treebag_time, file="treebag_time.Rdata")
#treebag_vi <- vi(treebag$finalModel)
#save(treebag_vi, file = "treebag_vi.Rdata")



## ADA: Does Boosted Classification Trees - 3 para: iter, maxdepth, nu
#install.packages("ada")
library(ada)
ada_bench <- microbenchmark(
  "ada" = {ada <- train(orig_recipe, data = nfhs_train,
                                method = "ada",
                                metric = "ROC",
                                tuneLength = 8,
                                trControl = tr_ctrl)
  }, times =1)
saveRDS(ada, "./ada.rds")
ada_time <- data.frame(summary(ada_bench))
save(ada_time, file="ada_time.Rdata")
#ada_vi <- vi(ada$finalModel)
#save(ada_vi, file = "ada_vi.Rdata")

## Boosted Logistic -  1 params: nIter
#install.packages("caTools")
library(caTools)
logitboost_bench <- microbenchmark(
  "logitboost" = {logitboost <- train(orig_recipe, data = nfhs_train,
                              method = "LogitBoost",
                              metric = "ROC",
                              tuneLength = 8,
                              trControl = tr_ctrl)
                  }, times=1)
saveRDS(logitboost, "./logitboost.rds")
logitboost_time <- data.frame(summary(logitboost_bench))
save(logitboost_time, file="logitboost_time.Rdata")
logitboost_vi <- vi(logitboost$finalModel)
#save(logitboost_vi, file = "logitboost_vi.Rdata")

## Gradient Boosting Machine -  4 params: 
#install.packages("gbm")
library(gbm)
gbm_bench <- microbenchmark(
"gbm" = {gbm <- train(orig_recipe, data = nfhs_train,
                            method = "gbm",
                            metric = "ROC",
                            tuneLength = 8,
                            trControl = tr_ctrl)
            }, times=1)
saveRDS(gbm, "./gbm.rds")
gbm_time <- data.frame(summary(gbm_bench))
save(gbm_time, file="gbm_time.Rdata")
gbm_vi <- vi(gbm$finalModel)
save(gbm_vi, file = "gbm_vi.Rdata")

## Extreme Gradient Boost - we choose xgbTree 
# tree and linear base learners yields comparable results for classification problems, 
#while tree learners are superior for regression problems
 # tree based XGBoost models suffer from higher estimation variance compared to their linear counterparts.
# Cross validation through xgb.cv() - use this to find optimal nrounds
# Set evaluation metric to default "error"
# We choose to run a logistic model at each node -- use dummy encoding of categorical vars
# SMOTE over_rate of 1 --> balance out minority and majority class 
library(xgboost)
xgb_prep <- recipe(husb_beat ~ ., data = nfhs_train) %>%
  step_nzv(all_predictors()) %>%               
  step_naomit(all_predictors()) %>%
  step_dummy(all_nominal(), -husb_beat) %>%
  step_smote(husb_beat, over_ratio = 1) %>%
  step_integer(husb_beat) %>% prep()%>%juice()

xgb_prep$husb_beat = xgb_prep$husb_beat-1
X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "husb_beat")])
Y <- xgb_prep$husb_beat
dtrain <- xgb.DMatrix(data = X,label = Y) 
#default parameters - we let the algo use the default error rate as the metric
watchlist <- list(train=dtrain, test=dtest)
params <- list(verbosity = 1,
               disable_default_eval_metric =1, 
               booster = "gbtree", 
               objective = "binary:logistic", 
               eta=0.3, 
               gamma=0, 
               max_depth=6, 
               min_child_weight=1, 
               subsample=1, 
               colsample_bytree=1,
               colsample_bylevel  = 1,                 
               lambda             = 1,                 
               alpha              = 0,
               eval_metric = "error")

# Run CV to decide nfolds - best iteration = 200
cv_bench <- microbenchmark(
  "cv" = {xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 1000, 
                           nfold = 10, showsd = T, stratified = T, 
                           print.every.n = 10, early.stop.round = 100, maximize = F, seed = 888)},
  times = 1)

xgb_bench <- microbenchmark(
  "xgb_def" = {xgb_def <- xgb.train(params = params, data = dtrain, nrounds = xgbcv$best_iteration)},
  times = 1)


xgb.save(xgb_def, "xgb_def.model")
xgb_time <- data.frame(summary(xgb_bench))
save(xgb_time, file="xgb_time.Rdata")
xgb_def_vi <- vi(xgb_def)
save(xgb_def_vi, file = "xgb_def_vi.Rdata")



## NOT WORKING  
## Conditional Inference Forest - 1 parameter: mtry
#install.packages("party")
# Do not use the allowParallel option in trainControl for any partykit algos
set.seed(888, sample.kind = "Rounding")
party_recipe <- recipe(husb_beat ~ ., data = nfhs_train) %>%
  step_nzv(all_predictors()) %>%               
  step_naomit(all_predictors()) %>%
  step_dummy(all_nominal(), -husb_beat) 

party_ctrl <- trainControl(summaryFunction = twoClassSummary, 
                           verboseIter = TRUE, 
                           savePredictions =  TRUE, 
                           sampling = "smote", 
                           method = "repeatedcv", 
                           number= 5, 
                           repeats = 2,
                           classProbs = TRUE) 


library(partykit)
library(pROC)
cforest_bench <- microbenchmark(
  "cforest" = {cforest <- train(husb_beat ~., data = nfhs_train,
                                method = "cforest",
                                metric = "ROC",
                                tuneLength = 2,
                                trControl = party_ctrl)
  }, times =1)
saveRDS(cforest, "./cforest.rds")
cforest_time <- data.frame(summary(cforest_bench))
save(cforest_time, file="cforest_time.Rdata")
cforest_vi <- vi(cforest$finalModel)
save(cforest_vi, file = "cforest_vi.Rdata")

# CTREE: conditional inference trees - TWO tuning parameters: maxdepth, mincriterion
#install.packages("party")
library(partykit)
ctree_bench <- microbenchmark(
  "ctree" = {ctree <- train(husb_beat ~., data = smote_tr_data,
                            method = "ctree",
                            metric = "ROC",
                            tuneLength = 8,
                            trControl = party_ctrl
  )}, 
  times = 1)
saveRDS(ctree, "./ctree.rds")
ctree_time <- data.frame(summary(ctree_bench))
save(ctree_time, file="ctree_time.Rdata")
#ctree_vi <- vi(ctree$finalModel)
#save(ctree_vi, file = "ctree_vi.Rdata")



##Naive Bayes - 
#At a high level, Naive Bayes tries to classify instances based on the 
#probabilities of previously seen attributes/instances, 
#assuming complete attribute independence.
# 2 parameters - laplace, adjust
#install.packages("e1071")
library(e1071)
naive_bench <- microbenchmark(
  "naive" = {naive <- train(orig_recipe, data = nfhs_train,
                            method = "naive_bayes",
                            usekernel = FALSE,
                            metric = "ROC",
                            tuneLength = 8,
                            trControl = tr_ctrl)},  
  times = 1)
saveRDS(naive, "./naive.rds")
# to see confusion matrix of cross-validation
confusionMatrix(naive)
naive_time <- data.frame(summary(naive_bench))
save(naive_time, file="naive_time.Rdata")
#naive_vi <- vi(naive$finalModel)
#save(naive_vi, file = "naive_vi.Rdata")

# TAKES TOO LONG 
#Support Vector Machine - does not have simple var importance
#install.packages("kernlab")
library(kernlab)
svm_bench <- microbenchmark(
  "svm" = {svm <- train(ohe_recipe, data = nfhs_train,
                        method = "svmRadial",
                        metric = "ROC",
                        tuneLength = 8,
                        trControl = tr_ctrl)},  
  times = 1)
saveRDS(svm, "./svm.rds")
#confusionMatrix(svm) - to get cv results
svm_time <- data.frame(summary(svm_bench))
save(svm_time, file="svm_time.Rdata")
#svm_vi <- vi(svm$finalModel)
#save(svm_vi, file = "svm_vi.Rdata")

