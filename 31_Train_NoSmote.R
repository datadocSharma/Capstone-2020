############## Step 3: Initial Train ##################################
#- this will run on the original nfhs_train data 
# Run this for selected models - Logistic, C50, Random Forest, XGBOOst,  Logit Boost

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
tr_ctrl_ns <- trainControl(summaryFunction = twoClassSummary, 
                        verboseIter = TRUE, 
                        savePredictions =  TRUE, 
                        #sampling = "smote", 
                        method = "repeatedcv", 
                        number= 5, 
                        repeats = 2,
                        classProbs = TRUE, 
                        allowParallel = TRUE)

# Logit model 
#install.packages("stats")
library(stats)
logit_bench_ns <- microbenchmark(
  "logit_ns" = {logit_ns <- train(orig_recipe, data = nfhs_train,
                            method = "glm",
                            family = 'binomial',
                            metric = "ROC",
                            trControl = tr_ctrl_ns
  )}
  , times = 1)
saveRDS(logit_ns, "./logit_ns.rds")
logit_time_ns <- data.frame(summary(logit_bench_ns))
save(logit_time_ns, file="logit_time_ns.Rdata")
logit_vi_ns <- vi(logit_ns$finalModel)
save(logit_vi_ns, file = "logit_vi_ns.Rdata")


#### TREE BASED ALGOs 
# C5.0 - 3 parameters : trials, model, winnow
#install.packages("c50")
library(c50)
c50_bench_ns <- microbenchmark(
  "c50_ns" = {c50_ns <- train(nodum_recipe, data = nfhs_train,
                        method = "C5.0",
                        importance = TRUE,
                        n.trees = 100, 
                        metric = "ROC",
                        tuneLength = 8,
                        trControl = tr_ctrl_ns)}, 
  times = 1)
saveRDS(c50_ns, "./c50_ns.rds")
c50_time_ns <- data.frame(summary(c50_bench_ns))
save(c50_time_ns, file="c50_time_ns.Rdata")
c50_vi_ns <- vi(c50_ns$finalModel)
save(c50_vi_ns, file = "c50_vi_ns.Rdata")

# random forest performed consistently worse for datasets with high cardinality categorical variables.
# By one-hot encoding a categorical variable, we are inducing sparsity into the dataset which is undesirable.
#install.packages("ranger")
library(ranger)
rf_bench_ns <- microbenchmark(
  "rf_ns" = {rf_ns <- train(nodum_recipe, data = nfhs_train,
                      method = "ranger",
                      importance = "impurity",
                      num.trees = 100,
                      metric = "ROC",
                      tuneLength = 8,
                      trControl = tr_ctrl_ns)
  }, times = 1)
saveRDS(rf_ns, "./rf_ns.rds")
rf_time_ns <- data.frame(summary(rf_bench_ns))
save(rf_time_ns, file="rf_time_ns.Rdata")
rf_vi_ns <- vi(rf_ns$finalModel)
save(rf_vi_ns, file = "rf_vi_ns.Rdata")

## Boosted Logistic -  1 params: nIter
#install.packages("caTools")
library(caTools)
logitboost_bench_ns <- microbenchmark(
  "logitboost_ns" = {logitboost_ns <- train(orig_recipe, data = nfhs_train,
                                      method = "LogitBoost",
                                      metric = "ROC",
                                      tuneLength = 8,
                                      trControl = tr_ctrl_ns)
  }, times=1)
saveRDS(logitboost_ns, "./logitboost_ns.rds")
logitboost_time_ns <- data.frame(summary(logitboost_bench_ns))
save(logitboost_time_ns, file="logitboost_time_ns.Rdata")
#logitboost_vi_ns <- vi(logitboost_ns$finalModel)
#save(logitboost_vi, file = "logitboost_vi.Rdata")

## Extreme Gradient Boost - we choose xgbTree 
# tree and linear base learners yields comparable results for classification problems, 
#while tree learners are superior for regression problems
# tree based XGBoost models suffer from higher estimation variance compared to their linear counterparts.
# Cross validation through xgb.cv() - use this to find optimal nrounds
# Set evaluation metric to default "error"
# We choose to run a logistic model at each node -- use dummy encoding of categorical vars
# SMOTE over_rate of 1 --> balance out minority and majority class 
library(xgboost)
xgb_prep_ns <- recipe(husb_beat ~ ., data = nfhs_train) %>%
  step_nzv(all_predictors()) %>%               
  step_naomit(all_predictors()) %>%
  step_dummy(all_nominal(), -husb_beat) %>%
  #step_smote(husb_beat, over_ratio = 1) %>%
  step_integer(husb_beat) %>% prep()%>%juice()

xgb_prep_ns$husb_beat = xgb_prep_ns$husb_beat-1
X_ns <- as.matrix(xgb_prep_ns[setdiff(names(xgb_prep_ns), "husb_beat")])
Y_ns <- xgb_prep_ns$husb_beat
dtrain_ns <- xgb.DMatrix(data = X_ns,label = Y_ns) 
#default parameters - we let the algo use the default error rate as the metric
watchlist <- list(train=dtrain_ns, test=dtest)
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
cv_bench_ns <- microbenchmark(
  "cv_ns" = {xgbcv_ns <- xgb.cv( params = params, data = dtrain_ns, nrounds = 1000, 
                           nfold = 10, showsd = T, stratified = T, 
                           print.every.n = 10, early.stop.round = 100, maximize = F, seed = 888)},
  times = 1)

xgb_bench_ns <- microbenchmark(
  "xgb_ns" = {xgb_ns <- xgb.train(params = params, data = dtrain_ns, nrounds = xgbcv_ns$best_iteration)},
  times = 1)


xgb.save(xgb_ns, "xgb_ns.model")
xgb_time_ns <- data.frame(summary(xgb_bench_ns))
save(xgb_time_ns, file="xgb_time_ns.Rdata")
xgb_vi_ns <- vi(xgb_ns)
save(xgb_vi_ns, file = "xgb_vi_ns.Rdata")


