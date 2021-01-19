############################### 
#STEP 4: HARVEST MODEL RESULTS
# Load the indv model libraries 
###############################

library(tidyverse)
library(caret) # make sure to load this while using saved models
#library(vip) ## Note: vip does not work on saved model object. 
library(recipes)
library(pdp)
library(themis)
library(ROCR)
library(resample)
library(magrittr)
library(knitr)

#setwd("~/Dropbox/Capstone")
setwd("C:/Users/Gunjan/Desktop/Harvard Data Science/Capstone")
load("nfhs_train.Rdata")
load("validation.Rdata")

### Collect all Time Stats
load("logit_time.Rdata")
logit_time <- as.matrix(as.numeric(logit_time))[2,1]
load("enet_time.Rdata")
enet_time <- as.matrix(as.numeric(enet_time))[2,1]
load("mars_time.Rdata")
mars_time <- as.matrix(as.numeric(mars_time))[2,1]
load("c50_time.Rdata")
c50_time <- as.matrix(as.numeric(c50_time))[2,1]
load("cart_time.Rdata")
cart_time <- as.matrix(as.numeric(cart_time))[2,1]
load("rf_time.Rdata")
rf_time <- as.matrix(as.numeric(rf_time))[2,1]
load("treebag_time.Rdata")
treebag_time <- as.matrix(as.numeric(treebag_time))[2,1]
load("ada_time.Rdata")
ada_time <- as.matrix(as.numeric(ada_time))[2,1]
load("logitboost_time.Rdata")
logitboost_time <- as.matrix(as.numeric(logitboost_time))[2,1]
load("gbm_time.Rdata")
gbm_time <- as.matrix(as.numeric(gbm_time))[2,1]
load("xgb_time.Rdata")
xgb_time <- as.matrix(as.numeric(xgb_time))[2,1]

overall_time <- rbind(logit_time, enet_time,  mars_time, c50_time, 
                      cart_time, rf_time, treebag_time, ada_time, 
                      logitboost_time, gbm_time, xgb_time)
overall_time <- round(overall_time *(1/3600), 2)
modList <- c("Logistic",  "Elastic Net", "C50", "CART", "Random Forest", "Tree Bag", 
             "Ada Boost", "Logistic Boost", "Gradient Boost Machine", "Extreme Gradient Boost")
rownames(overall_time) <- modList
colnames(overall_time) <- c(" Time in Hours")

```{r time, echo = FALSE}
kable(overall_time)
```

### COLLECT ALL CONFUSION MATRICES

logit <- readRDS("./logit.rds")
library(stats)
#Make predictions and create confusion matrix
logit_class50 <- predict(logit, validation, models = logit$finalModel)
logit_prob <- predict(logit, validation, models = logit$finalModel, type = 'prob')
logit_class30 <- (as.factor(logit_prob[,2]>0.3))
levels(logit_class30)[levels(logit_class30)=="FALSE"] <- "no"
levels(logit_class30)[levels(logit_class30)=="TRUE"] <- "yes"
logit_cf50 <- confusionMatrix(logit_class50, validation$husb_beat, positive="yes")
logit_cf30 <- confusionMatrix(logit_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
logit_p <- prediction(logit_prob[,2], validation$husb_beat) ##this is ROCR
logit_perf <- performance(logit_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
#logit_roc <- plot(logit_perf, colorize=TRUE) + abline(a=0,b=1)
#savePlot("roc_lda.Rplot", type = "png")
logit_auc <- performance(logit_p, 'auc')
logit_auc.n<- logit_auc@y.values[[1]] # AUC number
LOGIT50 <- as.table(as.matrix(logit_cf50, what = "classes")) %>% rbind(logit_auc.n)
LOGIT30 <- as.table(as.matrix(logit_cf30, what = "classes")) %>% rbind(logit_auc.n) 
save(LOGIT50, LOGIT30 , file ="LOGIT.RData")


## GLMNET
enet <- readRDS("./enet.rds")
library(glmnet)
#Make predictions and create confusion matrix
  enet_class50 <- predict(enet, validation, models = enet$finalModel)
  enet_prob <- predict(enet, validation, models = enet$finalModel, type = 'prob')
  enet_class30 <- (as.factor(enet_prob[,2]>0.3))
  levels(enet_class30)[levels(enet_class30)=="FALSE"] <- "no"
  levels(enet_class30)[levels(enet_class30)=="TRUE"] <- "yes"
  enet_cf50 <- confusionMatrix(enet_class50, validation$husb_beat, positive="yes")
  enet_cf30 <- confusionMatrix(enet_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
  enet_p <- prediction(enet_prob[,2], validation$husb_beat) ##this is ROCR
  enet_perf <- performance(enet_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
#enet_roc <- plot(enet_perf, colorize=TRUE) + abline(a=0,b=1)
#savePlot("roc_lda.Rplot", type = "png")
  enet_auc <- performance(enet_p, 'auc')
  enet_auc.n<- enet_auc@y.values[[1]] # AUC number
ENET50 <- as.table(as.matrix(enet_cf50, what = "classes")) %>% rbind(enet_auc.n)
ENET30 <- as.table(as.matrix(enet_cf30, what = "classes")) %>% rbind(enet_auc.n) 
save(ENET50, ENET30 , file ="ENET.RData")

## MARS
mars <- readRDS("./mars.rds")
library(earth)
#Make predictions and create confusion matrix
  mars_class50 <- predict(mars, validation, models = mars$finalModel)
  mars_prob <- predict(mars, validation, models = mars$finalModel, type = 'prob')
  mars_class30 <- (as.factor(mars_prob[,2]>0.3))
  levels(mars_class30)[levels(mars_class30)=="FALSE"] <- "no"
  levels(mars_class30)[levels(mars_class30)=="TRUE"] <- "yes"
  mars_cf50 <- confusionMatrix(mars_class50, validation$husb_beat, positive="yes")
  mars_cf30 <- confusionMatrix(mars_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
  mars_p <- prediction(mars_prob[,2], validation$husb_beat) ##this is ROCR
  mars_perf <- performance(mars_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
  #mars_roc <- plot(mars_perf, colorize=TRUE) + abline(a=0,b=1)
  #savePlot("roc_lda.Rplot", type = "png")
  mars_auc <- performance(mars_p, 'auc')
  mars_auc.n<- mars_auc@y.values[[1]] # AUC number
MARS50 <- as.table(as.matrix(mars_cf50, what = "classes")) %>% rbind(mars_auc.n)
MARS30 <- as.table(as.matrix(mars_cf30, what = "classes")) %>% rbind(mars_auc.n) 
save(MARS50, MARS30 , file ="MARS.RData")

# C.50
library(c50)
c50 <- readRDS("./c50.rds")
#Make predictions and create confusion matrix
  c50_class50 <- predict(c50, validation, models = c50$finalModel)
  c50_prob <- predict(c50, validation, models = c50$finalModel, type = 'prob')
  c50_class30 <- (as.factor(c50_prob[,2]>0.3))
  levels(c50_class30)[levels(c50_class30)=="FALSE"] <- "no"
  levels(c50_class30)[levels(c50_class30)=="TRUE"] <- "yes"
  c50_cf50 <- confusionMatrix(c50_class50, validation$husb_beat, positive="yes")
  c50_cf30 <- confusionMatrix(c50_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
  c50_p <- prediction(c50_prob[,2], validation$husb_beat) ##this is ROCR
  c50_perf <- performance(c50_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
  #c50_roc <- plot(c50_perf, colorize=TRUE) + abline(a=0,b=1)
  #savePlot("roc_lda.Rplot", type = "png")
  c50_auc <- performance(c50_p, 'auc')
  c50_auc.n<- c50_auc@y.values[[1]] # AUC number
C5050 <- as.table(as.matrix(c50_cf50, what = "classes")) %>% rbind(c50_auc.n)
C5030 <- as.table(as.matrix(c50_cf30, what = "classes")) %>% rbind(c50_auc.n) 
save(C5050, C5030 , file ="C50.RData")


# CART
library(rpart)
cart <- readRDS("./cart.rds")
#Make predictions and create confusion matrix
  cart_class50 <- predict(cart, validation, models = cart$finalModel)
  cart_prob <- predict(cart, validation, models = cart$finalModel, type = 'prob')
  cart_class30 <- (as.factor(cart_prob[,2]>0.3))
  levels(cart_class30)[levels(cart_class30)=="FALSE"] <- "no"
  levels(cart_class30)[levels(cart_class30)=="TRUE"] <- "yes"
  cart_cf50 <- confusionMatrix(cart_class50, validation$husb_beat, positive="yes")
  cart_cf30 <- confusionMatrix(cart_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
  cart_p <- prediction(cart_prob[,2], validation$husb_beat) ##this is ROCR
  cart_perf <- performance(cart_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
  #cart_roc <- plot(cart_perf, colorize=TRUE) + abline(a=0,b=1)
  #savePlot("roc_lda.Rplot", type = "png")
  cart_auc <- performance(cart_p, 'auc')
  cart_auc.n<- cart_auc@y.values[[1]] # AUC number
CART50 <- as.table(as.matrix(cart_cf50, what = "classes")) %>% rbind(cart_auc.n)
CART30 <- as.table(as.matrix(cart_cf30, what = "classes")) %>% rbind(cart_auc.n) 
save(CART50, CART30 , file ="CART.RData")

# RANDOM FOREST
library(ranger)
rf <- readRDS("./rf.rds")
#Make predictions and create confusion matrix
  rf_class50 <- predict(rf, validation, models = rf$finalModel)
  rf_prob <- predict(rf, validation, models = rf$finalModel, type = 'prob')
  rf_class30 <- (as.factor(rf_prob[,2]>0.3))
  levels(rf_class30)[levels(rf_class30)=="FALSE"] <- "no"
  levels(rf_class30)[levels(rf_class30)=="TRUE"] <- "yes"
  rf_cf50 <- confusionMatrix(rf_class50, validation$husb_beat, positive="yes")
  rf_cf30 <- confusionMatrix(rf_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
  rf_p <- prediction(rf_prob[,2], validation$husb_beat) ##this is ROCR
  rf_perf <- performance(rf_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
  #rf_roc <- plot(rf_perf, colorize=TRUE) + abline(a=0,b=1)
  #savePlot("roc_lda.Rplot", type = "png")
  rf_auc <- performance(rf_p, 'auc')
  rf_auc.n<- rf_auc@y.values[[1]] # AUC number
RF50 <- as.table(as.matrix(rf_cf50, what = "classes")) %>% rbind(rf_auc.n)
RF30 <- as.table(as.matrix(rf_cf30, what = "classes")) %>% rbind(rf_auc.n) 
save(RF50, RF30 , file ="RF.RData")

# TREE BAG
library(ipred, plyr, e1071)
treebag <- readRDS("./treebag.rds")
#Make predictions and create confusion matrix
  treebag_class50 <- predict(treebag, validation, models = treebag$finalModel)
  treebag_prob <- predict(treebag, validation, models = treebag$finalModel, type = 'prob')
  treebag_class30 <- (as.factor(treebag_prob[,2]>0.3))
  levels(treebag_class30)[levels(treebag_class30)=="FALSE"] <- "no"
  levels(treebag_class30)[levels(treebag_class30)=="TRUE"] <- "yes"
  treebag_cf50 <- confusionMatrix(treebag_class50, validation$husb_beat, positive="yes")
  treebag_cf30 <- confusionMatrix(treebag_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
  treebag_p <- prediction(treebag_prob[,2], validation$husb_beat) ##this is ROCR
  treebag_perf <- performance(treebag_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
  #treebag_roc <- plot(treebag_petreebag, colorize=TRUE) + abline(a=0,b=1)
  #savePlot("roc_lda.Rplot", type = "png")
  treebag_auc <- performance(treebag_p, 'auc')
  treebag_auc.n<- treebag_auc@y.values[[1]] # AUC number
TREEBAG50 <- as.table(as.matrix(treebag_cf50, what = "classes")) %>% rbind(treebag_auc.n)
TREEBAG30 <- as.table(as.matrix(treebag_cf30, what = "classes")) %>% rbind(treebag_auc.n) 
save(TREEBAG50, TREEBAG30 , file ="TREEBAG.RData")

# ada boost
library(ada)
ada <- readRDS("./ada.rds")
#Make predictions and create confusion matrix
  ada_class50 <- predict(ada, validation, models = ada$finalModel)
  ada_prob <- predict(ada, validation, models = ada$finalModel, type = 'prob')
  ada_class30 <- (as.factor(ada_prob[,2]>0.3))
  levels(ada_class30)[levels(ada_class30)=="FALSE"] <- "no"
  levels(ada_class30)[levels(ada_class30)=="TRUE"] <- "yes"
  ada_cf50 <- confusionMatrix(ada_class50, validation$husb_beat, positive="yes")
  ada_cf30 <- confusionMatrix(ada_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
  ada_p <- prediction(ada_prob[,2], validation$husb_beat) ##this is ROCR
  ada_perf <- performance(ada_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
  #ada_roc <- plot(ada_peada, colorize=TRUE) + abline(a=0,b=1)
  #savePlot("roc_lda.Rplot", type = "png")
  ada_auc <- performance(ada_p, 'auc')
  ada_auc.n<- ada_auc@y.values[[1]] # AUC number
ADA50 <- as.table(as.matrix(ada_cf50, what = "classes")) %>% rbind(ada_auc.n)
ADA30 <- as.table(as.matrix(ada_cf30, what = "classes")) %>% rbind(ada_auc.n) 
save(ADA50, ADA30 , file ="ADA.RData")

# Logistic Boost
library(caTools)
logitboost <- readRDS("./logitboost.rds")
#Make predictions and create confusion matrix
  logitboost_class50 <- predict(logitboost, validation, models = logitboost$finalModel)
  logitboost_prob <- predict(logitboost, validation, models = logitboost$finalModel, type = 'prob')
  logitboost_class30 <- (as.factor(logitboost_prob[,2]>0.3))
  levels(logitboost_class30)[levels(logitboost_class30)=="FALSE"] <- "no"
  levels(logitboost_class30)[levels(logitboost_class30)=="TRUE"] <- "yes"
  logitboost_cf50 <- confusionMatrix(logitboost_class50, validation$husb_beat, positive="yes")
  logitboost_cf30 <- confusionMatrix(logitboost_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
  logitboost_p <- prediction(logitboost_prob[,2], validation$husb_beat) ##this is ROCR
  logitboost_perf <- performance(logitboost_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
  #logitboost_roc <- plot(logitboost_pelogitboost, colorize=TRUE) + abline(a=0,b=1)
  #savePlot("roc_lda.Rplot", type = "png")
  logitboost_auc <- performance(logitboost_p, 'auc')
  logitboost_auc.n<- logitboost_auc@y.values[[1]] # AUC number
LOGITBOOST50 <- as.table(as.matrix(logitboost_cf50, what = "classes")) %>% rbind(logitboost_auc.n)
LOGITBOOST30 <- as.table(as.matrix(logitboost_cf30, what = "classes")) %>% rbind(logitboost_auc.n) 
save(LOGITBOOST50, LOGITBOOST30 , file ="LOGITBOOST.RData")


# GBM
library(gbm)
gbm <- readRDS("./gbm.rds")
#Make predictions and create confusion matrix
  gbm_class50 <- predict(gbm, validation, models = gbm$finalModel)
  gbm_prob <- predict(gbm, validation, models = gbm$finalModel, type = 'prob')
  gbm_class30 <- (as.factor(gbm_prob[,2]>0.3))
  levels(gbm_class30)[levels(gbm_class30)=="FALSE"] <- "no"
  levels(gbm_class30)[levels(gbm_class30)=="TRUE"] <- "yes"
  gbm_cf50 <- confusionMatrix(gbm_class50, validation$husb_beat, positive="yes")
  gbm_cf30 <- confusionMatrix(gbm_class30, validation$husb_beat, positive="yes")
#ROCR package to calc and plot ROC curve and AUC
  gbm_p <- prediction(gbm_prob[,2], validation$husb_beat) ##this is ROCR
  gbm_perf <- performance(gbm_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
  #gbm_roc <- plot(gbm_pegbm, colorize=TRUE) + abline(a=0,b=1)
  #savePlot("roc_lda.Rplot", type = "png")
  gbm_auc <- performance(gbm_p, 'auc')
  gbm_auc.n<- gbm_auc@y.values[[1]] # AUC number
GBM50 <- as.table(as.matrix(gbm_cf50, what = "classes")) %>% rbind(gbm_auc.n)
GBM30 <- as.table(as.matrix(gbm_cf30, what = "classes")) %>% rbind(gbm_auc.n) 
save(GBM50, GBM30 , file ="GBM.RData")

# XGBOOST
library(xgboost)
xgb <- xgb.load("xgb_def.model")
xgb_prep_test <- recipe(husb_beat ~ ., data = validation) %>%
  step_nzv(all_predictors()) %>%               
  step_naomit(all_predictors()) %>%
  step_dummy(all_nominal(), -husb_beat) %>%
  #step_smote(husb_beat, over_ratio = 1) %>%
  step_integer(husb_beat) %>% prep()%>%juice()

xgb_prep_test$husb_beat = xgb_prep_test$husb_beat-1
X_test <- as.matrix(xgb_prep_test[setdiff(names(xgb_prep_test), "husb_beat")])
Y_test <- xgb_prep_test$husb_beat
dtest <- xgb.DMatrix(data = X_test,label = Y_test)

  xgb_prob <- predict(xgb, dtest)
  xgb_class50 <- as.factor(xgb_prob > 0.5)
  levels(xgb_class50) <- c("no", "yes")
  xgb_cf50 <- confusionMatrix(xgb_class50, validation$husb_beat, positive="yes")
  xgb_class30 <- as.factor(xgb_prob > 0.3)
  levels(xgb_class30) <- c("no", "yes")
  xgb_cf30 <- confusionMatrix(xgb_class30, validation$husb_beat, positive="yes")

#ROCR package to calc and plot ROC curve and AUC
  xgb_p <- prediction(xgb_prob, validation$husb_beat) ##this is ROCR
  xgb_perf <- performance(xgb_p, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
  xgb_roc <- plot(xgb_perf, colorize=TRUE) + abline(a=0,b=1)
  #savePlot("roc_lda.Rplot", type = "png")
  xgb_auc <- performance(xgb_p, 'auc')
  xgb_auc.n<- xgb_auc@y.values[[1]] # AUC number
XGB50 <- as.table(as.matrix(xgb_cf50, what = "classes")) %>% rbind(xgb_auc.n)
XGB30 <- as.table(as.matrix(xgb_cf30, what = "classes")) %>% rbind(xgb_auc.n) 
save(XGB50, XGB30 , file ="XGB.RData")


## Comparing CV results across models
model_list <- list(Logistic = logit, GLMNET = enet, MARS = mars, 
                   C50 = c50, CART = cart, RandomForest = rf, TreeBag = treebag, 
                   ADA = ada, LogitBoost = logitboost,  GBM = gbm)
res <- resamples(model_list)
cv_results <- summary(res)
save(cv_results, file= "cv_results.RData")
# Write this table to a comma separated .txt file:    
write.table(t, file = "CV_results.txt", sep = ",", quote = FALSE, row.names = F)
## Need to figure out kabel for this table
```{r Resamples Table, echo=FALSE}
library(knitr)
kable(summary(resamples(model_list)))
```

###########################################
# OVERALL CONFUSION MATRIX 
# - need to figure out how to name the rows with the model names
##########################################
#load(LOGIT, GLM, MARS, C50, CART, RF, TREEBAG, ADA, LOGITBOOST, GBM, XGB)
overall_cf50 <- cbind(LOGIT50, ENET50, MARS50, C5050, 
                      CART50, RF50, TREEBAG50, ADA50, 
                      LOGITBOOST50, GBM50, XGB50)
modList <- c("Logistic", "Elastic Net", "MARS", "C50", "CART", "Random Forest", 
             "Tree Bag", "Ada Boost", "Logit Boost", "Gradient Boost", "XGBoost")
colnames(overall_cf50) <- modList
rownames(overall_cf50) <- c("Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value", "Precision", 
                    "Recall", "F1", "Prevalence", "Detection Rate", "Detection Prevalence", 
                    "Balanced Accuracy", "AUC")

save(overall_cf50, file="overall_cf50.RData")
write.table(overall_cf50, file = "overall_cf50.txt", sep = ",", quote = FALSE, row.names = F)

overall_cf30 <- cbind(LOGIT30, ENET30, MARS30, C5030, 
                      CART30, RF30, TREEBAG30, ADA30, 
                      LOGITBOOST30, GBM30, XGB30)
modList <- c("Logistic", "Elastic Net", "MARS", "C50", "CART", "Random Forest", 
             "Tree Bag", "Ada Boost", "Logit Boost", "Gradient Boost", "XGBoost")
colnames(overall_cf30) <- modList
rownames(overall_cf30) <- c("Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value", "Precision", 
                            "Recall", "F1", "Prevalence", "Detection Rate", "Detection Prevalence", 
                            "Balanced Accuracy", "AUC")

save(overall_cf30, file="overall_cf30.RData")
write.table(overall_cf30, file = "overall_cf30.txt", sep = ",", quote = FALSE, row.names = F)


overall_cf.t <- xtable(overall_cf)
```{r table, echo=FALSE}
library(knitr)
kable(overall_cf)
```

## Plot 2 ROC's together
plot(logit_perf)
lines(enet_perf)
plot(enet_perf, add = TRUE)
plot(mars_perf, add=TRUE)
legend("bottomright", 
       legend = c("Logit", "GLMNET", "MARS"), 
       col = c("black", "grey35", "grey48"),
       lty = c(1, 2, 3), 
       lwd = c(1, 1, 4))

PERF <- cbind(logit_perf, enet_perf, mars_perf, c50_perf, cart_perf, 
              rf_perf, treebag_perf,  ada_perf, logitboost_perf, gbm_perf, 
              xgb_perf)
n <- 3 # you have n models
colors <- c('red', 'blue', 'green') # 2 colors
for (i in 1:n) {
  plot(logit_per[,i]),"tpr","fpr"), 
       add=(i!=1),col=colors[i],lwd=2)
}

#
library(pROC)
roc_logit = roc(nfhs_train$husb_beat, pred1)
roc2 = roc(y, pred2)


#TO see results of resampling within the training data -- need to get this command to work
res <- summary(resamples(modList))
summary(res)

files = list.files(path = 'C:/rf_models', pattern = '.rds$', full.names = TRUE)
for (i in 1:80){
  model <- readRDS(files[i])
  prediction <- predict(model, newdata = as.data.frame(test_data))
  print(prediction)
}







































## Read in the Model Data
f <- file.path("C:/Users/Gunjan/Desktop/Harvard Data Science/Capstone", 
               c("logit.rds","enet.rds"))
lapply(f, readRDS)

logit <- readRDS("./logit.rds")
enet <- readRDS("./enet.rds")

models <- list(logit=logit,enet=enet)  # create vector of model objects
# Code to read in multiple models is not working
for (m in names(models)) {
  print(m)
  m <- readRDS("./m.rds") 
}

# #Make predictions and create confusion matrix
m = length(models)
n = count(validation)[1]

class_pred <- as.data.frame(matrix(ncol = m, nrow=17451))

for (m in names(models)) {
  print(m$finalModel)
  #pred[[m]] <- predict(models[[m]], data = validation, models = m$finalModel) 
}
names(pred) %<>% paste0("_prob")
models <- list(logit=logit,enet=enet)  # create vector of model objects

# 1) for loop
for (mod in sequence_along(names(models))) {
  print(mod)
  class[[mod]] <- predict(models[[mod]], validation)
  #print(confusionMatrix(mod))
}
  
mkpred <- function(mod){
      #mod <- apply(mod, readRDS)
      mod_class <- predict(mod, validation, models = mod$finalModel)
}
      mod_prob <- predict(mod, validation, models = mod$finalModel, type = 'prob')
      mod_cf <- confusionMatrix(mod_class, validation$husb_beat, positive="yes")
      
      }

lapply(logit, mkpred)
      #ROCR package to calc and plot ROC curve and AUC
      mod_pred <- prediction(mod_prob[,2], validation$husb_beat) ##this is ROCR
      mod_pred <- performance(mod_pred, measure = 'tpr', x.measure = 'fpr') #this will be used to plot the roc
      mod_roc <- plot(perf_lda, colorize=TRUE) + abline(a=0,b=1)
      #savePlot("roc_lda.Rplot", type = "png")
      mod_auc <- performance(mod_pred, 'auc')
      mod_auc_n <- mod_auc@y.values[[1]] # AUC number
}









## Permutation Approach to Variable Importance
as.vector(logit_prob)
logit_prob <- predict(logit, validation, models = logit$finalModel, type = 'prob')
vip(
  logit,
  train = as.data.frame(nfhs_train),
  method = "permute",
  target = "husb_drink",
  metric = "auc",
  reference_class = "yes",
  nsim = 5,
  sample_frac = 0.5,
  pred_wrapper = as.vector(logit_prob)
)
