# Step 2: Set up the pre-processing steps
# We are not centering and standardizing since most variables are categorical
library(tidyverse)
library(caret)
library(recipes)
library(themis) # needed for step_smote

setwd("~/Dropbox/Capstone")
#setwd("C:/Users/Gunjan/Desktop/Harvard Data Science/Capstone")
## Dec 29: discovered data bug. 
## nfhs_train now has 58169 obs
## validation has 24928

load("nfhs_train.Rdata")
load("validation.Rdata")

table(nfhs_train$husb_beat) %>% prop.table()
table(validation$husb_beat) %>% prop.table()


# So this one sets the recipe - DO NOT prep this before feeding in to train!!!

orig_recipe <- recipe(husb_beat ~ ., data = nfhs_train) %>%
                step_nzv(all_predictors()) %>%               
                step_naomit(all_predictors()) %>%
                step_dummy(all_nominal(), -husb_beat) %>%
                step_interact(~starts_with("State"):starts_with("wave"))%>%
                step_interact(~starts_with("husb_drink"):starts_with("husb_legal"))

smote_recipe <- recipe(husb_beat ~ ., data = nfhs_train) %>%
                  step_nzv(all_predictors()) %>%               
                  step_naomit(all_predictors()) %>%
                  step_dummy(all_nominal(), -husb_beat) %>%
                  step_interact(~starts_with("State"):starts_with("wave"))%>%
                  step_interact(~starts_with("husb_drink"):starts_with("husb_legal"))%>%
                  step_smote(husb_beat)

ohe_recipe <- recipe(husb_beat ~ ., data = nfhs_train) %>%
                step_nzv(all_predictors()) %>%               
                step_naomit(all_predictors()) %>%
                step_dummy(all_nominal(), -husb_beat, one_hot = T) %>%
                step_interact(~starts_with("State"):starts_with("wave"))%>%
                step_interact(~starts_with("husb_drink"):starts_with("husb_legal"))

nodum_recipe <- recipe(husb_beat ~ ., data = nfhs_train) %>%
  step_nzv(all_predictors()) %>%               
  step_naomit(all_predictors()) %>%
  step_integer(all_nominal(), -husb_beat) %>%
  step_interact(~starts_with("State"):starts_with("wave"))%>%
  step_interact(~starts_with("husb_drink"):starts_with("husb_legal"))



# Juice/Bake creates the data set with preprocessed features 
orig_tr_data <- orig_recipe %>% prep %>% juice

smote_tr_data <- smote_recipe %>% prep %>% juice




# See original data
sort(table(orig_tr_data$husb_beat, useNA = "always"))                 
# See data after SMOTE
sort(table(smote_tr_data$husb_beat, useNA = "always"))

