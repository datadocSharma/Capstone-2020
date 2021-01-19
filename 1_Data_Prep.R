library(tidyverse)
library(caret)
library(vip) 
library(recipe)
library(pdp)
library(themis)

#setwd("home/Dropbox/Capstone")

setwd("C:/Users/Gunjan/Desktop/Harvard Data Science/Capstone")

## Import NFHS Data
nfhs_orig <- read.csv("C:/Users/Gunjan/Documents/Research/Alcohol regulation India/data/nfhs.csv")
# Step 0: Clean data - this is NOT pre-processing. We are simply tidying the data before Splitting into Train and Test

# Relevel main target variable to remove an unused third level
is.na(nfhs_orig$husb_beat) <- nfhs_orig$husb_beat == ""
nfhs_orig$husb_beat <- factor(nfhs_orig$husb_beat)
nfhs_use <- nfhs_orig %>% filter(!is.na(husb_beat))
# There is a big outlier in rep_educ
nfhs_use$rep_educ[nfhs_use$rep_educ>25] <-25

## Filter the near-zero variance features
# shows that minority religions (christian, sikh, other) & rep_drink (women drinks)
# have near zero variance. So can drop dummies for these. 
caret::nearZeroVar(nfhs_use, saveMetrics = TRUE) %>% 
  tibble::rownames_to_column() %>% 
  filter(nzv)
nfhs_use <- select(nfhs_use, -c('christian', 'sikh', 'other_relig', 'over', 'over18', 'over21', 'over25', 'used', 'rep_drink'))

## Check for multicollinearity - there is something wrong with the toilet variable
# As expected husb_age and rep_age are highly correlated, as are husb_educ & rep_educ
# can shrink feature set by using husb_educ & husb_age (as proxy for income), then age gap and educ gap 
nfhs_use <- nfhs_use %>% mutate(ipv = as.numeric(husb_beat))
nfhs_numeric_features <- nfhs_use %>% 
  select(ipv, rep_age, rep_educ, hhsize, children, urban, toilet,  electricity,  religion, husb_age, husb_educ, husb_drink, ownmoney, b_unfaithful, MLDA1, husb_legal1, prohib, agegap, educgap)
nfhs_numeric_features <- na.omit(nfhs_numeric_features)
source("http://www.sthda.com/upload/rquery_cormat.r") ## brings in code to make easy graphs
rquery.cormat(nfhs_numeric_features)
# See full matrix
col<- colorRampPalette(c("blue", "white", "red"))(20)
cormat<-rquery.cormat(nfhs_numeric_features, type="full", col=col)
#heat map
df <- nfhs_orig %>% select(ipv, state_id, rep_age, rep_educ, hhsize, children, urban, toilet,  electricity,  religion, husb_age, husb_educ, husb_drink, ownmoney, b_unfaithful, MLDA1, husb_legal1, prohib)
#nfhs_numeric_features <- na.omit(nfhs_numeric_features)
cormat<-rquery.cormat(df, graphType="heatmap")

cormat <- round(cor(nfhs_numeric_features),2)
head(cormat)

nfhs_use <- select(nfhs_use, -c('toilet', 'rep_age', 'rep_educ'))


#########################################################
## Create the Data Set to Use
# nfhs_use.Rdata
# Contains sample of 83,391 respondents with husbands between the age of 15-50
# Spread over 19 states & 2 waves (47327 in wave 1 and 36064 in wave 2)
# mytab <-table(nfhs_use$wave)
##########################################################
nfhs_use <- nfhs_use %>% 
  select(husb_beat, hhsize, children, urban, electricity,  hindu, muslim, husb_age, husb_educ, husb_drink, ownmoney, b_unfaithful, MLDA1, husb_legal1, prohib, agegap, educgap, State, wave)
# Remove rows with any missing values
nfhs_use <- na.omit(nfhs_use)
# Convert categorical vars into factors
cols <- c("urban", 'electricity', 'hindu', 'muslim', 'husb_drink', 'ownmoney', 'b_unfaithful', 'MLDA1', 'husb_legal1', 'prohib', 'State', 'wave')
nfhs_use[cols] <- lapply(nfhs_use[cols], factor)  ## quick view str(nfhs_use)
save(nfhs_use, file="nfhs_use.Rdata")
##############################################################

########################################################### 
# Step 1: SPlit Data into Test and Train
# We will use Stratified Sampling
# nfhs_train.Rdata
# Contains sample of 58, 375 respondents with husbands between the age of 15-50
# Spread over 19 states & 2 waves (33269 in wave 1 and 25106 in wave 2)
# mytab <-table(nfhs_use$wave)
# Target var ("husb_beat") is 1 for 9992 respondents, 0 for 48383 --> 17.1 % rate of dom. violence
# Validation.Rdata
# Contains sample of 25, 016 respondents with husbands between the age of 15-50
# Spread over 19 states & 2 waves (14058 in wave 1 and 10958 in wave 2)
# mytab <-table(validation$wave)
# Target var ("husb_beat") is 1 for 4281 respondents, 0 for 20,753 --> 17.1 % rate of dom. violence
############################################################

load("nfhs_use.Rdata")
library(rsample)
set.seed(123, sample.kind = 'Rounding')
split_strat  <- initial_split(nfhs_use, prop = 0.7, strata = "husb_beat")
nfhs_train  <- training(split_strat)
validation   <- testing(split_strat)
table(nfhs_train$husb_beat) %>% prop.table()
table(validation$husb_beat) %>% prop.table()

save(nfhs_train, file="nfhs_train.Rdata")
save(validation, file="validation.Rdata")