###
# R Studio Server Script
# Run this on the server at the begining of each session
library("RStudioAMI")  ## this is needed to link dropbox
linkDropbox() ## follow the link to complete this procedure
excludeSyncDropbox("*")
includeSyncDropbox("Capstone")
#options(install.packages.compile.from.source = "always")

install.packages("tidyverse", "caret", "vip", "recipes", "DMwR", "doParallel", "pdp", "themis", "microbenchmark", "ROCR")
install.packages(c("caret"), repos = "http://cran.r-project.org")
## Need to install packages for the algorithms wez
library(tidyverse)
library(caret)
library(vip) 
library(recipes)
library(pdp)
library(themis)
library(microbenchmark)
library(ROCR)


setwd("~/Dropbox/Capstone")


#install.packages(c("feather","tidyr"), type = "both")