#set working directory
setwd("C:/Users/jflygare/Documents")
#load libraries
library(data.table)
library(mlr)

#set parallel backend
library(parallel)
library(xgboost)
library(mlrMBO)
library(parallelMap)
library(smoof)
library(dplyr)
library(Metrics)

#load data
calldata <- read.csv("Rcode/1177/df1177_4dec.csv") %>%
  select(-"X",-"lagwDay",-"lagMonth",-"lagMonth_NA",-"lagwDay_NA")

calldata$Date <- as.Date(calldata$Date)
calldata <- calldata %>%
  mutate(yeardayPredict = yday(Date + predictHorizon), weekPredict = week(Date + predictHorizon))


calldata <- calldata %>%
  filter(predictHorizon == 8)
#which(colnames(calldata) == "lagwDay_söndag")
#calldata2 <- calldata[,-c(which(colnames(calldata) == "lagwDay_söndag"):which(colnames(calldata) == "lagwDay_lördag"))]
#calldata <- calldata2
#calldata2 <- calldata[,-grepl("lagWday",names(calldata))]
#calldata <- calldata[,-grep("lagWMonth",colnames(calldata))]
#calldata2 <- calldata[,-grep("year",colnames(calldata))]
#calldata2 <- calldata2[,-grep("_month",colnames(calldata))]
#calldata <- calldata2

set.seed(1178)
train_default <- sample_frac(calldata,0.7,replace=F)
test_default <-  calldata[-as.numeric(rownames(train_default)),]

train_default <- calldata %>%
  filter(Date < '2018-03-01' & Date > '2017-01-01')

test_default <- calldata %>%
  filter(Date >= '2018-03-01')
######################################################### Train Data ####################################################
#check missing values
table(is.na(train_default))
sapply(train_default, function(x) sum(is.na(x))/length(x))*100

train_default <- train_default[complete.cases(train_default), ]

train <- train_default %>%
  select(-"Date",-"samtalLog",-"samtal")


setDT(train)

#using one hot encoding
labels <- train$leadSamtal
new_tr <- as.matrix(train[,-c("leadSamtal")])

#preparing matrix
dtrain <- xgb.DMatrix(data = new_tr,label = labels)



######################################################### Validate Data ####################################################
table(is.na(test_default))
sapply(test_default, function(x) sum(is.na(x))/length(x))*100

test_default <- test_default[complete.cases(test_default), ]

test<- test_default %>%
  select(-"Date",-"samtalLog",-"samtal")
#convert data frame to data table
setDT(test)

#using one hot encoding
ts_label <- test$leadSamtal
new_ts <- as.matrix(test[,-c("leadSamtal")])

#preparing matrix
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)



#default parameters
params <- list(
  booster = "gbtree",
  objective = "reg:linear",
  eta=0.3,
  gamma=0.02,
  #lambda = 0.8,
  max_depth=9,
  min_child_weight=1,
  subsample=0.6,
  colsample_bytree=0.9
)

xgbcv <- xgb.cv(params = params
                ,data = dtrain
                ,nrounds = 200
                ,nfold = 8
                ,showsd = T
                ,stratified = T
                ,print_every_n = 10
                ,early_stop_round = 20
                ,maximize = F
)
##best iteration = 100 
#nrounds = number of trees

min(xgbcv$evaluation_log$test_rmse_mean)
#0.1263

#first default - model training
xgb1 <- xgb.train(
  params = params
  ,data = dtrain
  ,nrounds = 200
  ,watchlist = list(train=dtrain,val=dtest)
  ,print_every_n = 10
  ,early_stop_round = 20
  ,maximize = F
  ,eval_metric = "rmse"
)

#model prediction
xgbpred <- predict(xgb1,dtest)

expLabel <- exp(as.numeric(ts_label))
expPred <- exp(as.numeric(xgbpred))

sum((abs(expLabel-expPred))<100)/length(expLabel)

err <- rmse(exp(as.numeric(ts_label)), exp(as.numeric(xgbpred)))

#confusion matrix
library(recipes)
library(rlang)
library(caret)

#view variable importance plot
mat <- xgb.importance(feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance(importance_matrix = mat[1:20]) #first 20 variables
#sum(mat$Gain)

 
resultDf <- cbind("actual" = expLabel,"predicted" = expPred,test_default)
perHorizon <- resultDf %>%
  group_by(predictHorizon) %>%
  summarize(rmse(actual,predicted))

resultDf %>%
  filter(predictHorizon >6 & predictHorizon < 15) %>%
  summarize(rmse(actual,predicted))


#create tasks
traintask <- makeRegrTask(data =as.data.frame(train),target = "logLeadSamtal")
testtask <- makeRegrTask(data = as.data.frame(test),target = "logLeadSamtal")


#create learner
lrn <- makeLearner("regr.xgboost",predict.type = "response")
lrn$par.vals <- list(
  objective="reg:linear",
  eval_metric="rmse",
  eta=0.3
)

#set parameter space
params <- makeParamSet(
  makeDiscreteParam("booster",values = "gbtree"),
  makeIntegerParam("max_depth",lower = 5L,upper = 10L),
  makeIntegerParam("nrounds",lower = 100L, upper = 500L),
  #makeNumericParam("min_child_weight",lower = 1L,upper = 3L),
  makeNumericParam("subsample",lower = 0.5,upper = 1),
  #makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
  makeNumericParam("eta", lower = .1, upper = .5),
  makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x)
)

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = F ,iters=10L)
#GrowingWindowCV som resample metod tar as lång tid. 
#search strategy
ctrl <- makeTuneControlRandom(maxit = 50L)

#parallelLibrary('mlr')
#parallelStartSocket(cpus = 3)

#parameter tuning
mytune <- tuneParams(learner = lrn
                     ,task = traintask
                     ,resampling = rdesc
                     #,measures = acc
                     ,par.set = params
                     ,control = ctrl
                     ,show.info = T)

mytune$y #0.873069

#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- mlr::train(learner = lrn_tune,task = traintask)

#predict model
xgpred <- predict(xgmodel,testtask)

error <- rmse(exp(xgpred$data$truth),exp(xgpred$data$response))
#380 for count:poisson regression
#375.8704 for reg:linear regression log samtal. 378.3714 when samtal
#373.0574 for reg:linear regression with GrowingWindowCV as sampling method
#367.5956 for reg:linear regression log samtal with CV. Tune Samples 1000 did not complete.
#363.3445 for reg:linear regression log samtal with CV.

plot(xgpred$data$id,exp(xgpred$data$truth),type = 'l',col = 'blue')
lines(xgpred$data$id,exp(xgpred$data$response),col="black")


confusionMatrix(xgpred$data$response,xgpred$data$truth)
#Accuracy : 0.8747

#stop parallelization
parallelStop()


resultDf <- cbind("actual" = exp(xgpred$data$truth),"predicted" = exp(xgpred$data$response),test_default)
perHorizon <- resultDf %>%
  group_by(predictHorizon) %>%
  summarize(rmse(actual,predicted))

resultDf %>%
  filter(predictHorizon >6 & predictHorizon < 15) %>%
  summarize(rmse(actual,predicted))

## Ta ut snitt-estimate per dag
#Endast horisont 7-14
dailyPredict <- resultDf %>%
  filter(predictHorizon >6 & predictHorizon < 15) %>%
  group_by(Date) %>%
  summarize("actual2" = mean(actual),"predict2" = mean(predicted),"diff" = mean(actual) - mean(predicted))

#Inom +/-1 100
sum(dailyPredict$diff<100)/nrow(dailyPredict)

#+/- 50
sum(dailyPredict$diff<50)/nrow(dailyPredict)

#Weighted predictions (rätt kass tbh)
step1 <- perHorizon %>%
  filter(predictHorizon > 6 & predictHorizon < 15) %>%
  mutate(invError = 1/`rmse(actual, predicted)`,total = sum(1/`rmse(actual, predicted)`),
         weight = invError / total)

wDailyPredict <- merge(step1,resultDf,by="predictHorizon") %>%
  group_by(Date) %>%
  summarize("actual2" = mean(actual),"wPredict" = sum(((1/`rmse(actual, predicted)`)/
                                                     sum(1/`rmse(actual, predicted)`))*predicted),"diff" = mean(actual)-sum(((1/`rmse(actual, predicted)`)/
                                                                                                                               sum(1/`rmse(actual, predicted)`))*predicted))
#Inom +/-1 100
sum(wDailyPredict$diff<100)/nrow(wDailyPredict)

#+/- 50
sum(wDailyPredict$diff<50)/nrow(wDailyPredict)
