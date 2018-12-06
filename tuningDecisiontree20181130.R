#set working directory
#setwd("C:/Users/jflygare/Documents")
setwd("C:/Users/sdanielsson/Documents/DataInovation")
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
library(recipes)
library(rlang)
library(caret)
library(ggplot2)
library(lubridate)
library(scales)


#load data
calldata <- read.csv("df1177_5dec.csv") %>%
  select(-"X",-"lagwDay",-"lagMonth",-"lagMonth_NA",-"lagwDay_NA",-"logLeadSamtal")

calldata$Date <- as.Date(calldata$Date)
calldata <- calldata %>%
  mutate(yeardayPredict = yday(Date + predictHorizon), weekPredict = week(Date + predictHorizon))

calldata$Date <- as.Date(calldata$Date)

#calldata2 <- calldata[,-grep("year",colnames(calldata))]
#calldata2 <- calldata2[,-grep("_month",colnames(calldata))]
#calldata <- calldata2

resultSimpMod <- data.frame()
resultImportance <- data.frame()


for (horizon in sort(unique(calldata$predictHorizon))) {
  
  currentSet <- calldata[calldata$predictHorizon == horizon, ]
  #testFinal <- testData[testData$predictHorizon == horizon, ]
  
  set.seed(1177)
  sampleSets <- sample(c(1:3),size = nrow(currentSet),replace = T, prob = c(0.7,0.2,0.1))
  #train_default <- sample_frac(currentSet,0.7,replace=F)
  #test_default <-  currentSet[-as.numeric(rownames(train_default)),]
  train_default <-  currentSet[sampleSets == 1, ]
  test_default <-  currentSet[sampleSets == 2, ]
  testFinal <-     currentSet[sampleSets == 3, ]
  ######################################################### Train Data ####################################################
  
  train_default <- train_default[complete.cases(train_default), ]
  
  train <- train_default %>%
    mutate(logLeadSamtal = log(leadSamtal)) %>%
    select(-"Date",-"samtalLog",-"samtal",-"leadSamtal",-"predictHorizon")
  
  #convert data frame to data table
  setDT(train)
  
  #using one hot encoding
  labels <- train$logLeadSamtal
  new_tr <- as.matrix(train[,-c("logLeadSamtal")])
  
  #preparing matrix
  dtrain <- xgb.DMatrix(data = new_tr,label = labels)
  
  ######################################################### Validate Data ####################################################
  
  test_default <- test_default[complete.cases(test_default), ]
  
  test<- test_default %>%
    mutate(logLeadSamtal = log(leadSamtal)) %>%
    select(-"Date",-"samtalLog",-"samtal",-"leadSamtal",-"predictHorizon")
  
  #convert data frame to data table
  setDT(test)
  
  #using one hot encoding
  ts_label <- test$logLeadSamtal
  new_ts <- as.matrix(test[,-c("logLeadSamtal")])
  
  #preparing matrix
  dtest <- xgb.DMatrix(data = new_ts,label=ts_label)
  
  ######################################################### Test Data ####################################################
  
  testFinal <- testFinal[complete.cases(testFinal), ]
  
  testSet<- testFinal %>%
    mutate(logLeadSamtal = log(leadSamtal)) %>%
    select(-"Date",-"samtalLog",-"samtal",-"leadSamtal",-"predictHorizon")
  
  #convert data frame to data table
  setDT(testSet)
  
  #using one hot encoding
  test_label <- testSet$logLeadSamtal
  new_test <- as.matrix(testSet[,-c("logLeadSamtal")])
  
  #preparing matrix
  dtestFinal <- xgb.DMatrix(data = new_test,label=test_label)
  
  
  ##################################################### Simple training with default parameters #####################################################
  
  #default parameters
  params <- list(
    booster = "gbtree",
    objective = "reg:linear",
    eta=0.1,
    gamma=0,
    max_depth=10,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=1
  )
  
  xgbcv <- xgb.cv(params = params
                  ,data = dtrain
                  ,nrounds = 500
                  ,nfold = 5
                  ,showsd = T
                  ,stratified = T
                  ,print_every_n = 10
                  ,early_stop_round = 20
                  ,maximize = F
  )

  #Select the best number of rounds from CV training
  best_rounds <- which.min(xgbcv$evaluation_log$test_rmse_mean)
  
  
  #first default - model training
  xgb1 <- xgb.train(
    params = params
    ,data = dtrain
    ,nrounds = best_rounds
    ,watchlist = list(val=dtest,train=dtrain)
    ,print_every_n = 10
    ,early_stop_round = 10
    ,maximize = F
    ,eval_metric = "rmse"
  )
  
  #model prediction
  xgbpred <- predict(xgb1,dtestFinal)
  
  #expLabel <- exp(as.numeric(ts_label))
  #expPred <- exp(as.numeric(xgbpred))
  
  #in_100 <- sum((abs(expLabel-expPred))<100)/length(expLabel)
  
  #err <- rmse(exp(as.numeric(ts_label)), exp(as.numeric(xgbpred)))
  
  testFinal$PredictRes <- xgbpred
   
  resultSimpMod <- rbind(resultSimpMod,testFinal)
  
  mat <- xgb.importance(feature_names = colnames(new_tr),model = xgb1)
  mat$horizon <- horizon
  resultImportance <- rbind(resultImportance,mat)
  
}

errorSimpMod <- resultSimpMod%>%
  group_by(predictHorizon)%>%
  summarize(error = rmse(leadSamtal, exp(as.numeric(PredictRes))),
            in_100 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<100)/n(),
            rmspe = rmse(leadSamtal, exp(as.numeric(PredictRes)))/mean(leadSamtal))

meanDay <- resultSimpMod%>%
  group_by(Date)%>%
  summarize(meanPredict = mean(exp(PredictRes)), truth = mean(leadSamtal))%>%
  ggplot(aes(Date))+geom_line(aes(y=meanPredict,colour="predict"))+geom_line(aes(y=truth,colour="trueValue"))

plot <- ggplot(subset(resultSimpMod, predictHorizon %in% c(4)), aes(x = Date+predictHorizon))+
   geom_line(aes(y = exp(PredictRes),colour="prediction",group =Date)) + 
  geom_line(aes(y = leadSamtal,colour="trueValue",group =Date))+
  xlab("Predicted day")+ylab("Number of Calls")+
  scale_x_date(date_breaks = "1 month", labels=date_format(format = "%Y-%m-%d"),limits = as.Date(c('2017-01-01','2018-01-01')))
  
plot + facet_grid(predictHorizon~.)

#view variable importance plot for specific horizon
resultImportance %>%
  filter(horizon == 7)%>%
  setDT()%>%
  xgb.plot.importance() 

resultTuneMod <- data.frame()
bestParam <- data.frame()


for (horizon in sort(unique(calldata$predictHorizon))) {
 
  currentSet <- calldata[calldata$predictHorizon == horizon, ]

  
  set.seed(1177)
  sampleSets <- sample(c(1:2),size = nrow(currentSet),replace = T, prob = c(0.7,0.3))
  train_default <-  currentSet[sampleSets == 1, ]
  test_default <-  currentSet[sampleSets == 2, ]
  ######################################################### Train Data ####################################################
  
  train_default <- train_default[complete.cases(train_default), ]
  
  train <- train_default %>%
    mutate(logLeadSamtal = log(leadSamtal)) %>%
    select(-"Date",-"samtalLog",-"samtal",-"leadSamtal",-"predictHorizon")
  
  ######################################################### Validate Data ####################################################
  
  test_default <- test_default[complete.cases(test_default), ]
  
  test<- test_default %>%
    mutate(logLeadSamtal = log(leadSamtal)) %>%
    select(-"Date",-"samtalLog",-"samtal",-"leadSamtal",-"predictHorizon")
  
  ######################################################### Tuning Data ####################################################
  
  #create tasks
  traintask <- makeRegrTask(data =train,target = "logLeadSamtal")
  testtask <- makeRegrTask(data = test,target = "logLeadSamtal")
  
  #create learner
  lrn <- makeLearner("regr.xgboost",predict.type = "response")
  lrn$par.vals <- list(
    objective="reg:linear",
    eval_metric="rmse"
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
  test_default$PredictRes <- xgpred$data$response 
  
  #error <- rmse(exp(xgpred$data$truth),exp(xgpred$data$response))
  #380 for count:poisson regression
  #375.8704 for reg:linear regression log samtal. 378.3714 when samtal
  #373.0574 for reg:linear regression with GrowingWindowCV as sampling method
  #367.5956 for reg:linear regression log samtal with CV. Tune Samples 1000 did not complete.
  #363.3445 for reg:linear regression log samtal with CV.
  
  optParam <- data.frame(mytune$x,Horizon = horizon)
  bestParam <- rbind(bestParam,optParam)
  
  
  resultTuneMod <- rbind(resultTuneMod,test_default)
  
}
  
errorTuneMod <- resultTuneMod %>%
  group_by(predictHorizon)%>%
  summarize(error = rmse(leadSamtal, exp(as.numeric(PredictRes))),
            in_100 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<100)/n())


resultDf %>%
  filter(predictHorizon >6 & predictHorizon < 15) %>%
  summarize(rmse(actual,predicted))
