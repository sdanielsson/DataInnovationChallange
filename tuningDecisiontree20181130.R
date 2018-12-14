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

######################################################### Import Data ####################################################

#load data
calldata <- read.csv("df1177_13dec.csv",stringsAsFactors = F) %>%
  select(-"X",-"predictDay")
  #select(-"X",-"lagwDay",-"lagMonth",-"lagMonth_NA",-"lagwDay_NA")

calldata$Date <- as.Date(calldata$Date)
calldata <- calldata %>%
  mutate(yeardayPredict = yday(Date + predictHorizon))
  #mutate(yeardayPredict = yday(Date + predictHorizon), weekPredict = week(Date + predictHorizon))

calldata2 <- calldata[,-grep("_twoWeek",colnames(calldata))]
calldata2 <- calldata2[,-grep("_threeWeek",colnames(calldata2))]
#calldata2 <- calldata2[,-grep("lagwDay_",colnames(calldata2))]
#calldata2 <- calldata2[,-grep("predictwDay_",colnames(calldata2))]
calldata <- calldata2

######################################################### Make Final Pred Data ####################################################

fromDate <- '2017-05-01'
toDate <- '2017-09-01'
  
#Pick out one month for final prediction
calldataFinalPredAll <- calldata%>%
  filter((Date>= fromDate& Date<toDate))

#Remove the month used for final prediction
calldata <- calldata%>%
  filter(!(Date>= fromDate& Date<toDate))  

######################################################### Simple Training Loop ####################################################


resultSimpMod <- data.frame()
resultImportance <- data.frame()
resultPredict <- data.frame()


for (horizon in sort(unique(calldata$predictHorizon))) {
  
  currentSet <- calldata[calldata$predictHorizon == horizon, ]
  
  calldataFinalPred <- calldataFinalPredAll[calldataFinalPredAll$predictHorizon == horizon,]
  #testFinal <- testData[testData$predictHorizon == horizon, ]
  
  set.seed(1)
  sampleSets <- sample(c(1:3),size = nrow(currentSet),replace = T, prob = c(0.7,0.2,0.1))
  #sampleSets <- sample(c(1:2),size = nrow(currentSet),replace = T, prob = c(0.7,0.3))
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
 
   ######################################################### Final Pred Data ####################################################
  
  calldataFinalPred <- calldataFinalPred[complete.cases(calldataFinalPred), ]
  
  precentPred <- calldataFinalPred %>%
    mutate(logLeadSamtal = log(leadSamtal)) %>%
    select(-"Date",-"samtalLog",-"samtal",-"leadSamtal",-"predictHorizon")
  
  #convert data frame to data table
  setDT(precentPred)
  
  #using one hot encoding
  labels_pred <- precentPred$logLeadSamtal
  new_tr_pred <- as.matrix(precentPred[,-c("logLeadSamtal")])
  
  #preparing matrix
  dtestPred <- xgb.DMatrix(data = new_tr_pred,label = labels_pred)
  
  
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
  #best_rounds <- which.min(xgbcv$evaluation_log$test_poisson_nloglik_mean)
  
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
  predFinal <- predict(xgb1,dtestPred)
  #xgbpred <- predict(xgb1,dtest)
  
  #expLabel <- exp(as.numeric(ts_label))
  #expPred <- exp(as.numeric(xgbpred))
  
  #in_100 <- sum((abs(expLabel-expPred))<100)/length(expLabel)
  
  #err <- rmse(exp(as.numeric(ts_label)), exp(as.numeric(xgbpred)))
  
  testFinal$PredictRes <- xgbpred
  calldataFinalPred$PredictRes <- predFinal 
  #test_default$PredictRes <- xgbpred
  resultPredict <- rbind(resultPredict,calldataFinalPred)
  resultSimpMod <- rbind(resultSimpMod,testFinal)
  
  mat <- xgb.importance(feature_names = colnames(new_tr),model = xgb1)
  mat$horizon <- horizon
  resultImportance <- rbind(resultImportance,mat)
  
}

##################################################### Error Statistics #####################################################
#Error simple model for each predict horizon
errorSimpMod <- resultSimpMod%>%
  group_by(predictHorizon)%>%
  summarize(error = rmse(leadSamtal, exp(as.numeric(PredictRes))),
            in_200 = sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<200)/n(), 
            in_100 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<100)/n(), 
            in_50 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<50)/n(),
            rmspe = rmse(leadSamtal, exp(as.numeric(PredictRes)))/mean(leadSamtal))

#Error simple model for each yearday
errorSimpModTime <- resultSimpMod%>%
  group_by(yeardayPredict)%>%
  summarize(error = rmse(leadSamtal, exp(as.numeric(PredictRes))),
            in_200 = sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<200)/n(), 
            in_100 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<100)/n(), 
            in_50 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<50)/n(),
            rmspe = rmse(leadSamtal, exp(as.numeric(PredictRes)))/mean(leadSamtal),
            nObs = n())

#Error simple model for removed predicted period
errorPred <- resultPredict%>%
  group_by(predictHorizon)%>%
  summarize(error = rmse(leadSamtal, exp(as.numeric(PredictRes))),
            in_200 = sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<200)/n(), 
            in_100 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<100)/n(), 
            in_50 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<50)/n(),
            rmspe = rmse(leadSamtal, exp(as.numeric(PredictRes)))/mean(leadSamtal))

##################################################### Result Visualisations #####################################################

#Plot time axis of whole data set 
plot <- ggplot(subset(calldata, predictHorizon %in% c(1)), aes(x = Date+predictHorizon))+
  #geom_line(aes(y = exp(PredictRes),colour="prediction")) + 
  geom_line(aes(y = leadSamtal,colour="trueValue"))+
  xlab("Predicted day")+ylab("Number of Calls")

#Plot prediction for sepcific horizon
plot <- ggplot(subset(resultPredict, predictHorizon %in% c(2,8,9,10)), aes(x = Date+predictHorizon))+
   geom_line(aes(y = exp(PredictRes),colour="prediction")) + 
  geom_line(aes(y = leadSamtal,colour="trueValue"))+
  xlab("Predicted day")+ylab("Number of Calls")#+
  #scale_x_date(date_breaks = "1 month", labels=date_format(format = "%Y-%m-%d"),limits = as.Date(c('2017-01-01','2018-01-01')))
  
plot + facet_grid(predictHorizon~.)

#Plot prediction for sepcific horizon
meanPred <- mean(exp(resultPredict$PredictRes))
meanTrue <- mean(resultPredict$leadSamtal)
sdPred <- sd(exp(resultPredict$PredictRes))
sdTrue <- sd(resultPredict$leadSamtal)
  
resultPredict%>%
  mutate(predicSamtal = exp(PredictRes))%>%
  select(leadSamtal,predicSamtal)%>%
  gather(key = "type",value = "samtal")%>%
  ggplot(aes(samtal,fill=type))+
  geom_histogram(alpha = 0.5, aes(y = ..density..),position = "identity")+
  stat_function(fun = dnorm, args = list(mean =meanPred, sd = sdPred)) +
  stat_function(fun = dnorm, args = list(mean =meanTrue, sd = sdTrue))


write.csv(resultPredict, file = "resultSimpMod181214_data13dec.csv")



#view variable importance plot for specific horizon
resultImportance %>%
  filter(horizon == 7)%>%
  setDT()%>%
  xgb.plot.importance()



resultImportanceold%>%
  group_by(horizon)%>%
  arrange(desc(Gain))%>%
  dplyr::top_n(10,wt=Gain)%>%
  filter(horizon %in% c(1,7,14))%>%
  ggplot(aes(y=Gain, x=reorder(Feature,-Gain), color=factor(horizon), fill=factor(horizon))) +
  geom_bar(stat="identity") +   
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  facet_wrap(~horizon)

resultTuneMod <- data.frame()
resultTunePred <- data.frame()
bestParam <- data.frame()


for (horizon in sort(unique(calldata$predictHorizon))) {
 
  currentSet <- calldata[calldata$predictHorizon == horizon, ]
  calldataFinalPredTune <- calldataFinalPredAll[calldataFinalPredAll$predictHorizon == horizon,]
  
  set.seed(1)
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
  
  ######################################################### Final Pred Data ####################################################
  
  calldataFinalPredTune <- calldataFinalPredTune[complete.cases(calldataFinalPredTune), ]
  
  precentPred <- calldataFinalPredTune %>%
    mutate(logLeadSamtal = log(leadSamtal)) %>%
    select(-"Date",-"samtalLog",-"samtal",-"leadSamtal",-"predictHorizon")
  
  
  ######################################################### Tuning Data ####################################################
  
  #create tasks
  traintask <- makeRegrTask(data =train,target = "logLeadSamtal")
  testtask <- makeRegrTask(data = test,target = "logLeadSamtal")
  predTast <- makeRegrTask(data = precentPred,target = "logLeadSamtal")
  
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
  ctrl <- makeTuneControlRandom(maxit = 10L)
  
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
  predFinal <- predict(xgmodel,predTast)
  
  test_default$PredictRes <- xgpred$data$response 
  calldataFinalPredTune$PredictRes <- predFinal$data$response
  #error <- rmse(exp(xgpred$data$truth),exp(xgpred$data$response))
  #380 for count:poisson regression
  #375.8704 for reg:linear regression log samtal. 378.3714 when samtal
  #373.0574 for reg:linear regression with GrowingWindowCV as sampling method
  #367.5956 for reg:linear regression log samtal with CV. Tune Samples 1000 did not complete.
  #363.3445 for reg:linear regression log samtal with CV.
  
  optParam <- data.frame(mytune$x,Horizon = horizon)
  bestParam <- rbind(bestParam,optParam)
  
  
  resultTuneMod <- rbind(resultTuneMod,test_default)
  resultTunePred <- rbind(resultTunePred,calldataFinalPredTune)
  
}
  
errorTuneMod <- resultTuneMod %>%
  group_by(predictHorizon)%>%
  summarize(error = rmse(leadSamtal, exp(as.numeric(PredictRes))),
            in_100 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<100)/n())

errorTuneModPred <- resultTunePred %>%
  group_by(predictHorizon)%>%
  summarize(error = rmse(leadSamtal, exp(as.numeric(PredictRes))),
            in_100 =  sum((abs(leadSamtal-exp(as.numeric(PredictRes))))<100)/n())

 


#Plot prediction for sepcific horizon
plot <- ggplot(subset(resultTunePred, predictHorizon %in% c(4,7,9,13)), aes(x = Date+predictHorizon))+
  geom_line(aes(y = exp(PredictRes),colour="prediction")) + 
  geom_line(aes(y = leadSamtal,colour="trueValue"))+
  xlab("Predicted day")+ylab("Number of Calls")#+
#scale_x_date(date_breaks = "1 month", labels=date_format(format = "%Y-%m-%d"),limits = as.Date(c('2017-01-01','2018-01-01')))

plot + facet_grid(predictHorizon~.)


summaryDate <- resultTunePred%>%
  group_by(Date)%>%
  summarise(meanPred = mean(exp(PredictRes)),trueVal = mean(samtal),minPred = min(exp(PredictRes)),maxPred = max(exp(PredictRes)),sdPred = sd(exp(PredictRes)))

resultTunePred%>%
  group_by(Date)%>%
  summarise(meanPred = mean(exp(PredictRes)),trueVal = mean(samtal))%>%
  ggplot(aes(x = Date))+
  geom_line(aes(y = meanPred,colour="prediction")) + 
  geom_line(aes(y = trueVal,colour="trueValue"))+
  xlab("Predicted day")+ylab("Number of Calls")#+



