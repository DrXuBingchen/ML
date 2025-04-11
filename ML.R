
library(caret)
library(DALEX)
library(ggplot2)
library(randomForest)
library(kernlab)
library(xgboost)
library(pROC)
library(fs)

set.seed(123) 

control=trainControl(method="repeatedcv", number=5, savePredictions=TRUE)
mod_rf = train(Type ~ ., data = train, method='rf', 
               trControl = control)

mod_svm=train(Type ~., data = train, method = "svmRadial",
              prob.model=TRUE, trControl=control)

mod_glm=train(Type ~., data = train, method = "glm",
              family="binomial", trControl=control)

mod_gbm=train(Type ~., data = train, method = "gbm", 
              trControl=control)
mod_knn=train(Type ~., data = train, method = "knn", 
              trControl=control)

mod_nnet=train(Type ~., data = train, method = "nnet",
               trControl=control)

mod_lasso=train(Type ~., data = train, method = "glmnet", 
                trControl=control)

mod_dt=train(Type ~., data = train, method = "rpart",
             trControl=control)

p_fun=function(object, newdata){
  predict(object, newdata=newdata, type="prob")[,2]
}
yTrain=ifelse(train$Type=="Control", 0, 1)

explainer_rf=explain(mod_rf, label = "RF",
                     data = train, y = yTrain,
                     predict_function = p_fun,
                     verbose = FALSE)
mp_rf=model_performance(explainer_rf)

explainer_svm=explain(mod_svm, label = "SVM",
                      data = train, y = yTrain,
                      predict_function = p_fun,
                      verbose = FALSE)
mp_svm=model_performance(explainer_svm)

explainer_glm=explain(mod_glm, label = "GLM",
                      data = train, y = yTrain,
                      predict_function = p_fun,
                      verbose = FALSE)
mp_glm=model_performance(explainer_glm)

explainer_gbm=explain(mod_gbm, label = "GBM",
                      data = train, y = yTrain,
                      predict_function = p_fun,
                      verbose = FALSE)
mp_gbm=model_performance(explainer_gbm)

explainer_knn=explain(mod_knn, label = "KNN",
                       data = train, y = yTrain,
                       predict_function = p_fun,
                       verbose = FALSE)
mp_knn=model_performance(explainer_knn)

explainer_nnet=explain(mod_nnet, label = "NNET",
                       data = train, y = yTrain,
                       predict_function = p_fun,
                       verbose = FALSE)
mp_nnet=model_performance(explainer_nnet)

explainer_lasso=explain(mod_lasso, label = "LASSO",
                        data = train, y = yTrain,
                        predict_function = p_fun,
                        verbose = FALSE)
mp_lasso=model_performance(explainer_lasso)

explainer_dt=explain(mod_dt, label = "DT",
                        data = train, y = yTrain,
                        predict_function = p_fun,
                        verbose = FALSE)
mp_dt=model_performance(explainer_dt)

pred1=predict(mod_rf, newdata=train, type="prob")
pred2=predict(mod_svm, newdata=train, type="prob")
pred3=predict(mod_glm, newdata=train, type="prob")
pred4=predict(mod_gbm, newdata=train, type="prob")
pred5=predict(mod_knn, newdata=train, type="prob")
pred6=predict(mod_nnet, newdata=train, type="prob")
pred7=predict(mod_lasso, newdata=train, type="prob")
pred8=predict(mod_dt, newdata=train, type="prob")

roc1=roc(yTrain, as.numeric(pred1[,2]), ci = TRUE)
roc1

roc2=roc(yTrain, as.numeric(pred2[,2]), ci = TRUE)
roc2

roc3=roc(yTrain, as.numeric(pred3[,2]), ci = TRUE)
roc3

roc4=roc(yTrain, as.numeric(pred4[,2]), ci = TRUE)
roc4

roc5=roc(yTrain, as.numeric(pred5[,2]), ci = TRUE)
roc5

roc6=roc(yTrain, as.numeric(pred6[,2]), ci = TRUE)
roc6

roc7=roc(yTrain, as.numeric(pred7[,2]), ci = TRUE)
roc7

roc8=roc(yTrain, as.numeric(pred8[,2]), ci = TRUE)
roc8
plot(roc1, print.auc=F, legacy.axes=T, main="", col="chocolate")
plot(roc2, print.auc=F, legacy.axes=T, main="", col="aquamarine3", add=T)
plot(roc3, print.auc=F, legacy.axes=T, main="", col="bisque3", add=T)
plot(roc4, print.auc=F, legacy.axes=T, main="", col="burlywood", add=T)
plot(roc5, print.auc=F, legacy.axes=T, main="", col="darkgoldenrod3", add=T)
plot(roc6, print.auc=F, legacy.axes=T, main="", col="darkolivegreen3", add=T)
plot(roc7, print.auc=F, legacy.axes=T, main="", col="dodgerblue3", add=T)
plot(roc8, print.auc=F, legacy.axes=T, main="", col="darksalmon", add=T)
legend('bottomright',
       c(paste0('RF: ',sprintf("%.03f",roc1$auc)),
         paste0('SVM: ',sprintf("%.03f",roc2$auc)),
         paste0('GLM: ',sprintf("%.03f",roc3$auc)),
         paste0('GBM: ',sprintf("%.03f",roc4$auc)),
         paste0('KNN: ',sprintf("%.03f",roc5$auc)),
         paste0('NNET: ',sprintf("%.03f",roc6$auc)),
         paste0('LASSO: ',sprintf("%.03f",roc7$auc)),
         paste0('DT: ',sprintf("%.03f",roc8$auc))),
       
       
       col=c("chocolate","aquamarine3","bisque3",
             "burlywood","darkgoldenrod3","darkolivegreen3",
             "dodgerblue3","darksalmon"), lwd=2, bty = 'n')
dev.off()


