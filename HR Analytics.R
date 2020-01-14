#Loading training data
train <- read.csv(file.choose(), header=TRUE,sep=",",na.strings=c("","NA"))

#We first omit id of employees and NA values
train <- train[,2:14]
train <-na.omit(train)

#We fit a random forest on the remaining data and use SMOTE since 
#is_promoted=1 is a minority class
set.seed(123)
library(DMwR)
library(randomForest)
train$is_promoted=as.factor(train$is_promoted)
smoted_train <- SMOTE(is_promoted~., train, perc.over=500)
fit1 <- randomForest(is_promoted~., data = smoted_train,mtry=3,ntree=500)
print(fit1)
varImpPlot(fit1)

#from the varImpPlot we see that education has a low Mean Decrease Gini contrary
#to previous_year_rating. We decide to forget about education

#FINE TUNING
#nombre d'arbres selon OOB error
plot(fit1$err.rate[, 1], type = "l", xlab = "nombre d'arbres", ylab = "erreur OOB")
#on remarque que l'erreur se stabilise à partir de ntree=200

#We train a second time our random forest 
train <-read.csv(file.choose(), header=TRUE,sep=",",na.strings=c("","NA"))

#We remove the first column containing employees' id and the fourth column 
#containing education
train[,c(1,4)] <- NULL
train[is.na(train)] <- 3
train$is_promoted=as.factor(train$is_promoted)
smoted_train <- SMOTE(is_promoted~., train, perc.over=500)
fit2 <- randomForest(is_promoted~., data = smoted_train,mtry=3,ntree=500)
varImpPlot(fit2)





#Loading test data
test <- read.csv(file.choose(), header=TRUE,sep=",",na.strings=c("","NA"))
test[,c(1,4)] <- NULL

#impute NA values of orevious_year_rating with 3 (mean of previous_year rating)
test[is.na(test)] <- 3

#Apply random forest model to predict who will be promoted
test$is_promoted <- predict(fit2, test)

#Output saved in a csv file 
write.csv(test$is_promoted, file = "MySolution.csv")


