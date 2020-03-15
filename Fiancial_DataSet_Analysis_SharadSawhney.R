#
#Packages and libariries Required for Execution
#

install.packages("class")
install.packages("gmodels")
install.packages("e1071")
install.packages("C50")
install.packages("RWeka")
install.packages("dplyr")
install.packages("caret")
install.packages("ggplot2")
install.packages("caTools")
install.packages("randomForest")
install.packages("rpart.plot")
install.packages("xgboost")
install.packages("data.table")
install.packages("Metrics")
install.packages("gridExtra")
install.packages("InformationValue")
library(RWeka)
library(C50)
library(e1071)
library(gmodels)
library(dplyr)
library(caret)
library(ggplot2)
library(caTools)
library(randomForest)
library(rpart)
library(rpart.plot)
library(Matrix)
library(xgboost)
library(data.table)
library(magrittr)
library(Metrics)
library(gridExtra)
library(class)
library(InformationValue)

#
# ************Step 1 : Importing the Data*******
#

Bank_Tranaction1 <- read.csv(file.choose(), stringsAsFactors = FALSE) # use the bankTransact data set 
head(Bank_Tranaction1)

#
#************Step 2 : Data Preparation & Exploratory Data Analysis********
#

#
# 2.0 Checking Missing values
#
sum(is.na(Bank_Tranaction1))
# 2.1 : Data Preparation 
input_ones <- Bank_Tranaction1[which(Bank_Tranaction1$isFraud == 1), ]  # all 1's
input_zeros <- Bank_Tranaction1[which(Bank_Tranaction1$isFraud == 0), ]  # all 0's


#
# 2.2. Handling Class Bias
#

set.seed(100)  # for repeatability of samples
input_ones_training_rows <- sample(1:nrow(input_ones), 0.8*nrow(input_ones))  # 1's for training
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.8*nrow(input_ones))  # 0's for training. Pick as many 0's as 1's
training_ones <- input_ones[input_ones_training_rows, ]  
training_zeros <- input_zeros[input_zeros_training_rows, ]
Bank_Tranaction <- rbind(training_ones, training_zeros) 
str(Bank_Tranaction)

#
# 2.3 Exploratory Data Analysis
#

fraud_count<- Bank_Tranaction %>% count(isFraud)
print(fraud_count)
#
# graph between Fraud vs Non Fraud
#
barplot(prop.table(fraud_count$n)*100, names.arg = c('not fraud' ,  'fraud'), ylab = 'No of Transactions' ,main = "Fraud vs Not Fraud" ,col = 'light pink' , ylim = c(0,100))
#
#Transactions as per Type 
#
ggplot(data = Bank_Tranaction, aes(x = type , fill = type)) + geom_bar() + labs(title = "Transactions as per Type",  x = 'Transaction Type' , y = 'No of transactions' ) +theme_classic()
#
#Fraud transactions as Per type
#
Fraud_trans_type <- Bank_Tranaction %>% group_by(type) %>% summarise(fraud_transactions = sum(isFraud))
ggplot(data = Fraud_trans_type, aes(x = type,  y = fraud_transactions)) + geom_col(aes(fill = 'type'), show.legend = FALSE) + labs(title = 'Fraud transactions as Per type', x = 'Transcation type', y = 'No of Fraud Transactions') + geom_text(aes(label = fraud_transactions), size = 4, hjust = 0.5) + theme_classic()
#
#Fraud transaction Amount distribution
#
ggplot(data = Bank_Tranaction[Bank_Tranaction$isFraud==1,], aes(x = amount ,  fill =amount)) + geom_histogram(bins = 30, aes(fill = 'amount')) + labs(title = 'Fraud transaction Amount distribution', y = 'No. of Fraud transacts', x = 'Amount in Dollars')
#
#Total transactions at different Hours
#
Bank_Tranaction$hour <- mod(Bank_Tranaction$step, 24)
p5<- ggplot(data = Bank_Tranaction, aes(x = hour)) + geom_bar(aes(fill = 'isFraud'), show.legend = FALSE) +labs(title= 'Total transactions at different Hours', y = 'No. of transactions') + theme_classic()
p6<-ggplot(data = Bank_Tranaction[Bank_Tranaction$isFraud==1,], aes(x = hour)) + geom_bar(aes(fill = 'isFraud'), show.legend = FALSE) +labs(title= 'Fraud transactions at different Hours', y = 'No. of fraud transactions') + theme_classic()
grid.arrange(p5, p6, ncol = 1, nrow = 2)

#
# 2.4 Creating new features
#

Bank_Tranaction$adjustedBalanceOrg<-round(Bank_Tranaction$newbalanceOrig+Bank_Tranaction$amount-Bank_Tranaction$oldbalanceOrg, 2)
Bank_Tranaction$adjustedBalanceDest<-round(Bank_Tranaction$oldbalanceDest+Bank_Tranaction$amount-Bank_Tranaction$newbalanceDest, 2)
colnames(Bank_Tranaction)

#
# 2.5 Filtering data set based on type
#

transactions1<- Bank_Tranaction %>% 
  select( -one_of('step','nameOrig', 'nameDest', 'isFlaggedFraud')) %>%
  filter(type %in% c('CASH_OUT','TRANSFER'))
colnames(transactions1)
#
# 2.6 Creating train and test data sets 
#
library(fastDummies)
transactions1 <- dummy_cols(transactions1)
#ransactions1$isFraud <- as.factor(transactions1$isFraud)
transactionsDT <- transactions1[,-1]
transactions1 <- transactions1[,-1]
dim(transactions1)
colnames(transactions1)

#
# 2.7 Scalling the data
#

dfNormZ1 <-data.frame(scale(transactions1[c(1:5,7:9)]),transactions1[(c(6,10:11))])
summary(dfNormZ1)

#
#2.8 Creating test and Train data sets 
#

str(dfNormZ1)
dfNormZ1$isFraud <- factor(dfNormZ1$isFraud, levels=c(0,1))
set.seed(102)
ran <- sample(1:nrow(dfNormZ1), 0.7 * nrow(dfNormZ1)) 
transactions_train <- dfNormZ1[ran,] 
transactions_test <- dfNormZ1[-ran,] 
transactions_train_lables <- dfNormZ1[ran,9]
transactions_test_lables <- dfNormZ1[-ran,9]
str(transactions_test_lables)
sum(is.na(transactions_test_lables))
transactions_test_lables
#
# ****** Step 3. Model Buliding and Evalauating results********
#

#
# 3.1 KNN Algorithem : Classfication
#

# for k = 1
# Training the model
bank_test_pred <- knn(train = transactions_train, test = transactions_test,cl = transactions_train_lables, k=1)
# Evaluating the reslts 
(CrossTable(x = transactions_test_lables, y = bank_test_pred, prop.chisq=FALSE,dnn = c('predicted', 'actual')))

# for k = 20
# Training the model
bank_test_pred <- knn(train = transactions_train, test = transactions_test,
                      cl = transactions_train_lables, k=20)
# Evaluating the reslts 
(CrossTable(x = transactions_test_lables, y = bank_test_pred, prop.chisq=FALSE,dnn = c('predicted', 'actual')))

# for k = 50
# Training the model
bank_test_pred <- knn(train = transactions_train, test = transactions_test,
                      cl = transactions_train_lables, k=50)
# Evaluating the reslts 
(CrossTable(x = transactions_test_lables, y = bank_test_pred, prop.chisq=FALSE,dnn = c('predicted', 'actual')))

# for k = 81   we selected bec0z its suare root of number of rows in data set
# Training the model
bank_test_pred <- knn(train = transactions_train, test = transactions_test,
                      cl = transactions_train_lables, k=81)
# Evaluating the reslts 
(CrossTable(x = bank_test_pred,y = transactions_test_lables, prop.chisq=FALSE,dnn = c('predicted','actual')))

# for k = 86
# Training the model
bank_test_pred <- knn(train = transactions_train, test = transactions_test,
                      cl = transactions_train_lables, k=86)
# Evaluating the reslts 
(CrossTable(x = transactions_test_lables, y = bank_test_pred, prop.chisq=FALSE,dnn = c('predicted', 'actual')))

#
# 3.2. Navies Bayes Algorithm : Classfication
#

#detach("package:dplyr", character.only = TRUE)
#library("dplyr", character.only = TRUE)
#
# Discretizing  numeric features
#
transactions2<- transactions1 %>% 
  select('amount','oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest','newbalanceDest','adjustedBalanceOrg','adjustedBalanceDest') 
summary(transactions2)
binning <- function(x) {
  return (cut(x,25))
}
transactions2 <- as.data.frame(lapply(transactions2, binning))
dfNormZ1 <-data.frame(transactions2,transactions1[(c('amount','type_TRANSFER','type_CASH_OUT','isFraud'))])
summary(dfNormZ1)

#
# Creating Test and Train Data sets 
#

set.seed(104)
dfNormZ1$isFraud <- factor(dfNormZ1$isFraud, levels=c(1,0))
ran <- sample(1:nrow(dfNormZ1), 0.7 * nrow(dfNormZ1)) 
transactions_trainBT <- dfNormZ1[ran,] 
transactions_testBT <- dfNormZ1[-ran,] 
transactions_train_lablesBT <- dfNormZ1[ran,9]
transactions_test_lablesBT <- dfNormZ1[-ran,9]

#
# Traianing the Model
#
NBclassfier<-naiveBayes(isFraud ~ ., data=transactions_trainBT,laplace = 1)
NBclassfier

#
# Evaluating the Model
#

testPred=predict(NBclassfier, newdata=transactions_testBT, type="class")
testTable=table(transactions_testBT$isFraud, testPred)
(CrossTable(y = transactions_testBT$isFraud, x = testPred, prop.chisq=FALSE,dnn = c('predicted','actual')))

#
# Printing the Perfomance metric of the model
#

printALL=function(model){
  trainPred=predict(model, newdata = transactions_trainBT, type = "class")
  trainTable=table(transactions_trainBT$isFraud, trainPred)
  testPred=predict(NBclassfier, newdata=transactions_testBT, type="class")
  testTable=table(transactions_testBT$isFraud, testPred)
  trainAcc=(trainTable[1,1]+trainTable[2,2])/sum(trainTable)
  testAcc=(testTable[1,1]+testTable[2,2])/sum(testTable)
  message("Contingency Table for Training Data")
  print(trainTable)
  message("Contingency Table for Test Data")
  print(testTable)
  message("Accuracy")
  print(round(cbind(trainAccuracy=trainAcc, testAccuracy=testAcc),3))
}
printALL(NBclassfier)

#
# 3.3 Decision Tree Algorithem : Classfication
#

head(transactionsDT)
#
#Spliiting the Data set 
#
set.seed(424)
transactionsDT$isFraud <- as.factor(transactionsDT$isFraud,levels=c(1,0))
ran <- sample(1:nrow(transactionsDT), 0.7 * nrow(transactionsDT)) 
transactions_train <- transactionsDT[ran,] 
transactions_test <- transactionsDT[-ran,] 
transactions_train_lables <- transactionsDT[ran,9]
transactions_test_lables <- transactionsDT[-ran,9]
head(transactions_train)

#
# Training the model using C5.0 Algorithem
#
credit_model <- C5.0(transactions_train[-6], as.factor(transactions_train$isFraud))
credit_model
summary(credit_model)
#
# Predicting results and evalating model
#
credit_pred <- predict(credit_model, transactions_test)
CrossTable(y = transactions_test$isFraud, x = credit_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('Predicted','Actual'))


#
# 3.4 Random Forest Algorithem : Classfication 
#

#set.seed(1)
#
#Trainaing the model
#
set.seed(18)
transactions_train$isFraud = as.factor(transactions_train$isFraud)
fit_forest <- randomForest(isFraud ~ type_TRANSFER+oldbalanceOrg+newbalanceOrig+oldbalanceDest+newbalanceDest+adjustedBalanceOrg, data = transactions_train, ntree = 20, mtry = 3)
plot(fit_forest)
#
# Ploting the importnat features in data set
#
importance_matrix <- data.frame(Variables = rownames(fit_forest$importance), fit_forest$importance, row.names = NULL)
ggplot(data = importance_matrix , aes(y = MeanDecreaseGini , x = Variables, fill = Variables))+ geom_col() + coord_flip() + labs(title= 'Variiable importance plot')+ theme_classic()
#
# testing the model and evalvating model performance 
#
pred <- predict(fit_forest, newdata = transactions_test)

CrossTable(y = transactions_test$isFraud, x = pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('Predicted','Actual'))
confusion <- confusionMatrix(pred, transactions_test$isFraud )
print(confusion)

#
# 3.5 Logistic Regression Algorithem : Classfication
#

head(transactionsDT)
#
# spliting teh test and train data sets 
#
set.seed(129)
ran <- sample(1:nrow(transactionsDT), 0.8 * nrow(transactionsDT)) 
transactions_train <- transactionsDT[ran,] 
transactions_test <- transactionsDT[-ran,] 
transactions_train_lables <- transactionsDT[ran,9]
transactions_test_lables <- transactionsDT[-ran,9]
head(transactions_train)

#
# Trainaing the model
#

#factor_vars <- c ("type_TRANSFER", "type_CASH_OUT")
#continuous_vars <- c("oldbalanceOrg", "newbalanceOrig","oldbalanceDest", "newbalanceDest", "hour", "adjustedBalanceOrg","adjustedBalanceDest")

logitMod <- glm(isFraud~type_TRANSFER+oldbalanceOrg+newbalanceOrig+oldbalanceDest+newbalanceDest+adjustedBalanceOrg, data=transactions_train, family=binomial(link="logit"))
summary(logitMod)
#
# testing the model and Predcting the values 
#
predicted <- plogis(predict(logitMod, transactions_test)) 
predicted
#
#Evalavating the performance of the model -Model Diagnostics
#
#Finding the Optimal cutoff score for classfication
#
optCutOff <- optimalCutoff(transactions_test$isFraud, predicted)[1] 
#
#Confusion Matrix
#
cm<-confusionMatrix(transactions_test$isFraud, predicted, threshold = optCutOff)
cm
#
#Accuracey
#
Accuracy<- 1-misClassError(transactions_test$isFraud, predicted, threshold = optCutOff)
Accuracy
#
#Misclassification Error
#
misClassError(transactions_test$isFraud, predicted, threshold = optCutOff)
#
#Specificity and Sensitivity
#
sensitivity(transactions_test$isFraud, predicted, threshold = optCutOff)
specificity(transactions_test$isFraud, predicted, threshold = optCutOff)
#
#Flase Postive Rate
#
(FPR<-1-specificity(transactions_test$isFraud, predicted, threshold = optCutOff))
#Concordance
#
Concordance(transactions_test$isFraud, predicted)
#
#VIF
#
install.packages('car')
library(car)
vif(logitMod)
Accuracy
misClassError(transactions_test$isFraud, predicted, threshold = optCutOff)
sensitivity(transactions_test$isFraud, predicted, threshold = optCutOff)
specificity(transactions_test$isFraud, predicted, threshold = optCutOff)
#Flase Postive Rate
(FPR<-1-specificity(transactions_test$isFraud, predicted, threshold = optCutOff))

#Concordance
Concordance(transactions_test$isFraud, predicted)

