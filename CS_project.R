install.packages("ggplot2")
install.packages("randomForest")
install.packages("logistf")
library(ggplot2)
library(randomForest)
library(rpart)
library(rpart.plot)
library(tree)
library(logistf)
library(Metrics)
library(caTools)
library(caret)
library(maxLik)

#Data Generating Process
#Generating independent variables with a re-usable function(using logodds) 
set.seed(100)
n<-1500
data.gen <- function(n)
{
  n_var<-20
  beta0 <- 0.2
  beta1 <- 0.3
  beta2 <- 0.6
  beta3 <- 0.5
  beta4 <- -0.8
  beta5 <- 1
  beta6 <- -0.9
  beta7 <-- 0.65
  beta8 <- 0.35
  beta9 <- 0.45
  beta10 <- 0.85
  beta11 <- -0.1
  beta12 <- 0.6
  beta13 <- -0.9
  beta14 <- 0.4
  beta15 <- -0.55
  beta16 <- 0.6
  beta17 <- -0.75
  beta18 <- 0.25
  beta19 <-- 0.15

  beta <- rbind(beta0,beta1,beta2,beta3,beta4,beta5,beta6,beta7,beta8,beta9,beta10,beta11,beta12,beta13,beta14,beta15,beta16,beta17,beta18,beta19)

  X0 <- rep(1, n)
  X1<-c(runif(n, runif(1, 0, 2), runif(1,3,5)))
  X<-cbind(X0,X1)
  varnames<-c("X0","X1")
    for (var in 1:(n_var-2))
    {
      X<-cbind(X, c(runif(n, runif(1, -2, 2), runif(1,3,5))))
      varnames <- c(varnames, paste("X", var+1, sep=""))
    }

  print(X)
  print(beta)
  
  #beta <- runif(n_var, -1, 1)
  logodds <- X %*% beta
  pi_x <- 1 / (1 + exp(-logodds))
  Turnover <- rbinom(n, 1, prob = pi_x)
  varnames<-c(varnames, "Turnover")
  data <- cbind.data.frame(X, as.factor(Turnover))
  colnames(data)<-varnames
  return(data)
}



data<-data.gen(1500)

## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

## set the seed to make the partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train.1 <- data[train_ind, ]
test.1 <- data[-train_ind, ]

train.logit = glm(Turnover ~  X0+X1 + X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19, data = train.1, family = binomial(link = "logit"))

built.prob.x.train.fit = predict(train.logit, train.1, type = "response")
built.prob.x.test.fit = predict(train.logit, test.1, type = "response")
built.log.odds.train.fit = predict(train.logit, train.1, se = T)$fit
built.log.odds.test.fit = predict(train.logit, test.1, se = T)$fit

built.log.odds.train.fit[smp_size]

y.pred.train.1 = c()
y.pred.test.1 = c()
threshold = 0.65
for(i in 1:length(built.log.odds.train.fit)){
  if(built.log.odds.train.fit[i] >= threshold){
    y.pred.train.1[i] = 1
  }else{
    y.pred.train.1[i] = 0 
  }
}
for (i in 1:length(built.prob.x.test.fit)){
  if(built.prob.x.test.fit[i] >= threshold){
    y.pred.test.1[i] = 1
  }else{
    y.pred.test.1[i] = 0
  }  
}
y.pred.train.1


# Training Mean squared Error is the average of the sum of the squared difference between true y and predicted y
# Test Average squared Error is just like MSE, but on a test data set
logistic.mse = sum((as.numeric(test.1$Turnover) - as.numeric(y.pred.test.1))**2) / (nrow(test.1))
print(logistic.mse)
plot(train.logit)
plot(logistic.mse)


# Confusion Matrix for test and train data sets and their sums in the margins
conf.trains = table(y.pred.train, train.1$Turnover, dnn = c("Predicted ys", "True ys"))
conf.tests = table(y.pred.test, test.1$Turnover, dnn = c("Predicted ys", "True ys"))
 
# Confusion Matrix with proportions and their sums in the margins:
conf.prop.trains = addmargins(prop.table(as.table(conf.trains)))
conf.prop.tests = addmargins(prop.table(as.table(conf.tests)))
print(conf.tests)
plot(conf.tests)
print(conf.trains)
plot(conf.trains)
print(conf.prop.trains)
print(conf.prop.tests)


#Regression tree
Regression.Tree.FUll <- rpart(Turnover~ X0+X1 + X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19, data = train.1, cp = -1)
plot(Regression.Tree.FUll, uniform = TRUE)
text(Regression.Tree.FUll, use.n=TRUE, all=TRUE, cex=.80)
fullTree.MSE <- (1/nrow(test.1))*(sum(as.numeric((predict(Regression.Tree.FUll , test.1))-as.numeric(test.1$Turnover))^2))
printcp(Regression.Tree.FUll)
plotcp(Regression.Tree.FUll)
print(fullTree.MSE)

#var((predict(tree1, test.1)))

#print((predict(tree1, test)))
#confusion matrix 
#conf.tests.tree = table((predict(tree1, test.1)), test.1$Turnover, dnn = c("Predicted ys", "True ys"))
conf.tests.tree = table((predict(Regression.Tree.FUll, test.1)), test.1$Turnover, dnn = c("Predicted ys", "True ys"))
plot(conf.tests.tree) 


#pruned tree
pruneRegTree <- rpart(formula = Turnover~ X0+X1 + X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19, data = train.1)
plot(pruneRegTree, uniform = TRUE)
text(pruneRegTree, use.n=TRUE, all=TRUE, cex=.8)
pruneTree.MSE <- (1/nrow(test.1))*(sum(as.numeric((predict(pruneRegTree, test.1))-as.numeric(test.1$Turnover))^2))
print(pruneTree.MSE)
printcp(pruneRegTree)
plotcp(pruneRegTree)



#Random Forest

rf <- randomForest(Turnover ~ ., data=train.1, importance=TRUE, proximity=TRUE)
print(rf) 
plot(rf)
mse_RF <- (1/nrow(test.1))*(sum((as.numeric(predict(rf, test.1))-as.numeric(test.1$Turnover))^2))
print(mse_RF)
conf.tests.rf = table((predict(rf, test.1)), test.1$Turnover, dnn = c("Predicted ys", "True ys"))
plot(conf.tests.rf) 

# Generate plot

MSE.chart <- c(logistic.mse, mse_RF, fullTree.MSE)
M <- c("Logistic","RF","RTree")
barplot(MSE.chart,names.arg=M,xlab="Method",ylab="Error",col="blue",
        main="MSE Comparison",border="red")

#simulation study
mse_logistic_simulation <- c() 
mse_tree_simulation <- c() 
confusion.matrix_rf <- c()

logistic.mse = 0
fullTree.MSE = 0
mse_RF = 0

for (i in 1:100 ) 
{
     df_simulation <- data.gen(2000)
     ## 75% of the sample size
     smp_size <- floor(0.75 * nrow(df_simulation))
     
     ## set the seed to make your partition reproducible
     set.seed(123)
     train_ind <- sample(seq_len(nrow(data)), size = smp_size)
     
     train <- df_simulation[train_ind, ]
     test <- df_simulation[-train_ind, ]
     
     train.logit = glm(Turnover ~  X0+X1 + X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19, data = train, family = binomial(link = "logit"))
     
     built.prob.x.train.fit = predict(train.logit, train, type = "response")
     built.prob.x.test.fit = predict(train.logit, test, type = "response")
     built.log.odds.train.fit = predict(train.logit, train, se = T)$fit
     built.log.odds.test.fit = predict(train.logit, test, se = T)$fit
     
     built.log.odds.train.fit[smp_size]
     
     y.pred.train = c()
     y.pred.test = c()
     threshold = 0.65
     for(i in 1:length(built.log.odds.train.fit)){
       if(built.log.odds.train.fit[i] >= threshold){
         y.pred.train[i] = 1
       }else{
         y.pred.train[i] = 0 
       }
     }
     for (i in 1:length(built.prob.x.test.fit)){
       if(built.prob.x.test.fit[i] >= threshold){
         y.pred.test[i] = 1
       }else{
         y.pred.test[i] = 0
       }  
     }
     
     tree1 <- rpart(Turnover~ X0+X1 + X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19, data = train, cp = -1)
     
     rf <- randomForest(Turnover ~ ., data=train, importance=TRUE, proximity=TRUE)
     
     logistic.mse <- logistic.mse + sum((as.numeric(test$Turnover) - as.numeric(y.pred.test))**2) / (nrow(test))
     fullTree.MSE <- fullTree.MSE + (1/nrow(test))*(sum(as.numeric((predict(tree1, test))-as.numeric(test$Turnover))^2))    
     mse_RF <- mse_RF + (1/nrow(test))*(sum((as.numeric(predict(rf, test))-as.numeric(test$Turnover))^2))
     
}

logistic.mse <- logistic.mse / 500
fullTree.MSE <- fullTree.MSE / 500
mse_RF <- mse_RF / 500

print(logistic.mse)
print(fullTree.MSE)
print(mse_RF)
#how can i include the mse of logistic
# how can i implement a mean of the confusion matrices.
