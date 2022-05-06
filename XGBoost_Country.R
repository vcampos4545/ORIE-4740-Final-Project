#All code refrenced from 'Demonstration of XGBoost'


library(dplyr)
library(glmnet)
library(boot)
library(leaps)
library(Matrix)
library(pls)

# Cleaning data and removing outliers
df <- read.csv("WineData.csv")
df <- na.omit(df)
df <- subset(df, Acidity != -1 & Year > 1950 & Country != "Mexico" & Country != "Hungary" & Country != "Uruguay")
df <- subset(df, select = -c(X, Id, Name, Winery,StyleName,Region))


df$Type <- as.factor(df$Type)
df$Vintage <- as.factor(df$Vintage)
df$Nat <- as.factor(df$Nat)
df$Country = as.factor(df$Country)
df$Size <- as.factor(df$Size)


df.Pricelm <- lm(Price ~ ., data=df)
df.Ratinglm <- lm(Rating ~ ., data = df)

dfPriceRes <- rstudent(df.Pricelm)
dfRatingRes <- rstudent(df.Ratinglm)


dfPrice.post <- df[abs(dfPriceRes) < 3,]
dfRating.post <- df[abs(dfRatingRes) < 3,]

wine.price <- na.omit(dfPrice.post)
wine.rating <- na.omit(dfRating.post)

wine.price$NumRatings <- log(wine.price$NumRatings)
wine.price$Price <- log(wine.price$Price)



data <- wine.price
data <- data[,!names(data) %in% 'id']

# Check data dimension
dim(data); sum(data[,1:(ncol(data)-1)] == 0)/(nrow(data)*(ncol(data)-1))

# Check response variable

set.seed(1)
train_ind <- sample(1:nrow(data), 2/3*nrow(data))
train <- data[train_ind,]
test  <- data[-train_ind,]

train.x <- train[, !names(train) %in% 'Country'] # extract predictors from training set
test.x  <- test[, !names(test) %in% 'Country']  # extract predictors from test set
train.y <- train$Country # extract response from training set
test.y  <- test$Country  # extract response from test set


require(nnet)
ptm <- proc.time()
set.seed(1)
lr <- multinom(Country ~ ., data = train,MaxNWts = 25000)
print(lr.time <- proc.time() - ptm)  

lr.prob <- predict(lr, newdata = test, "probs")
lr.pred <- apply(lr.prob, 1, which.max)    # use the class with maximum probability as prediction
table(lr.pred, test.y) # show confusion matrix
print(lr.acc <- mean(lr.pred == as.numeric(test.y))) # classification accuracy on test set


require(class)
data.x.scaled  <- scale( data[, !names(data) %in% 'Country'] ) # normalize training and test set together
train.x.scaled <- data.x.scaled[train_ind, ]  # then split training and test sets
test.x.scaled  <- data.x.scaled[-train_ind, ] # then split training and test sets

ptm <- proc.time()
set.seed(1)
knn.pred <- knn(train.x.scaled, 
                test.x.scaled, 
                train.y, 
                k=10)                # number of neighbors
print(knn.time <- proc.time() - ptm) 

table(knn.pred, test.y) # show confusion matrix
print(knn.acc <- mean(knn.pred == test.y))      # classification accuracy on test set


require(gbm)

ptm <- proc.time()
set.seed(1)
tree <- gbm(Country~., 
            data=train, 
            distribution="multinomial",   # for multi-class problem
            n.trees=200,                  # number of trees
            interaction.depth=4,          # d = tree size
            shrinkage=0.05)               # shrinkage parameter
print(tree.time <- proc.time() - ptm)     # running time: 4.5 min on my laptop



tree.prob <- predict(tree, newdata=test[,], n.tree=200, Country='response')
tree.pred <- apply(tree.prob, 1, which.max)  # use the class with maximum probability as prediction
#table(tree.pred, test.y) # show confusion matrix
print(tree.acc <- mean(tree.pred == as.numeric(test.y))) # classification accuracy on test set


## --- XGBoost ---
require(xgboost)
trainXMatrix <- as.matrix(train.x)     # XGBoost only accepts matrices not data frame
storage.mode(trainXMatrix) <- 'double' # Matrix must be real valued, not integer
testXMatrix <- as.matrix(test.x)       # convert data frame to matrices
storage.mode(testXMatrix) <- 'double'  # convert integer to real
trainYvec <- as.integer(train.y) -1    # extract response from training set; class label starts from 0
testYvec  <- as.integer(test.y) -1     # extract response from test set; class label starts from 0
numberOfClasses <- max(trainYvec) + 1

# algorithm parameters for XGBoost
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)
nround <- 200 # number of rounds/trees

ptm <- proc.time()
set.seed(1)
xgbtree <- xgboost(param=param, 
                   data = trainXMatrix, 
                   label = trainYvec, 
                   nrounds = nround,  # number of trees
                   max.depth = 4,     # tree depth (not the same as interaction.depth in gbm!)
                   eta = 0.3)           # shrinkage parameter
print(xgbtree.time <- proc.time() - ptm)         

xgbtree.prob <- predict(xgbtree, testXMatrix)   
xgbtree.prob <- t( matrix(xgbtree.prob, nrow=numberOfClasses, ncol=nrow(test.x)) ) # need to convert it to a matrix
xgbtree.pred <-apply(xgbtree.prob, 1, which.max) # use the class with maximum probability as prediction
table(xgbtree.pred, test.y) # show confusion matrix
print(xgbtree.acc <- mean(xgbtree.pred == as.numeric(test.y)))  # classification accuracy on test set

## --- XGBoost with sparse data ---

require(Matrix)
trainXSMatrix <- sparse.model.matrix(Country~.-1, data = train) # use sparse matrix for predictors in training set
testXSMatrix  <- sparse.model.matrix(Country~.-1, data = test)  # use sparse matrix for predictors in test set
ptm <- proc.time()
set.seed(1)
xgbstree <- xgboost(param=param, 
                    data = trainXSMatrix, 
                    label = trainYvec, 
                    nrounds = nround, # same number of trees as before
                    max.depth = 4,    # tree depth (not the same as interaction.depth in gbm!)
                    eta = 0.5)        # shrinkage parameter
print(xgbstree.time <- proc.time() - ptm)          # running time: .5 min on my laptop

xgbstree.prob <- predict(xgbstree, testXSMatrix)   # this is a long vector
xgbstree.prob <- t( matrix(xgbstree.prob, nrow=numberOfClasses, ncol=nrow(test.x)) ) # need to convert it to a matrix
xgbstree.pred <-apply(xgbstree.prob, 1, which.max) # use the class with maximum probability as prediction
table(xgbstree.pred, test.y) # show confusion matrix
print(xgbstree.acc<- mean(xgbstree.pred == as.numeric(test.y)))


imp_mat <- xgb.importance(model = xgbstree)
xgb.plot.importance(importance_matrix = imp_mat[1:20,])


require(knitr)
sumtable <- data.frame(
  "Algorithm" = c("Logistic Regression", "kNN", "Boosted Tree", "XGBoost", "XGBoost Sparse"),
  "Time_in_sec" = c(lr.time[3], knn.time[3], tree.time[3], xgbtree.time[3], xgbstree.time[3]),
  "Accuracy" = c(lr.acc, knn.acc, tree.acc, xgbtree.acc, xgbstree.acc),
  stringsAsFactors = FALSE)
kable(sumtable, caption="Summary", digits = c(0,0,3))