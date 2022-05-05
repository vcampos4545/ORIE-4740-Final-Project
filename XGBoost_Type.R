data <- wine.price
data <- data[,!names(data) %in% 'id']

# Check data dimension
dim(data); sum(data[,1:(ncol(data)-1)] == 0)/(nrow(data)*(ncol(data)-1))

# Check response variable

set.seed(1)
train_ind <- sample(1:nrow(data), 2/3*nrow(data))
train <- data[train_ind,]
test  <- data[-train_ind,]



train.x <- train[, !names(train) %in% 'Type'] # extract predictors from training set
test.x  <- test[, !names(test) %in% 'Type']  # extract predictors from test set
train.y <- train$Type # extract response from training set
test.y  <- test$Type  # extract response from test set

require(nnet)
ptm <- proc.time()
set.seed(1)
lr <- multinom(Type ~ ., data = train,MaxNWts = 25000)
print(lr.time <- proc.time() - ptm)  # running time: 0.5 min on my laptop

lr.prob <- predict(lr, newdata = test, "probs")
lr.pred <- apply(lr.prob, 1, which.max)    # use the class with maximum probability as prediction
#table(lr.pred, test.y) # show confusion matrix
print(lr.acc <- mean(lr.pred == as.numeric(test.y))) # classification accuracy on test set
# I got acurracy 0.7416368

require(class)
data.x.scaled  <- scale( data[, !names(data) %in% 'Type'] ) # normalize training and test set together
train.x.scaled <- data.x.scaled[train_ind, ]  # then split training and test sets
test.x.scaled  <- data.x.scaled[-train_ind, ] # then split training and test sets

ptm <- proc.time()
set.seed(1)
knn.pred <- knn(train.x.scaled, 
                test.x.scaled, 
                train.y, 
                k=10)                # number of neighbors
print(knn.time <- proc.time() - ptm) # running time: 2 min on my laptop

#table(knn.pred, test.y) # show confusion matrix
print(knn.acc <- mean(knn.pred == test.y))      # classification accuracy on test set
# I got accuracy 0.7676719

require(gbm)
# In the latest version of gbm, multi-class classification seems to have issues.
# Running the code below gives the warning:
# "Setting `distribution = "multinomial"` is ill-advised as it is currently broken. It exists only for backwards compatibility. Use at your own risk."

ptm <- proc.time()
set.seed(1)
tree <- gbm(Type~., 
            data=train, 
            distribution="multinomial",   # for multi-class problem
            n.trees=200,                  # number of trees
            interaction.depth=4,          # d = tree size
            shrinkage=0.05)               # shrinkage parameter
print(tree.time <- proc.time() - ptm)     # running time: 4.5 min on my laptop



tree.prob <- predict(tree, newdata=test[,], n.tree=200, Type='response')
tree.pred <- apply(tree.prob, 1, which.max)  # use the class with maximum probability as prediction
#table(tree.pred, test.y) # show confusion matrix
print(tree.acc <- mean(tree.pred == as.numeric(test.y))) # classification accuracy on test set
# I got accuracy 0.7791622

# Using n.tree=400 and shrinkage=0.05, I get 0.79 accuracy in 9 minutes


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
print(xgbtree.time <- proc.time() - ptm)         # running time: 1 min on my laptop

xgbtree.prob <- predict(xgbtree, testXMatrix)    # this is a long vector
xgbtree.prob <- t( matrix(xgbtree.prob, nrow=numberOfClasses, ncol=nrow(test.x)) ) # need to convert it to a matrix
xgbtree.pred <-apply(xgbtree.prob, 1, which.max) # use the class with maximum probability as prediction
table(xgbtree.pred, test.y) # show confusion matrix
print(xgbtree.acc <- mean(xgbtree.pred == as.numeric(test.y)))  # classification accuracy on test set


require(Matrix)
trainXSMatrix <- sparse.model.matrix(Type ~.-1, data = train) # use sparse matrix for predictors in training set
testXSMatrix  <- sparse.model.matrix(Type ~.-1, data = test)  # use sparse matrix for predictors in test set

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