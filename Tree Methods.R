wine <- read.csv("WineData.csv")
wine <- subset(wine, select = -c(X, Id, Name,StyleName,Region,Winery))
wine <- subset(wine, Body != -1 & Acidity != -1 & Country != 'NONE')
wine <- as_tibble(wine)
wine$Type <- as.factor(wine$Type)
wine$Vintage <- as.factor(wine$Vintage)
wine$Nat <- as.factor(wine$Nat)
wine$Acidity <- as.factor(wine$Acidity)
wine$Country <- as.factor(wine$Country)
wine$Size <- as.factor(wine$Size)
wine$Price <- log(wine$Price)



set.seed(1)
train <- sample(1:nrow(wine), 0.6*nrow(wine))
wineTrain <- wine[train,]
wineTest <- wine[-train,]
par(mfrow=c(1,1))


# Regression Tree predicting Price
tree.wine <- tree(Price~.,wine,subset = train)
summary(tree.wine)
plot(tree.wine)
text(tree.wine,pretty=0)
tree.pred <- predict(tree.wine, wineTest)
tree.mse_nocv <- mean((tree.pred - wineTest$Price)^2)
cv.wine <- cv.tree(tree.wine)
bestSize <- cv.wine$size[which.min(cv.wine$dev)]
prune.wine <- prune.tree(tree.wine, best=bestSize)
tree.pred <- predict(prune.wine, wineTest)
tree.mse_cv <- mean((tree.pred - wineTest$Price)^2)
plot(cv.wine$size,cv.wine$dev)


# Regression Tree predicting Rating
tree.wine <- tree(Rating~.,wine,subset = train)
summary(tree.wine)
plot(tree.wine)
text(tree.wine,pretty=0)
tree.pred <- predict(tree.wine, wineTest)
tree.mse_nocv <- mean((tree.pred - wineTest$Rating)^2)
cv.wine <- cv.tree(tree.wine)
bestSize <- cv.wine$size[which.min(cv.wine$dev)]
prune.wine <- prune.tree(tree.wine, best=bestSize)
tree.pred <- predict(prune.wine, wineTest)
tree.mse_cv <- mean((tree.pred - wineTest$Rating)^2)
plot(cv.wine$size,cv.wine$dev)


# Bagging predicting Price
bag.wine <- randomForest(Price~., wine, subset=train, mtry=10, importance=TRUE)
bag.pred <- predict(bag.wine, newdata=wineTest)
mean((bag.pred - wineTest$Price)^2)

# Bagging predicting Rating
bag.wine <- randomForest(Rating~., wine, subset=train, mtry=10, importance=TRUE)
bag.pred <- predict(bag.wine, newdata=wineTest)
mean((bag.pred - wineTest$Rating)^2)


# Random Forests Predicting Price
rf.errs <- rep(0, 10)
for (m in c(1:10)) {
  rf.wine <- randomForest(Price~., wine, subset=train,mtry=m, importance=TRUE)
  rf.pred <- predict(rf.wine, newdata=wineTest)
  rf.errs[m] <- mean((rf.pred - wineTest$Price)^2)
}
plot(rf.errs, type="b")


# Random Forests Predicting Rating
rf.errs <- rep(0, 10)
for (m in c(1:10)) {
  rf.wine <- randomForest(Rating~., wine, subset=train,mtry=m, importance=TRUE)
  rf.pred <- predict(rf.wine, newdata=wineTest)
  rf.errs[m] <- mean((rf.pred - wineTest$Rating)^2)
}
plot(rf.errs, type="b")



train <- wineTrain
test <- wineTest
# fit model
bst <- xgboost(data = train, label = train$label, max.depth = 2, eta = 1, nrounds = 2,
               nthread = 2, objective = "binary:logistic")
# predict
pred <- predict(bst, test)




lambda <- rep(0,10)
for (i in 1:10) {
  lambda[i] <- 2^(-i)
}
for (i in lambda) {
  boost.wine <- gbm()
}