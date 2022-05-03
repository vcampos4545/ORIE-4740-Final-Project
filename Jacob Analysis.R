library(dplyr)
library(glmnet)
library(boot)
library(leaps)
library(Matrix)
library(pls)

# Cleaning data and removing outliers

df <- read.csv("WineData.csv")

df <- na.omit(df)
df <- subset(df, select = -c(X, Id, Name, Winery, StyleName, Region))
df <- subset(df, Acidity != -1 & Year > 1950 & Country != "Mexico")

df$Type <- as.factor(df$Type)
df$Type <- model.matrix( ~ Type - 1, data=df)

df$Vintage <- as.factor(df$Vintage)
df$Vintage <- model.matrix( ~ Vintage - 1, data=df)

df$Nat <- as.factor(df$Nat)
df$Nat <- model.matrix( ~ Nat - 1, data=df)

df$Country = as.factor(df$Country)
df$Country <- model.matrix( ~ Country - 1, data=df)

df$Size <- as.factor(df$Size)
df$Size <- model.matrix( ~ Size - 1, data=df)

df.Pricelm <- lm(Price ~ ., data=df)
df.Ratinglm <- lm(Rating ~ ., data = df)

dfPriceRes <- rstudent(df.Pricelm)
dfRatingRes <- rstudent(df.Ratinglm)

dfPrice.post <- df[abs(dfPriceRes) < 3,]
dfRating.post <- df[abs(dfRatingRes) < 3,]

dfPrice.post <- na.omit(dfPrice.post)
dfRating.post <- na.omit(dfRating.post)

# PCA to pick Largest Components

Price_pr.out <- prcomp(dfPrice.post[-c(1)], scale=TRUE)

Price_pcr.fit <- pcr(Price ~ ., data = dfPrice.post, scale=TRUE, validation = "CV")

test <- predict(Price_pr.out, dfPrice.post[-c(1)])

dfPrice.post[ , which(apply(dfPrice.post, 2, var) == 0)]


which(apply(dfRating.post, 2, var)==0)
dfRating.post[ , which(apply(dfRating.post, 2, var) != 0)]

Rating_pr.out <- prcomp(subset(dfRating.post, select = -c(Rating)) , scale=TRUE)

#

rankifremoved <- sapply(1:ncol(dfPrice.post), function (x) qr(dfPrice.post[,-x])$rank)
which(rankifremoved == max(rankifremoved))

rankifremoved <- sapply(1:ncol(dfRating.post), function (x) qr(dfRating.post[,-x])$rank)
which(rankifremoved == max(rankifremoved))

# Training Linear Models and Checking for Non-constant Error Variance

set.seed(1)
trainPrice <- sample(1:nrow(dfPrice.post), nrow(dfPrice.post) * 0.6)
trainRating <- sample(1:nrow(dfRating.post), nrow(dfRating.post) * 0.6)

Price_lm.fit <- lm(Price ~ ., data = dfPrice.post, subset = trainPrice)
Rating_lm.fit <- lm(Rating ~ ., data = dfRating.post, subset = trainRating)

dfPrice.post_test <- dfPrice.post[-trainPrice,]
dfRating.post_test <- dfRating.post[-trainRating,]

Price.predict <- predict.lm(Price_lm.fit, dfPrice.post_test, type= "response")
Rating.predict <- predict.lm(Rating_lm.fit, dfRating.post_test, type="response")

mean((dfPrice.post_test$Price - Price.predict)^2)
mean((dfRating.post_test$Rating - Rating.predict)^2)

Price_resid <- dfPrice.post_test$Price - Price.predict
Rating_resid <- dfRating.post_test$Rating - Rating.predict

plot(dfPrice.post_test$Price, Price_resid, xlab = "Fitted values", ylab = "Residuals", main = "Price Response")
plot(dfRating.post_test$Rating, Rating_resid, xlab = "Fitted values", ylab = "Residuals", main = "Rating Response")

Price_lm.transform_fit <- lm(log(Price) ~ ., data = dfPrice.post, subset = trainPrice)
Rating_lm.transform_fit <- lm(log(Rating) ~ ., data = dfRating.post, subset = trainRating)

Price.predict_transform <- predict.lm(Price_lm.transform_fit, dfPrice.post_test, type= "response")
Rating.predict_transform <- predict.lm(Rating_lm.transform_fit, dfRating.post_test, type="response")

Price_resid.transform <- dfPrice.post_test$Price - Price.predict_transform
Rating_resid.transform <- dfRating.post_test$Rating - Rating.predict_transform

plot(dfPrice.post_test$Price, Price_resid.transform, xlab = "Fitted values", ylab = "Residuals", main = "Price Response")
plot(dfRating.post_test$Rating, Rating_resid.transform, xlab = "Fitted values", ylab = "Residuals", main = "Rating Response")

# 5 vs 10 Fold Cross Validation (Pointless)

Price_glm.fit <- glm(Price ~ ., data = dfPrice.post)
Rating_glm.fit <- glm(Rating ~ ., data = dfRating.post)

Price_cv.error5 = cv.glm(dfPrice.post, Price_glm.fit, K=5)$delta[1]
Rating_cv.error5 = cv.glm(dfRating.post, Rating_glm.fit, K=5)$delta[1]
Price_cv.error10 = cv.glm(dfPrice.post, Price_glm.fit, K=5)$delta[1]
Rating_cv.error10 = cv.glm(dfRating.post, Rating_glm.fit, K=10)$delta[1]

# Best Subset Selection

Price <- dfPrice.post[1]

dfPrice.post <- as.data.frame(scale(dfPrice.post[2:11]))

dfPrice.post <- cbind(Price, dfPrice.post)

Price_regfit.full <- regsubsets(Price ~ ., data = dfPrice.post, nvmax = 30)
Rating_regfit.full <- regsubsets(Rating ~ ., data = dfRating.post, nvmax = 30)

plot(summary(Price_regfit.full)$adjr2, type="b", xlab="p", ylab=expression(R^2))
plot(summary(Rating_regfit.full)$adjr2, type = "b", xlab = "p", ylab = expression(R^2))
plot(summary(Price_regfit.full)$bic, type="b", xlab="p", ylab="BIC")
plot(summary(Rating_regfit.full)$bic, type="b", xlab="p", ylab="BIC")
plot(summary(Price_regfit.full)$cp, type="b", xlab="p", ylab=expression(C[p]))
plot(summary(Rating_regfit.full)$cp, type="b", xlab="p", ylab=expression(C[p]))
