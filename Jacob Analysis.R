library(dplyr)
library(glmnet)
library(boot)
library(leaps)
library(Matrix)
library(pls)
library(fastDummies)

# Cleaning data and removing outliers

df <- read.csv("WineData.csv")

df <- na.omit(df)
df <- subset(df, select = -c(X, Id, Name, Winery, StyleName, Region))
df <- subset(df, Acidity != -1 & Year > 1950 & Country != "Mexico" & Country != "NONE")

df <- dummy_cols(df, select_columns = 'Type')
df <- dummy_cols(df, select_columns = 'Vintage')
df <- dummy_cols(df, select_columns = 'Nat')
df <- dummy_cols(df, select_columns = 'Country')
df <- dummy_cols(df, select_columns = 'Size')

#df <- subset(df, select = c(Type, Vintage, Nat, Country, Size))

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

# Training Linear Models and Checking for Non-constant Error Variance

set.seed(1)
trainPrice <- sample(1:nrow(dfPrice.post), nrow(dfPrice.post) * 0.6)
trainRating <- sample(1:nrow(dfRating.post), nrow(dfRating.post) * 0.6)

Price_lm.fit <- glm(Price ~ ., data = dfPrice.post)
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

plot(dfPrice.post_test$Price, Price_resid.transform, xlab = "Fitted values", ylab = "Residuals", main = "log(Price) Response")
plot(dfRating.post_test$Rating, Rating_resid.transform, xlab = "Fitted values", ylab = "Residuals", main = "log(Rating) Response")

mean((Price_resid.transform)^2)
mean((Rating_resid.transform)^2)

summary(Price_lm.transform_fit)
summary(Rating_lm.transform_fit)

# Best Subset Selection

Price <- dfPrice.post[1]

dfPrice.post <- as.data.frame(scale(dfPrice.post[2:11]))

dfPrice.post <- cbind(Price, dfPrice.post)

Price_regfit.full <- regsubsets(log(Price) ~ ., data = dfPrice.post[trainPrice,], nvmax = 30)
Rating_regfit.full <- regsubsets(log(Rating) ~ ., data = dfRating.post[trainRating,], nvmax = 30)

plot(summary(Price_regfit.full)$adjr2, type="b", xlab="# of Predictors", ylab=expression(R^2), main = "Best Subset Selection of Price Regression - R^2")
plot(summary(Rating_regfit.full)$adjr2, type = "b", xlab = "p", ylab = expression(R^2), main = "Best Subset Selection of Rating Regression - R^2")
plot(summary(Price_regfit.full)$cp, type="b", xlab="p", ylab=expression(C[p]), main = "Best Subset Selection of Price Regression - Cp")
plot(summary(Rating_regfit.full)$cp, type="b", xlab="p", ylab=expression(C[p]), main = "Best Subset Selection of Rating Regression - Cp")

which.max(summary(Price_regfit.full)$adjr2)
which.min(summary(Price_regfit.full)$cp)
which.max(summary(Rating_regfit.full)$adjr2)
which.min(summary(Rating_regfit.full)$cp)

summary(Price_regfit.full)$adjr2[24]
summary(Price_regfit.full)$cp[20]

summary(Rating_regfit.full)$adjr2[20]

coef(Price_regfit.full, 20)
coef(Rating_regfit.full, 23)

test_mat = model.matrix (Rating~., data = dfRating.post[-trainRating,])
coefi = coef(Rating_regfit.full, id = 20)
pred = test_mat[,names(coefi)]%*%coefi
val_errors = mean((dfRating.post$Rating[-trainRating]-pred)^2)