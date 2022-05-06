library(dplyr)
library(glmnet)
library(boot)
library(leaps)
library(Matrix)
library(pls)

# Cleaning data and removing outliers
df <- read.csv("WineData.csv")
df <- na.omit(df)
sum(df$Country_Uruguay)
df <- subset(df, Acidity != -1 & Year > 1950 & Country == 'Italy')
df <- subset(df, select = -c(X, Id, Name, Winery, StyleName, Country))

df$Region <- as.factor(df$Region)
df$Type <- as.factor(df$Type)
df$Vintage <- as.factor(df$Vintage)
df$Nat <- as.factor(df$Nat)
df$Country = as.factor(df$Country)
df$Size <- as.factor(df$Size)


#df <- dummy_cols(df, select_columns = c('Size','Nat','Country','Vintage'))
#df <- subset(df, select = -c(Size,Nat,Vintage,Country))

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


