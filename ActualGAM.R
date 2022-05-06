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
df <- subset(df, select = -c(X, Id, Name, Winery, StyleName, Region))


df$Type <- as.factor(df$Type)
df$Vintage <- as.factor(df$Vintage)
df$Nat <- as.factor(df$Nat)
df$Country = as.factor(df$Country)
df$Size <- as.factor(df$Size)
#df$Rating <- as.factor(df$Rating)


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



# GAM predicting Price
par(mfrow=c(2,5))

gam1 <- gam(Price~s(Rating)+s(NumRatings)+s(Year)+s(Body)+Type+Acidity+Size+Nat+Vintage+Country,data = wine.price,select = TRUE)
plot(gam1, se=TRUE, col="red")

gam2 <- gam(Price~s(Rating)+s(NumRatings)+s(Year)+s(Body)+Type,data = wine.price,select = TRUE)
par(mfrow=c(1,5))
plot(gam2, se=TRUE, col="blue")



#gam2 <- gam(Price~s(Rating)+s(NumRatings)+s(Body)+Country,data = wine.price)
#gam3 <- gam(Price~s(Rating)+s(NumRatings)+Year+s(Body)+Country,data = wine.price)
#gam4 <- gam(Price~s(Rating)+s(NumRatings)+poly(Year,2)+s(Body)+Country,data = wine.price)
#anova(gam2,gam3,gam1)

#gam5 <- gam(Price~s(Rating)+s(NumRatings)+s(Year,2)+s(Body)+Country,data = wine.price)
#gam6 <- gam(Price~s(Rating)+s(NumRatings)+s(Year,5)+s(Body)+Country,data = wine.price)
#gam7 <- gam(Price~s(Rating)+s(NumRatings)+s(Year,8)+s(Body)+Country,data = wine.price)
#anova(gam5,gam1,gam6,gam7)

# GAM predicting Rating

# GAM predicting Rating


#gam2 <- gam(Rating~s(Price)+s(NumRatings)+Type+s(Year,4)+s(Body,4)+Acidity+Vintage+Nat+Size+Country,data = wine)
#par(mfrow=c(2,5))
#plot(gam2, se=TRUE, col="red")
