wine <- read.csv("WineData.csv")
wine <- subset(wine, select = -c(X, Id, Name,StyleName,Region,Winery))
wine <- subset(wine, Body != -1 & Acidity != -1 & Country != 'NONE' & Year >0)
wine <- as_tibble(wine)
wine$Type <- as.factor(wine$Type)
wine$Vintage <- as.factor(wine$Vintage)
wine$Nat <- as.factor(wine$Nat)
wine$Acidity <- as.factor(wine$Acidity)
wine$Country <- as.factor(wine$Country)
#wine$Size <- as.factor(wine$Size)
wine$Price <- log(wine$Price)
wine$Rating <- scale(wine$Rating)[1,]

# GAM predicting Price
set.seed(2)
train <- sample(1:nrow(wine), 0.6*nrow(wine))
wineTrain <- wine[train,]
wineTest <- wine[-train,]
par(mfrow=c(2,5))
gam1 <- gam(Price~s(Rating,4)+s(NumRatings,4)+Type+s(Year,4)+s(Body,4)+Acidity+Vintage+Nat+s(Size,4)+Country,data = wineTrain)
plot(gam1, se=TRUE, col="red")




# GAM predicting Rating
gam2 <- gam(Rating~s(Price,4)+s(NumRatings,4)+Type+s(Year,4)+s(Body,4)+Acidity+Vintage+Nat+Size+Country,data = wine)
par(mfrow=c(2,5))
plot(gam2, se=TRUE, col="red")
