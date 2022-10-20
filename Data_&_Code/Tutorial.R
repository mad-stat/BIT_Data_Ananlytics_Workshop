########################## K Nearest Neighbour #################################

# Installing Packages
install.packages("e1071")
install.packages("caTools")
install.packages("class")

# Loading package
library(e1071)
library(caTools)
library(class)

# Loading data
data(iris)
head(iris)

# Splitting data into train and test data
split <- sample.split(iris, SplitRatio = 0.7)
train_cl <- subset(iris, split == "TRUE")
test_cl <- subset(iris, split == "FALSE")

# Feature Scaling
train_scale <- scale(train_cl[, 1:4])
test_scale <- scale(test_cl[, 1:4])

# Fitting KNN Model to training dataset with k =1
classifier_knn <- knn(train = train_scale,
                      test = test_scale,
                      cl = train_cl$Species,
                      k = 1)
classifier_knn

# Confusiin Matrix
cm <- table(test_cl$Species, classifier_knn)
cm

# Model Evaluation - Choosing K
# Calculate out of Sample error

# k = 1
misClassError <- mean(classifier_knn != test_cl$Species)
print(paste('Accuracy =', 1-misClassError))

# K = 3
classifier_knn <- knn(train = train_scale,
                      test = test_scale,
                      cl = train_cl$Species,
                      k = 3)
misClassError <- mean(classifier_knn != test_cl$Species)
print(paste('Accuracy =', 1-misClassError))

# K = 5
classifier_knn <- knn(train = train_scale,
                      test = test_scale,
                      cl = train_cl$Species,
                      k = 5)
misClassError <- mean(classifier_knn != test_cl$Species)
print(paste('Accuracy =', 1-misClassError))

# K = 7
classifier_knn <- knn(train = train_scale,
                      test = test_scale,
                      cl = train_cl$Species,
                      k = 7)
misClassError <- mean(classifier_knn != test_cl$Species)
print(paste('Accuracy =', 1-misClassError))

# K = 15
classifier_knn <- knn(train = train_scale,
                      test = test_scale,
                      cl = train_cl$Species,
                      k = 15)
misClassError <- mean(classifier_knn != test_cl$Species)
print(paste('Accuracy =', 1-misClassError))

# K = 19
classifier_knn <- knn(train = train_scale,
                      test = test_scale,
                      cl = train_cl$Species,
                      k = 19)
misClassError <- mean(classifier_knn != test_cl$Species)
print(paste('Accuracy =', 1-misClassError))

################### K Means Algorithm #########################

install.packages("tidyverse")
install.packages("cluster")
install.packages("factoextra")

library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

# USArrests dataset in R
df <- USArrests
head(df)

# Computing Euclidean distance between rows

distance <- get_dist(df)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

# Computing k means algorithm with 2 centers
k2 <- kmeans(df, centers = 2, nstart = 25)
str(k2)
k2

# Plotting the cluster
fviz_cluster(k2, data = df)

# Standard pairwise scatter plots
df %>%
  as_tibble() %>%
  mutate(cluster = k2$cluster,
         state = row.names(USArrests)) %>%
  ggplot(aes(UrbanPop, Murder, color = factor(cluster), label = state)) +
  geom_text()

# Checking different number of centers
k3 <- kmeans(df, centers = 3, nstart = 25)
k4 <- kmeans(df, centers = 4, nstart = 25)
k5 <- kmeans(df, centers = 5, nstart = 25)

# plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = df) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = df) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = df) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = df) + ggtitle("k = 5")

installed.packages("gridExtra")
library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)

# Selecting the optimal k value using Elbow Method
set.seed(123)
fviz_nbclust(df, kmeans, method = "wss") # We select 4 centers

# Compute k-means clustering with k = 4
set.seed(123)
final <- kmeans(df, 4, nstart = 25)
print(final)
fviz_cluster(final, data = df)

# Extracting descriptive statistics from the clusters
USArrests %>%
  mutate(Cluster = final$cluster) %>%
  group_by(Cluster) %>%
  summarise_all("mean")

######################## Time Series Forecasting ##########################

####################### Time series analysis using R ###########################

############### Time series plot ###############
mydata <- read.csv("E15demand.csv")
E15 = ts(mydata$Demand, start = c(2012,4), end = c(2013,10), frequency = 12)
E15
plot(E15, type = "b")


E15 = ts(mydata$Demand)
E15
plot(E15, type = "b")

############# Trend in GDP data ###############
mydata <- read.csv("Trens_GDP.csv")
GDP <- ts(mydata$GDP, start = 1993, end = 2003)
plot(GDP, type = "b")

############# Seasonality in Sales data ###############
mydata <- read.csv("Seasonal_sales.csv")
sales = ts(mydata$Sales, start = c(2002,1), end = c(2005,12), frequency = 12)
plot(sales, type = "b")

############# Trend & Seasonality in TS data ###############
mydata <- read.csv("Trend_&_Seasonal.csv")
sales = ts(mydata$Sales)
plot(sales, type = "b")

################## Stationarity Test ########################
mydata <- read.csv("shipment.csv")
shipments = ts(mydata$Shipments)
plot(shipments, type = "b")

############# Stationarity in GDP data ###############
mydata <- read.csv("Trens_GDP.csv")
GDP <- ts(mydata$GDP, start = 1993, end = 2003)
plot(GDP, type = "b")

# Differencing
install.packages("forecast")
library(forecast) 
ndiffs(GDP)

mydiffdata = diff(GDP, difference = 1) 
plot(mydiffdata, type = "b")

######################### Autocorrelation ###########################
mydata <- read.csv("Trens_GDP.csv")
GDP <- ts(mydata$GDP, start = 1993, end = 2003)
acf(GDP, 3)
acf(GDP)

######################## ARIMA Model #################################
mydata <- read.csv("Visits.csv")
mydata <- ts(mydata$Data)
plot(mydata, type = "b")

# Descriptive Statistics
summary(mydata)

# Check whether the series is stationary
ndiffs(mydata)

#Draw ACF & PACF Graphs
acf(mydata)
pacf(mydata)

# ARIMA Model Fitting
mymodel = auto.arima(mydata)
mymodel

# Identification of model manually
arima(mydata, c(0,0,1))
arima(mydata, c(1,0,0))
arima(mydata, c(1,0,1))

# Model diagnostics
summary(mymodel)

####################### End of Session ######################################
