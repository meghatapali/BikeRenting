rm(list=ls())


# set working directory
setwd("D:/Edwisor/Project 2")
getwd()

##############################################################

# loading Libraries
install.packages (c("tidyr", "ggplot2", "corrgram", "usdm", "caret", "DMwR", "rpart", "randomForest",'xgboost'))


# tidyr - drop_na
# ggplot2 - for visulization, boxplot, scatterplot
# corrgram - correlation plot
# usdm - vif
# caret - createDataPartition
# DMwR - regr.eval
# rpart - decision tree
# randomForest - random forest
# xgboost - xgboost



#############################################################

# loading dataset
df = read.csv("day.csv", header = T, na.strings = c(" ", "", "NA"))

######################
# Exploring Datasets
######################

# Structure of data
str(df)

# Summary of data
summary(df)

# Viewing the data

head(df,5)

#####################################
# EDA, Missing value and Outlier analysis
#####################################

#Variable "instant" can be dropped as it simply represents the index.
#casual and registered variables can be removed, as these two sums to dependent variable count
#Variable "dteday" can be ignored as output is not based on time series analysis. And we already have data in year and month column

df = subset(df, select = -c(instant,casual, registered, dteday) )

#Converting variables datatype to required datatypes
catnames=c("season","yr","mnth","holiday","weekday","workingday","weathersit")
for(i in catnames){
  print(i)
  df[,i]=as.factor(df[,i])
}

#Checking datatypes
str(df)


########Checking Missing data#########
sum(is.na(df))

# No missing values are present in the given data set.

########Outlier Analysis###########

num_index = sapply(df, is.numeric)
numeric_data = df[,num_index]
num_cnames = colnames(numeric_data)

for (i in 1:length(num_cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (num_cnames[i]), x = "cnt"), data = subset(df))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=num_cnames[i],x="cnt")+
           ggtitle(paste("Box plot of count for",num_cnames[i])))
}

# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=2)
gridExtra::grid.arrange(gn4,gn5,ncol=2)


# continous variables hum and windspeed  includes outliers


outlier_var=c("hum","windspeed")

#Replace all outliers with NA

for(i in outlier_var){
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  print(length(val))
  df[,i][df[,i] %in% val] = NA
}

# Checking Missing data - after outlier
sum(is.na(df))

# hum includes 2 outliers and windspeed includes 13 outliers, so impute them with mean
df$hum[is.na(df$hum)] <- mean(df$hum, na.rm = TRUE)
df$windspeed[is.na(df$windspeed)] <- mean(df$windspeed, na.rm = TRUE)

# Checking Missing data - after imputation
sum(is.na(df))


############### Visualization ########################
# Scatter plot between temp and cnt
ggplot(data = df, aes_string(x = df$temp, y = df$cnt))+ 
  geom_point()

# Scatter plot between atemp and cnt
ggplot(data = df, aes_string(x = df$atemp, y = df$cnt))+ 
  geom_point()

# Scatter plot between hum and cnt
ggplot(data = df, aes_string(x = df$hum, y = df$cnt))+ 
  geom_point()

# Scatter plot between windspeed and cnt
ggplot(data = df, aes_string(x = df$windspeed, y = df$cnt))+ 
  geom_point()

# Scatter plot between season and cnt
ggplot(data = df, aes_string(x = df$season, y = df$cnt))+ 
  geom_point()

# Scatter plot between month and cnt
ggplot(data = df, aes_string(x = df$mnth, y = df$cnt))+ 
  geom_point()

# Scatter plot between weekday and cnt
ggplot(data = df, aes_string(x = df$weekday, y = df$cnt))+ 
  geom_point()





##################### Feature Selection and Scaling #######################
# generate correlation plot between numeric variables

numeric_index=sapply(df, is.numeric)
corrgram(df[,numeric_index], order=F, upper.panel=panel.pie, 
         text.panel=panel.txt, main="Correlation plot")

# check VIF
library(usdm)
vif(df[,8:10])
# if vif is greater than 10 then variable is not suitable/multicollinerity
#1      temp 61.087196
#2     atemp 61.308023
#3       hum  1.029048

#From heatmap and VIF, Removing variables atemp beacuse it is highly correlated with temp
df = subset(df, select = -c(atemp) )

str(df)

# generate histogram of continous variables
d=density(df$temp)
plot(d,main="distribution")
polygon(d,col="green",border="red")

e=density(df$hum)
plot(e,main="distribution")
polygon(e,col="green",border="red")


f=density(df$windspeed)
plot(f,main="distribution")
polygon(f,col="green",border="red")

#Since the data is distributed normally, scaling isn't required. We can proceed with data splitting 


######################### Model Development ###################

############ Splitting df into train and test ###################
set.seed(101)
install.packages("caret")
split_index = createDataPartition(df$cnt, p = 0.80, list = FALSE) 
train_data = df[split_index,]
test_data = df[-split_index,]


#############  Linear regression Model  #################

lm_model = lm(cnt ~., data=train_data)

# summary of trained model
summary(lm_model)

# prediction on test_data
lm_predictions = predict(lm_model,test_data[,1:10])

regr.eval(test_data[,11],lm_predictions)
#   mae          mse         rmse       mape 
#5.288626e+02 4.837572e+05 6.955265e+02 1.739758e-01

# compute r^2
rss_lm = sum((lm_predictions - test_data$cnt) ^ 2)
tss_lm = sum((test_data$cnt - mean(test_data$cnt)) ^ 2)
rsq_lm = 1 - rss_lm/tss_lm
print(rsq_lm)
#    r^2 = 0.8649468


############## Decision Tree Model ###############

Dt_model = rpart(cnt ~ ., data=train_data, method = "anova")

# summary on trainned model
summary(Dt_model)

#Prediction on test_data
predictions_DT = predict(Dt_model, test_data[,1:10])

regr.eval(test_data[,11], predictions_DT)
#   mae          mse         rmse         mape 
# 7.258519e+02 9.767130e+05 9.882879e+02 2.898936e-01

# compute r^2
rss_dt = sum((predictions_DT - test_data$cnt) ^ 2)
tss_dt = sum((test_data$cnt - mean(test_data$cnt)) ^ 2)
rsq_dt = 1 - rss_dt/tss_dt
print(rsq_dt)
#    r^2 - 0.7273256

#############  Random forest Model #####################
rf_model = randomForest(cnt ~., data=train_data)

# summary on trained model
summary(rf_model)

# prediction of test_data
rf_predictions = predict(rf_model, test_data[,1:10])

regr.eval(test_data[,11], rf_predictions)
#  mae           mse         rmse         mape 
# 5.122251e+02 4.948365e+05 7.034462e+02 1.978135e-01 

# compute r^2
rss_rf = sum((rf_predictions - test_data$cnt) ^ 2)
tss_rf = sum((test_data$cnt - mean(test_data$cnt)) ^ 2)
rsq_rf = 1 - rss_rf/tss_rf
print(rsq_rf)
#    r^2 - 0.8618538

############  XGBOOST Model ###########################
train_data_matrix = as.matrix(sapply(train_data[-11],as.numeric))
test_data_matrix = as.matrix(sapply(test_data[-11],as.numeric))

xgboost_model = xgboost(data = train_data_matrix,label = train_data$cnt, nrounds = 15,verbose = FALSE)

# summary of trained model
summary(xgboost_model)

# prediction on test_data
xgb_predictions = predict(xgboost_model,test_data_matrix)

regr.eval(test_data[,11], xgb_predictions)
#   mae          mse         rmse      mape 
#4.876667e+02 4.252179e+05 6.520873e+02 1.583279e-01

# compute r^2
rss_xgb = sum((xgb_predictions - test_data$cnt) ^ 2)
tss_xgb = sum((test_data$cnt - mean(test_data$cnt)) ^ 2)
rsq_xgb = 1 - rss_xgb/tss_xgb
print(rsq_xgb)
#    r^2 = 0.8812896




# from above models, it is clear that xgboost performs better.


