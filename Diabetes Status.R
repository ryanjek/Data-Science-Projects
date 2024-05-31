setwd("~/Documents/NUS/Y1S1/DSA1101/Assignment")
data <- read.csv("diabetes_5050.csv")
set.seed(1101)
head(data)
attach(data)

Diabetes_binary = as.factor(Diabetes_binary)
## Checking for association between response variable and input variables ##

# Association between the quantitative input variable and categorical response variable

boxplot(BMI~Diabetes_binary)
boxplot(GenHlth~Diabetes_binary)
boxplot(MentHlth~Diabetes_binary)
boxplot(PhysHlth~Diabetes_binary)
boxplot(Age~Diabetes_binary)
boxplot(Education~Diabetes_binary)
boxplot(Income~Diabetes_binary)

# BMI: Significant
# The median BMI value and interquartile range of a diabetic person is slightly larger than a person without diabetes therefore a person with diabetes will be associated with a higher BMI. Hence BMI affect diabetic status.

# GenHlth:Significant
# The median general health value and interquartile range of a diabetic person is larger than a person without diabetes therefore a person with diabetes will be associated with a poor general health.Hence general health affect diabetic status.

# MentHlth: Insignificant
# The median values and interquartile range are similiar, this means that mental health does not affect the diabetic status.

# PhysHlth:Significant
# The median for physical health and interquartile range of a diabetic person is larger than a person without diabetes therefore a person with diabetes will be associated with a poorer physcial health. Hence physical health affect diabetic status.

# Age: Significant
#The median age of a diabetic person is larger than a person without diabetes, a diabetic person has a smaller interquartile range for age than a non diabetic person.Therefore a person with diabetes will be associated with being older.Hence age affect diabetic status.

# Education: Insignificant
# The median values and interquartile range are similiar, this means that there is no obvious difference between education level. Hence education does not affect diabetic status.

# Income:Significant 
# The median income of a diabetic person is lower than a person without diabetes, the interquartile range for income for a diabetic person is larger than a non diabetic person therefore a person with diabetes will be associated with having a lower income. Hence income affect diabetic status.


# Creating a new data set for catergorical input
drops <- c("BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education","Income")
cat_data <- data [,!(names(data) %in% drops )]
head(cat_data)

# Finding odds_ratio
corr = numeric(length(2:ncol(cat_data)))
for (i in 2:ncol(cat_data)){
  table_data = table(cat_data[,i],Diabetes_binary)
  odds_ratio <- (table_data[1, 1] / table_data[1, 2]) / (table_data[2, 1] / table_data[2, 2])
  corr[i-1] <- odds_ratio
}
# Odd Ratio of each categorical input
# HighBP:5.088 (Significant)
# HighChol:3.296 (Significant)
# CholCheck:6.491 (Significant)
# Smoker:1.412 (Insignificant)
# Stroke:3.093 (Significant)
# HeartDiseaseorAttack:3.656 (Significant)
# PhysActivity:0.494 (Insignificant)
# Fruits:0.801 (Insignificant)
# Veggies:0.676 (Insignificant)
# HvyAlcoholConsump:0.365 (Significant)
# AnyHealthcare:1.252 (Insignificant)
# NoDocbcCost:1.326 (Insignificant)
# DiffWalk: 3.807 (Significant)
# Sex:1.195 (Insignificant)

# Dropping of insig input variables
drop_insig <- c("Smoker","PhysActivity","Fruits", "Veggies", "AnyHealthcare", "NoDocbcCost", "Sex", "MentHlth","Education")
sig_data <- data [,!(names(data) %in% drop_insig )]

## Separating Data into 2 sets while keeping 50-50 ratio of diabetes and non diabetes ##
dim(sig_data)
attach(sig_data)
table(Diabetes_binary)

# Diabetes_binary
# 0     1 
# 35346 35346 

n_folds=5
folds_j_0 <- sample(rep(1:n_folds, length.out = 35346 ))  # for non diabetes
folds_j_1 <- sample(rep(1:n_folds, length.out = 35346 ))  # for diabetes

data1 = sig_data[Diabetes_binary == 0,] # data for non diabetes
data2 = sig_data[Diabetes_binary == 1,] # data for diabetes

##  Using KNN ##
library(class)

x = sig_data[,c("HighBP","HighChol","CholCheck","BMI","Stroke","HeartDiseaseorAttack","HvyAlcoholConsump","GenHlth","PhysHlth", "DiffWalk","Age","Income")]
y = sig_data[,c("Diabetes_binary")]

# In the case of diabetes, FPR can be tolerated and hence not considered.
# FNR is more severe.

acc = numeric(n_folds) 
pre = numeric(n_folds)
FNR = numeric(n_folds)

ave_acc = numeric(n_folds) 
ave_pre = numeric(n_folds)
ave_FNR = numeric(n_folds)


j= 1
for (i in 1:10){ 
  for (j in 1:n_folds) {
  
    test1 <- which(folds_j_0 == j)
    test2 <- which(folds_j_1 == j)
  
    train.1=data1[ -test1, ]
    train.2=data2[ -test2, ]
  
    train = rbind(train.1, train.2) # this is the training data set
  
    test = rbind(data1[test1,], data2[test2,]) # test data 
    
    train.x = train[,c("HighBP","HighChol","CholCheck","BMI","Stroke","HeartDiseaseorAttack","HvyAlcoholConsump","GenHlth","PhysHlth", "DiffWalk","Age","Income")]
    test.x = test[,c("HighBP","HighChol","CholCheck","BMI","Stroke","HeartDiseaseorAttack","HvyAlcoholConsump","GenHlth","PhysHlth", "DiffWalk","Age","Income")]
    train.y = train[,"Diabetes_binary"]
    test.y = test[,"Diabetes_binary"]
    
    pred <- knn(train.x, test.x, train.y, k=i)
    
    confusion.matrix <- table(pred,test.y)
    
    acc[j]= (confusion.matrix[1,1]+confusion.matrix[2,2])/ (sum(confusion.matrix[1,])+sum(confusion.matrix[2,]))
    pre[j]= confusion.matrix[2,2]/ sum(confusion.matrix[,2])
    FNR[j]= confusion.matrix[2,1]/ sum(confusion.matrix[2,])
  
  }
  # To store the average accuracy, precision and FNR values for that e.g k = 1 after N folds
  ave_acc[i] = mean(acc[j])
  ave_pre[i] = mean(pre[j])
  ave_FNR[i] = mean(FNR[j])}
# To store the average accuracy, precision and FNR for all the k values
  find_k = cbind(ave_acc,ave_pre,ave_FNR)
  #       ave_acc   ave_pre   ave_FNR
  # [1,]  0.685     0.701     0.320
  # [2,]  0.697     0.717     0.311
  # [3,]  0.711     0.740     0.300
  # [4,]  0.717     0.749     0.296
  # [5,]  0.722     0.759     0.293
  # [6,]  0.725     0.763     0.291
  # [7,]  0.727     0.768     0.291
  # [8,]  0.726     0.768     0.291
  # [9,]  0.7303    0.7722    0.2875
  # [10,] 0.7297    0.7722    0.2884
  
  # Best K = 9, with the lowest FNR

 ##  Using Decision Tree ##
library(rpart) 
library(rpart.plot)

acc_dt = numeric(n_folds) 
pre_dt = numeric(n_folds)
FNR_dt = numeric(n_folds)

ave_acc_dt = numeric(n_folds) 
ave_pre_dt = numeric(n_folds)
ave_FNR_dt = numeric(n_folds)

a = 1
  
for (a in 1:n_folds) {
  test1_dt <- which(folds_j_0 == a)
  test2_dt <- which(folds_j_1 == a)
      
  train.1_dt =data1[ -test1_dt, ]
  train.2_dt =data2[ -test2_dt, ]
      
  train_dt = rbind(train.1_dt, train.2_dt) # this is the training data set
      
  test_dt = rbind(data1[test1_dt,], data2[test2_dt,]) # test data 
  
  fit.diabetes_dt <- rpart(Diabetes_binary ~ .,
                        method = "class", data = train_dt, control = rpart.control( minsplit =7000),
                        parms = list( split ='gini'))
      
      
  pred_diabetes_dt = predict(fit.diabetes_dt, newdata = test_dt[,2:13], type = 'class')
      
  confusion.matrix_dt = table(pred_diabetes_dt,test_dt[,1])
      
  acc_dt[a] = sum(diag(confusion.matrix_dt))/sum(confusion.matrix_dt)
  pre_dt[a]= confusion.matrix_dt[2,2]/ sum(confusion.matrix_dt[,2])
  FNR_dt[a]= confusion.matrix_dt[2,1]/ sum(confusion.matrix_dt[2,])
}

ave_acc_dt = mean(acc_dt)
ave_pre_dt = mean(pre_dt)
ave_FNR_dt = mean(FNR_dt)
diagnostic_dt = cbind(ave_acc_dt,ave_pre_dt,ave_FNR_dt)
#    ave_acc_dt ave_pre_dt ave_FNR_dt
#    0.726      0.800      0.303

##  Using Naive Bayes ##
library(e1071)

acc_nb = numeric(n_folds) 
pre_nb = numeric(n_folds)
FNR_nb = numeric(n_folds)

ave_acc_nb = numeric(n_folds) 
ave_pre_nb = numeric(n_folds)
ave_FNR_nb = numeric(n_folds)

b = 1

for (b in 1:n_folds) {
  test1_nb <- which(folds_j_0 == b)
  test2_nb <- which(folds_j_1 == b)
  
  train.1_nb=data1[ -test1_nb, ]
  train.2_nb=data2[ -test2_nb, ]
  
  train_nb = rbind(train.1_nb, train.2_nb) # this is the training data set
  
  test_nb = rbind(data1[test1_nb,], data2[test2_nb,]) # test data 
  
  nb_model <- naiveBayes( Diabetes_binary ~., data = train_nb)
  
  
  pred_diabetes_nb = predict(nb_model, newdata = test_nb[,2:13], type = 'class')
  
  confusion.matrix_nb = table(pred_diabetes_nb,test_nb[,1])
  
  acc_nb[b] = sum(diag(confusion.matrix_nb))/sum(confusion.matrix_nb)
  pre_nb[b]= confusion.matrix_nb[2,2]/ sum(confusion.matrix_nb[,2])
  FNR_nb[b]= confusion.matrix_nb[2,1]/ sum(confusion.matrix_nb[2,])
}
ave_acc_nb = mean(acc_nb)
ave_pre_nb = mean(pre_nb)
ave_FNR_nb = mean(FNR_nb)
diagnostic_nb = cbind(ave_acc_nb,ave_pre_nb,ave_FNR_nb)
# ave_acc_nb ave_pre_nb ave_FNR_nb
# 0.730     0.734       0.272


##  Using Logistic Regression ##
library(ROCR)

# Calling categorical input variable as categorical
HighBP = as.factor(HighBP)
HighChol = as.factor(HighChol)
CholCheck = as.factor(CholCheck)
Smoker = as.factor(Smoker)
Stroke = as.factor(Stroke)
HeartDiseaseorAttack = as.factor(HeartDiseaseorAttack)
PhysActivity = as.factor(PhysActivity)
HvyAlcoholConsump = as.factor(HvyAlcoholConsump)
DiffWalk = as.factor(DiffWalk)

#Checking if input variables are significant before proceeding.
M1_LR<- glm( Diabetes_binary ~., data =sig_data,family = binomial)
summary(M1_LR)
# Coefficients:
# Estimate Std. Error z value Pr(>|z|)    
# (Intercept)          -7.022312   0.108529 -64.704  < 2e-16 ***
#  HighBP                0.748379   0.019680  38.027  < 2e-16 ***
#  HighChol              0.583299   0.018771  31.074  < 2e-16 ***
#  CholCheck             1.343072   0.080884  16.605  < 2e-16 ***
#  BMI                   0.076177   0.001566  48.642  < 2e-16 ***
#  Stroke                0.162703   0.040832   3.985 6.76e-05 ***
#  HeartDiseaseorAttack  0.300729   0.028159  10.680  < 2e-16 ***
#  HvyAlcoholConsump    -0.737126   0.048218 -15.287  < 2e-16 ***
#  GenHlth               0.591720   0.011252  52.586  < 2e-16 ***
#  PhysHlth             -0.009583   0.001152  -8.317  < 2e-16 ***
#  DiffWalk              0.096492   0.025553   3.776 0.000159 ***
#  Age                   0.153456   0.003755  40.869  < 2e-16 ***
#  Income               -0.055056   0.004612 -11.937  < 2e-16 ***
# Since all the Pvalue for all the inputs are less than threshold 0.5 they are significant.

# Selecting Threshold to determine classification

prob_lr = predict(M1_LR, type ="response")

pred_lr = prediction(prob_lr , Diabetes_binary )
roc_lr = performance(pred_lr , "tpr", "fpr")

# Extracting the Threshold(alpha), FPR , and TPR values from the ROC
alpha_lr <- round (as.numeric(unlist(roc_lr@alpha.values)) ,4)
length(alpha_lr) 
fpr_lr <- round(as.numeric(unlist(roc_lr@x.values)) ,4)
tpr_lr <- round(as.numeric(unlist(roc_lr@y.values)) ,4)

x = cbind(alpha_lr,tpr_lr,fpr_lr)
#To find the best suited TPR and FPR.

# adjust margins and plot TPR and FPR
par(mar = c(5 ,5 ,2 ,5))

plot(alpha_lr ,tpr_lr , xlab ="Threshold", xlim =c(0 ,1) ,
     ylab = "True positive rate ", type ="l", col = "blue")
par( new ="True")
plot(alpha_lr ,fpr_lr , xlab ="", ylab ="", axes =F, xlim =c(0 ,1) , type ="l", col = "red" )
axis( side =4) # to create an axis at the 4th side
mtext(side =4, line =3, "False positive rate")
text(0.18 ,0.18 , "FPR")
text(0.58 ,0.58 , "TPR")
# From the graph, optimal threshold should be 0.4


acc_lr = numeric(n_folds) 
pre_lr = numeric(n_folds)
FNR_lr = numeric(n_folds)

ave_acc_lr = numeric(n_folds) 
ave_pre_lr = numeric(n_folds)
ave_FNR_lr = numeric(n_folds)

c = 1

for (c in 1:n_folds) {
  test1_lr <- which(folds_j_0 == c)
  test2_lr <- which(folds_j_1 == c)
  
  train.1_lr=data1[ -test1_lr, ]
  train.2_lr=data2[ -test2_lr, ]
  
  train_lr = rbind(train.1_lr, train.2_lr) # this is the training data set
  
  test_lr = rbind(data1[test1_lr,], data2[test2_lr,]) # test data 
  
  M2_LR<- glm( Diabetes_binary ~., data = train_lr,family = binomial)
  
  pred_diabetes_lr = predict(M2_LR, newdata = test_lr[,2:13], type = 'response')
  
  lr_class = ifelse(pred_diabetes_lr > 0.4, "1", "0")
  
  confusion.matrix_lr = table(lr_class,test_lr[,1])
  
  acc_lr[c] = sum(diag(confusion.matrix_lr))/sum(confusion.matrix_lr)
  pre_lr[c]= confusion.matrix_lr[2,2]/ sum(confusion.matrix_lr[,2])
  FNR_lr[c]= confusion.matrix_lr[2,1]/ sum(confusion.matrix_lr[2,])
}
ave_acc_lr = mean(acc_lr)
ave_pre_lr = mean(pre_lr)
ave_FNR_lr = mean(FNR_lr)
diagnostic_lr = cbind(ave_acc_lr,ave_pre_lr,ave_FNR_lr)

# ave_acc_lr ave_pre_lr ave_FNR_lr
# 0.744      0.854      0.300


## Comparing between top 2 classifier ##

# ROC Curve and AUC value for Logistic Regresison
prob_lr = predict(M1_LR, type ="response")
pred_lr = prediction(prob_lr , Diabetes_binary )
roc_lr = performance(pred_lr, measure="tpr", x.measure="fpr")
plot(roc_lr, col = "red")

auc_lr = performance(pred_lr, measure ="auc")
auc_lr@y.values[[1]]
# auc_lr: 0.824


# ROC Curve and AUC value for Naive Bayes #

nb_model_roc <- naiveBayes( Diabetes_binary ~., data = sig_data)
prob_nb = predict(nb_model, newdata = sig_data[,2:13], type = 'raw')
prob_nb <- prob_nb[, 2]
pred_nb <- prediction(prob_nb, Diabetes_binary)
roc_nb = performance(pred_nb, measure="tpr", x.measure="fpr")
plot(roc_nb, add = TRUE, col = "blue")


auc_nb <- performance(pred_nb , "auc")@y.values[[1]]
auc_nb
# auc_nb:0.790

legend("bottomright", c("logistic regression","naive Bayes"),col=c("red","blue"), lty=1)

# Since AUC and ROC curve of LR is greater than NB, LR is the better classifier.