# Prediction of Barbell Lift Quality
ksplett1  
Sunday, March 22, 2015  
<hr />
## Practical Machine Learning Project (Coursera predmachlearn-012)##
<hr />
### Summary ###

This project predicts quality of barbell lifts for Unilateral Dumbbell Biceps Curl. The data set and related research can be found at http://groupware.les.inf.puc-rio.br/har (section on Weight Lifting Exercise). As described in the "Qualitative Activity Recognition of Weight Lifting Exercises" paper, users were recorded while they performed the same activity correctly and incorrectly based on a set of common mistakes. Wearable sensors (belt, glove, arm-band, and dumbbell) provided measurements of each mistake. 

In this project, 5 random forests are used as a classification model, based on 500 trees. 

Random Forest was selected as the classification model for the following reasons:
* Classification trees can predict non-linear models
* Predictor variables do not require normalization and the trees automatically rank variable significance
* Random Forest models are top-performing models
* Random Forest models can handle many predictor variables

Cross validation was done by partitioning the training data set into separate sub-training and validation sets. The random forest model was trained on the sub-training set, and then the model prediction was run on the validation set. The predicted classe output was compared to the actual classe output in the validation set to produce a confusion matrix. The accuracy percent was also calculated. Cross-validation checks for Type III or out-of-sample errors (errors suggested by the data - i.e., over-fitting)estimated a 98.6% out-of-sample accuracy.

note: Because the training set is large, the training set selected to train the model was 30% (4907 observation and  54 variables) of the original training set size. The decision to reduce the training set size was made because the model took about 15 minutes to run on my laptop. The resulting model is highly accurate (estimated 98.6% out-of-sample accuracy), so further iterations of the model were deemed unnecessary.

A variable reduction method could also have been applied to decrease the model training time and remove lesser  significant variables. A single classification tree was initially run to identify significant variables rapidly; however, this approach was not chosen as the final model because it would have required more development time to find the best reduced set of predictor variables.

Another possible model approach (not chosen) would be to re-summarize the raw device measurements (ex. mean, std deviation, total acceleration (sqrt(x^2+y^2+z^2), max+std), which seems to be a common way to analyze accelerometer data. Again, this model would have also required more development time. 

### Data Processing ###

Data Set Description:

* 19,621 observations collected from six participants. 160 variables.
* .csv file format
* Measurements are taken from accelerometers on the belt, forearm, arm, and dumbbell
* classe output variable: Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes (throw elbows to the front (Class B), lift dumbbell only halfway (Class C), lower  dumbbell only halfway (Class D) and throw hips to the front (Class E)).
* Training data is sorted by class output
* Training data is sequentially timestamped within a time window (indicated by new_window)
* Summarized measurements (kurtosis, skewness, variance, average, standard deviation, 
	minimum, maximum) are calculated once per time window

<hr />



```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.1
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.3
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```


```r
# assume setwd done before running this program
setwd("C:/Users/ksplett1/machinelearning")
setwd("data")
df <- read.csv("pml-training.csv", header = TRUE, sep = ",", na.strings = "NA" )
df_test <- read.csv("pml-testing.csv", header = TRUE, sep = ",", na.strings = "NA" )
setwd("../")

### Data Transformations ###

# Remove window summary columns that start with new_window, amplitude, kurtosis, skewness, var, avg, stddev, min, max
df_obs <- df[ , -grep("^(new_window|amplitude|kurtosis|skewness|var|avg|stddev|min|max)", names(df)) ]
df_tst <- df_test[ , -grep("^(new_window|amplitude|kurtosis|skewness|var|avg|stddev|min|max)", names(df_test)) ]

# Not performing sequence or time series analysis - remove timestamp columns
# remove raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, X, user_name, num_window
df_obs <- df_obs[ , -grep("^(raw_timestamp_part_1|raw_timestamp_part_2|cvtd_timestamp|X|user_name)", names(df_obs)) ]
df_tst <- df_tst[ , -grep("^(raw_timestamp_part_1|raw_timestamp_part_2|cvtd_timestamp|X|user_name)", names(df_tst)) ]

# Scale data set down to a runnable size
InTrain <- createDataPartition(y=df_obs$classe, p=0.25, list=FALSE)
training <- df_obs[InTrain,]
validating <- df_obs[-InTrain,]

dim(training)
```

```
## [1] 4907   54
```

```r
### Model Build ###

# Classification model using Random Forest. 
#  5 fold cross validation
set.seed(100)
rfModFit <-train(classe ~., data=training, method="rf", 
           trControl=trainControl(method="cv", number=5), prox=TRUE ,allowParallel=TRUE)

print(rfModFit)							
```

```
## Random Forest 
## 
## 4907 samples
##   53 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 3924, 3926, 3926, 3925, 3927 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa   Accuracy SD  Kappa SD
##    2    0.9749    0.9683  0.005410     0.006855
##   27    0.9849    0.9809  0.001688     0.002135
##   53    0.9851    0.9812  0.002460     0.003108
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 53.
```

```r
print(rfModFit$finalModel)		
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE,      allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 53
## 
##         OOB estimate of  error rate: 1.16%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1393   1   0   1   0    0.001434
## B   14 930   5   1   0    0.021053
## C    1  13 841   1   0    0.017523
## D    0   2   8 794   0    0.012438
## E    0   2   1   7 892    0.011086
```

<br />
<hr />
###  Prediction Results ###
<br />


```r
# Prediction output for Test set
# note: All test set outputs were predicted correctly 
predModFit <- predict(rfModFit, df_tst)

# Prediction output for Validation set
predModFit <- predict(rfModFit, validating)

# Confusion Matrix
table( predModFit, validating$classe )
```

```
##           
## predModFit    A    B    C    D    E
##          A 4181   57    0    0    0
##          B    3 2734   41    6   17
##          C    0   49 2517   19    1
##          D    0    7    8 2384   23
##          E    1    0    0    3 2664
```

Validation accuracy = 0.984



