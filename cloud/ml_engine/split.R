# Load 'rio' package to import (and export) datasets
if (!require(rio)) {
  install.packages("rio")
}

# This script undersamples and splits a dataset into training and validation sets
# in order to use them afterwards in ML Engine

# CSV Path
PATH <- "B:/Descargas/creditcard.csv"

# We read the csv and then undersampling is performed
data <- import(PATH)

# In this case strings are used as labels since we can easily adapt the Census sample for ML Engine
df <- data[which(data$Class == 1),]
df$Class <- "fraud"

df2 <- data[which(data$Class == 0),]
df2 <- df2[sample(nrow(df2), nrow(df)), ]
df2$Class <- "notfraud"

# Bind both dataframes and shuffle them
result <- rbind(df,df2)
result <- result[sample(nrow(result), nrow(result)), ]

# Training and validation percentages (80% and 20% resp.)
TRAIN_SIZE <- 0.8
tr <- as.integer(nrow(result)*TRAIN_SIZE)

train <- result[1:tr,]
test <- result[(tr+1):nrow(result),]


# Paths used to export the CSV files
TRAIN_PATH <- "B:/Descargas/creditcard.data.csv"
TEST_PATH <- "B:/Descargas/creditcard.test.csv"

write.table(train, TRAIN_PATH,row.names = FALSE, col.names = FALSE, sep=",")
write.table(test, TEST_PATH, row.names = FALSE, col.names = FALSE, sep=",")

# Another possibility using 'rio' package
# export(train, TRAIN_PATH)
# export(test, TEST_PATH)