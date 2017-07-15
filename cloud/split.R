
# Ruta del csv
PATH <- "B:/Descargas/creditcard.csv"

# Leemos el csv y hacemos undersampling con ratio 50-50
data <- read.csv(PATH)
df <- data[which(data$Class == 1),]
df$Class <- "fraud"
df2 <- data[which(data$Class == 0),]
df2 <- df2[sample(nrow(df2), nrow(df)), ]
df2$Class <- "notfraud"
result <- rbind(df,df2)
result <- result[sample(nrow(result), nrow(result)), ]

# Porcentaje para los conjuntos de entrenamiento y test (80% y 20% respectivamente)
TRAIN_SIZE <- 0.8
tr <- as.integer(nrow(result)*TRAIN_SIZE)

# Obtenemos los conjuntos de entrenamiento y test
train <- result[1:tr,]
test <- result[(tr+1):nrow(result),]


# Rutas para guardar los csv resultantes
TRAIN_PATH <- "B:/Descargas/creditcard.data.csv"
TEST_PATH <- "B:/Descargas/creditcard.test.csv"

# Guardamos los csv
write.table(train, TRAIN_PATH,row.names = FALSE, col.names = FALSE, sep=",")
write.table(test, TEST_PATH, row.names = FALSE, col.names = FALSE, sep=",")
