# ================================================
# Artificial Neural Network using H2O in R
# Dataset: Customer Churn (Churn_Modelling.csv)
# ================================================

# ==== Load Required Libraries ====
library(caTools)
library(h2o)

# ==== Load Dataset ====
dataset <- read.csv('../dataset/churn_modelling.csv')

# Drop irrelevant columns (RowNumber, CustomerId, Surname)
dataset <- dataset[, -c(1, 2, 3)]

# ==== Encode Categorical Variables ====
dataset$Geography <- as.factor(dataset$Geography)
dataset$Gender <- as.factor(dataset$Gender)
dataset$Exited <- as.factor(dataset$Exited)

# ==== Split Dataset ====
set.seed(123)
split <- sample.split(dataset$Exited, SplitRatio = 0.8)
train_set <- subset(dataset, split == TRUE)
test_set  <- subset(dataset, split == FALSE)

# ==== Initialize H2O ====
h2o.init(nthreads = -1)

# Convert data to H2O frames
train_h2o <- as.h2o(train_set)
test_h2o  <- as.h2o(test_set)

# ==== Train ANN Model ====
model <- h2o.deeplearning(
  y = 'Exited',
  training_frame = train_h2o,
  activation = 'Rectifier',
  hidden = c(6, 6),
  epochs = 100,
  train_samples_per_iteration = -2,
  seed = 123,
  standardize = TRUE
)

# ==== Make Predictions ====
predictions <- h2o.predict(model, newdata = test_h2o)
y_pred <- as.vector(predictions$predict)
actual <- as.vector(as.data.frame(test_h2o)$Exited)

# ==== Evaluate Model ====
conf_matrix <- table(Actual = actual, Predicted = y_pred)
print(conf_matrix)

accuracy <- sum(y_pred == actual) / length(actual)
cat("Accuracy:", round(accuracy * 100, 2), "%\n")

# ==== Shutdown H2O ====
h2o.shutdown(prompt = FALSE)
