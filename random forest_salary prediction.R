rm(list = ls())#clearing the environment history
setwd("C:/Users/Rakim/Desktop/sylvia/R")#setting the working directory
getwd()#getting the working diretory

# Load necessary libraries
library(tidyverse)  # For data manipulation
library(caret)  # For machine learning utilities
library(randomForest)  # For Random Forest model
library(e1071)  # For SVM model

# Load the dataset
df <- read.csv("jobs_in_data.csv")  # Read CSV file into a dataframe

# Explore the dataset
dim(df)#dimension of the dataset
str(df)  # Check the structure of the data
summary(df)  # Summary statistics of numerical columns

# Select relevant columns & drop unnecessary ones
df <- df %>% select(-salary_currency, -salary)  # Removing currency column as it's not needed

# Convert categorical variables to factors
# Factors are needed for categorical variables to be properly handled by ML models
df <- df %>% 
  mutate(across(where(is.character), as.factor))  # Convert categorical variables to factors

# Check for missing values
colSums(is.na(df))  # Count missing values per column

# Create dummy variable transformation **before** splitting
dummies <- dummyVars(~ ., data = df)
df_numeric <- predict(dummies, df) %>% as.data.frame()


# Split into training & testing sets
set.seed(123)  # Set seed for reproducibility
trainIndex <- createDataPartition(df_numeric$salary_in_usd, p = 0.8, list = FALSE)  # Create index for 80% training data

train_data <- df_numeric[trainIndex, ]  # Training dataset
dim(train_data)

test_data <- df_numeric[-trainIndex, ]  # Testing dataset
dim(test_data)

# Ensure both datasets have the same structure
missing_cols <- setdiff(names(train_data), names(test_data))
for (col in missing_cols) {
  test_data[[col]] <- 0  # Add missing columns with default value 0
}
missing_cols

# ---- REGRESSION: RANDOM FOREST MODEL ----
set.seed(123)
rf_model <- randomForest(salary_in_usd ~ ., data = train_data, ntree = 100, importance = TRUE)  # Train Random Forest model
rf_predictions <- predict(rf_model, newdata = test_data)  # Predictions

#evaluate model perfomane
rf_rmse <- sqrt(mean((rf_predictions - test_data$salary_in_usd)^2))  # Compute RMSE
rf_rsq <- cor(rf_predictions, test_data$salary_in_usd)^2  # Compute R-squared

print(paste("Random Forest Regression RMSE:", round(rf_rmse, 2)))
print(paste("Random Forest Regression R-squared:", round(rf_rsq, 2)))
