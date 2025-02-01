library(syuzhet)
library(readtext)
library(RTextTools)
library(e1071)
library(caret)
library(e1071)  # For SVM
library(glmnet) # For GLMNET
library(nnet) 
library(gtrendsR)
library(rtweet)
library(tm)
library(ggplot2)
rm(list=ls())

# Find thread URLs from WallStreetBets
thread_urls <- find_thread_urls(subreddit = "wallstreetbets", sort_by = "top", period = "year")
head(thread_urls$url)
# Remove NA values from thread_urls$url
valid_urls <- na.omit(thread_urls$url)


# Get content (including comments) for the first thread URL
thread_content <- get_thread_content(valid_urls[1])
head(thread_content$threads)
head(thread_content$comments)

# Clean the comments (convert to lowercase, remove punctuation, etc.)
clean_comments <- tolower(thread_content$comments$comment)
clean_comments <- removePunctuation(clean_comments)
clean_comments <- removeNumbers(clean_comments)
clean_comments <- removeWords(clean_comments, stopwords("en"))
clean_comments <- stripWhitespace(clean_comments)
clean_comments <- clean_comments[clean_comments != ""]

# View cleaned comments
head(clean_comments)

# Get sentiment scores for each comment
sentiment_scores <- get_nrc_sentiment(clean_comments)
head(sentiment_scores)
barplot(colSums(sentiment_scores),las = 2,  col = "blue", ylab = 'Count',main = 'Sentiment Scores')

# 1)Bing method of sentiment analysis
bing_vector=get_sentiment(clean_comments,method="bing")
bing_vector #We see the positive/negative of each sentence
afinn_vector=get_sentiment(clean_comments,method="afinn")
afinn_vector
my_text_values=get_sentiment(clean_comments,method="nrc",language="English")
my_text_values

# Create a Document-Term Matrix (DTM)
corpus_clean <- Corpus(VectorSource(clean_comments))
dtm <- DocumentTermMatrix(corpus_clean)
dtm_sparse <- removeSparseTerms(dtm, sparse = 0.95)
dtm_df <- as.data.frame(as.matrix(dtm_sparse))

# Assume 'bing_vector' has a positive/negative sentiment value
dtm_df$sentiment <- ifelse(bing_vector > 0, 1, 0)

# Check for mismatch between bing_vector and dtm_df
if (length(bing_vector) != nrow(dtm_df)) {
  stop("Mismatch between the number of comments and bing_vector!")
}
dtm_df$sentiment <- ifelse(bing_vector > 0, 1, 0)

# View the first few rows of the DTM with sentiment labels
head(dtm_df)

#------------Machine learning models training------------#

# Set seed for reproducibility
set.seed(123)

# Stratified train-test split
train_index <- createDataPartition(dtm_df$sentiment, p = 0.8, list = FALSE)
train_data <- dtm_df[train_index, ]
test_data <- dtm_df[-train_index, ]

# View the first few rows of the train and test data
head(train_data)
head(test_data)


# Remove zero-variance predictors
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
train_data <- train_data[, !nzv$nzv]
test_data <- test_data[, !nzv$nzv]

#------Train a logistic regression model using the train data------#
model <- glm(sentiment ~ ., data = train_data, family = binomial)

# Summary of the model
summary(model)

# Make predictions on the test data
predictions <- predict(model, test_data, type = "response")
predictions <- ifelse(predictions > 0.5, 1, 0)  # Convert probabilities to binary values (0 or 1)

# Evaluate the model performance using a confusion matrix
confusion_matrix <- table(predictions, test_data$sentiment)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(predictions == test_data$sentiment) / length(predictions)
cat("GLM Accuracy: ", accuracy, "\n")


#------SVM model------#

#factor for classification BECAUSE ITS SVM
train_data$sentiment <- as.factor(train_data$sentiment)
test_data$sentiment <- as.factor(test_data$sentiment)

SVM_model <- train(
  sentiment ~ ., 
  data = train_data, 
  method = "svmLinear", 
  preProcess = c("center", "scale"),  # Automatically scale features
  trControl = trainControl(method = "cv", number = 5)  # Cross-validation
)

# Summary of the SVM model
print(SVM_model)

# Make predictions with the SVM model
SVM_predictions <- predict(SVM_model, test_data)

# Evaluate the model performance
SVM_confusion_matrix <- table(SVM_predictions, test_data$sentiment)
print("SVM Confusion Matrix:")
print(SVM_confusion_matrix)

# Calculate SVM accuracy
SVM_accuracy <- sum(SVM_predictions == test_data$sentiment) / nrow(test_data)
cat("SVM Accuracy: ", SVM_accuracy, "\n")


#------NNET Model------#
NNET_model <- nnet(sentiment ~ ., data = train_data, size = 5, decay = 0.1, maxit = 200)  

# Summary of the neural network model
print(NNET_model)

# Make predictions with the neural network model
NNET_predictions <- predict(NNET_model, test_data, type = "class")

# Evaluate the model performance
NNET_confusion_matrix <- table(NNET_predictions, test_data$sentiment)
print("NNET Confusion Matrix:")
print(NNET_confusion_matrix)

# Calculate NNET accuracy
NNET_accuracy <- sum(NNET_predictions == test_data$sentiment) / nrow(test_data)
cat("NNET Accuracy: ", NNET_accuracy, "\n")


cat("Comparison of Model Accuracies:\n", "Logistic Regression Accuracy: ", accuracy, "\n","Neural Network Accuracy: ", NNET_accuracy, "\n","Logistic Regression Accuracy: ", accuracy, "\n")


# ---- Add New Data Prediction Workflow ---- #

# 1. Prepare New Data
# Fetch new comments from WallStreetBets or any other source
# Fetch URLs of the most recent threads
thread_urls <- find_thread_urls(subreddit = "wallstreetbets", sort_by = "top", period = "day")

# Fetch content (comments) from the first thread
thread_content <- get_thread_content(thread_urls$url[1])

# Extract the comments
new_comments <- thread_content$comments$comment

# Clean the new comments
new_clean_comments <- tolower(new_comments)
new_clean_comments <- removePunctuation(new_clean_comments)
new_clean_comments <- removeNumbers(new_clean_comments)
new_clean_comments <- removeWords(new_clean_comments, stopwords("en"))
new_clean_comments <- stripWhitespace(new_clean_comments)
new_clean_comments <- new_clean_comments[new_clean_comments != ""]
# Create a DTM for new comments
new_corpus <- Corpus(VectorSource(new_clean_comments))
new_dtm <- DocumentTermMatrix(new_corpus)

# Align the new DTM with the training data's features
common_terms <- intersect(colnames(train_data), colnames(as.matrix(new_dtm)))
new_dtm_aligned <- as.matrix(new_dtm)[, common_terms, drop = FALSE]

# Convert to DataFrame for predictions
new_dtm_df <- as.data.frame(new_dtm_aligned)

# Add missing columns with 0 values
missing_cols <- setdiff(colnames(train_data), colnames(new_dtm_df))
for (col in missing_cols) {
  new_dtm_df[[col]] <- 0
}
new_dtm_df <- new_dtm_df[, colnames(train_data)[-ncol(train_data)]]  # Align column order

# Predict sentiment using GLM
glm_predictions <- predict(model, new_dtm_df, type = "response")
glm_predictions <- ifelse(glm_predictions > 0.5, 1, 0)

# Predict sentiment using SVM
svm_predictions <- predict(SVM_model, new_dtm_df)

# Predict sentiment using Neural Network
nnet_predictions <- predict(NNET_model, new_dtm_df, type = "class")
summary_results <- data.frame(
  GLM_Positive = mean(glm_predictions == 1),
  SVM_Positive = mean(svm_predictions == 1),
  NNET_Positive = mean(nnet_predictions == 1)
)
print("Proportion of Positive Sentiments:")
print(summary_results)

sentiment_counts <- data.frame(
  Model = c("GLM", "SVM", "NNET"),
  Positive = c(
    mean(glm_predictions == 1),
    mean(svm_predictions == 1),
    mean(nnet_predictions == 1)
  )
)

ggplot(sentiment_counts, aes(x = Model, y = Positive, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Proportion of Positive Sentiment Predictions",
    x = "Model",
    y = "Proportion of Positive Sentiments"
  ) +
  theme_minimal()
prediction_results <- data.frame(
  Comment = new_clean_comments,
  GLM_Prediction = glm_predictions,
  SVM_Prediction = svm_predictions,
  NNET_Prediction = nnet_predictions
)
print("Predictions for New Comments:")
print(prediction_results)
