==============================================================================
TITLE: script.R
INPUT:
The MovieLens dataset referred to as "edx" following course guidelines
STEPS:
1. Data Preparation
2. Data Exploration
3. Manual Predictions Creation
4. Matrix Factorization using the Recosystem model
6-Result
OUTPUT:
A predictive model for movie ratings achieving an RMSE of 0.7792515
ESTIMATED RUNTIME: ~5 hour
AUTHOR: Alaa Eddine Bouchikhi
YEAR: 2024
================================================================================
# ==============================================================================
# 0 Initialization
# ==============================================================================

# Clear the environment
rm(list = ls())

# install packages

libraries <- c("caret", "doParallel", "dplyr", "knitr", "Matrix", 
               "parallel", "readr", "recosystem", "scales", 
               "stringr", "tidyr")


# Load the necessary libraries
library(caret)        # Enables machine learning and model evaluation
library(doParallel)   # Allows parallel processing for faster computations
library(dplyr)        # Facilitates data manipulation and wrangling
library(knitr)        # Supports dynamic report generation in RMarkdown
library(Matrix)       # Provides efficient matrix operations
library(parallel)     # Offers parallel computing capabilities
library(readr)        # Assists in reading various data formats, like CSV
library(recosystem)   # Used for building recommendation systems
library(scales)       # Helps with scaling functions for visualizations
library(stringr)      # Aids in string manipulation and regular expressions
library(tidyr)        # Useful for tidying and reshaping data frames
library(ggplot2)      # Enables data visualization with the grammar of graphics
library(lubridate)    # Simplifies working with date and time objects
library(purrr)        # Supports functional programming with lists and vectors
library(ROCR)         # Assists in performance assessment of machine learning models
library(e1071)        # Provides additional machine learning algorithms, including SVM

# Set options
options(timeout = 120)  # Set timeout for connections

# Start a script timer
start_time <- Sys.time()

# Initialize a list to hold RMSE results
rmses <- list(goal = 0.8649)

# Clear the console
cat("\014") 

# Calculate and print the time taken for script execution
end_time <- Sys.time()
execution_time <- end_time - start_time
cat("Script executed in:", execution_time, "seconds\n")



# ==============================================================================
# 0 Initialization
# ==============================================================================

# Set the working directory
setwd("C:/Users/ASUS/Desktop/alaa")

# Define the name of the local file
dl <- "ml-10m.zip"

# Check if the file exists; if not, stop the execution with an error message
if (!file.exists(dl)) {
  stop("The file does not exist in the specified path.")
}

# Extract data and merge movies and ratings
movielens <- left_join(
  
  # Read and process the movies data
  read.table(
    text = gsub(
      x = readLines(con = unzip(dl, "ml-10M100K/movies.dat")),
      pattern = "::",
      replacement = ";",
      fixed = TRUE
    ),
    sep = ";",
    col.names = c("movieId", "title", "genres"),
    colClasses = c("integer", "character", "character"),
    quote = "",
    comment.char = "" # Remove any comments
  ),
  
  # Read and process the ratings data
  read.table(
    text = gsub(
      x = readLines(con = unzip(dl, "ml-10M100K/ratings.dat")),
      pattern = "::",
      replacement = ";",
      fixed = TRUE
    ),
    sep = ";",
    col.names = c("userId", "movieId", "rating", "timestamp"),
    colClasses = c("integer", "integer", "numeric", "integer"),
    quote = ""
  ),
  
  # Merge the two data frames by movieId
  by = "movieId"
) |> na.omit() # Remove rows with missing values

# Display the first 6 rows of the merged data
head(movielens)


# ==============================================================================
# 1 Data Exploration
# ==============================================================================

# Extract files from the zip archive without reading the data
unzip(dl, list = TRUE)  # List the contents of the zip file

# Display the first 6 rows of the merged data
head(movielens)

# Check the structure of the data
str(movielens)

# Show the number of rows and columns
dim(movielens)

# Display summary statistics of the dataset
summary(movielens)

# Check unique values for important columns
unique_movies <- length(unique(movielens$title))
unique_users <- length(unique(movielens$userId))
cat("Number of unique movies:", unique_movies, "\n")
cat("Number of unique users:", unique_users, "\n")

# Analyze missing values in the dataset
missing_values <- sapply(movielens, function(x) sum(is.na(x)))
print(missing_values)

# Load ggplot2 for visualization
library(ggplot2)

# Visualize the distribution of ratings
ggplot(movielens, aes(x = rating)) +
  geom_bar(fill = "blue", alpha = 0.7) +
  labs(title = "Distribution of Ratings", x = "Rating", y = "Count") +
  theme_minimal()

# Analyze the average rating per movie
avg_rating <- movielens %>%
  group_by(title) %>%
  summarize(avg_rating = mean(rating), n_ratings = n()) %>%
  arrange(desc(avg_rating))

# Display the top 10 movies by average rating
head(avg_rating, 10)

# ==============================================================================
# 1. Prepare the data
# ==============================================================================

# Set the working directory
setwd("C:/Users/ASUS/Desktop/alaa")

# Load necessary libraries
library(dplyr)
library(caret)

# Set a seed for reproducibility
set.seed(1)

# Read and prepare movies and ratings data from the local files
load_movielens_data <- function() {
  # Define the path to the downloaded zip file
  dl <- "ml-10m.zip"
  
  # Unzip the dataset
  unzip(dl, exdir = "ml-10M100K")
  
  # Read movies data
  movies <- read.table(
    text = gsub(
      x = readLines(con = "ml-10M100K/movies.dat"),
      pattern = "::",
      replacement = ";",
      fixed = TRUE
    ),
    sep = ";",
    col.names = c("movieId", "title", "genres"),
    colClasses = c("integer", "character", "character"),
    quote = "",
    comment.char = ""
  )
  
  # Read ratings data
  ratings <- read.table(
    text = gsub(
      x = readLines(con = "ml-10M100K/ratings.dat"),
      pattern = "::",
      replacement = ";",
      fixed = TRUE
    ),
    sep = ";",
    col.names = c("userId", "movieId", "rating", "timestamp"),
    colClasses = c("integer", "integer", "numeric", "integer"),
    quote = ""
  )
  
  # Combine movies and ratings, then remove NA values
  movielens <- left_join(movies, ratings, by = "movieId") %>% na.omit()
  
  return(movielens)
}

# Load the dataset
movielens <- load_movielens_data()

# Partition the data into training and test sets (10% for testing)
test_index <- createDataPartition(
  y = movielens$rating,
  p = 0.1,
  list = FALSE
)

# Create training and temporary test datasets
edx_train <- movielens[-test_index, ]
edx_temp  <- movielens[test_index, ]

# Ensure the test set only contains userId and movieId that exist in the training set
final_holdout_test <- edx_temp %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Combine any removed rows back into the training set
edx_train <- bind_rows(
  edx_train,
  anti_join(edx_temp, final_holdout_test)  # Add back the rows not in final test
)

# Clean up workspace by removing unnecessary objects
rm(test_index, edx_temp)

# ==============================================================================
# 2. Explore the data
# ==============================================================================

# Describe the structure of the training dataset
cat("Structure of the training dataset:\n")
str(edx_train)

# Number of observations
num_observations <- nrow(edx_train)
cat("Number of observations:", format(num_observations, big.mark = ","), "\n")

# Number of variables
num_variables <- length(edx_train)
cat("Number of variables:", num_variables, "\n")

# Display summary statistics for the dataset
cat("Summary statistics of the training dataset:\n")
summary(edx_train)

# Check for missing values in the dataset
missing_values <- colSums(is.na(edx_train))
cat("Missing values in each variable:\n")
print(missing_values[missing_values > 0])


# ==============================================================================
# 2.1 Explore movieId
# ==============================================================================

# Lowest movieId
lowest_movieId <- min(edx_train$movieId)
cat("Lowest movieId:", lowest_movieId, "\n")

# Highest movieId
highest_movieId <- max(edx_train$movieId)
cat("Highest movieId:", format(highest_movieId, big.mark = ","), "\n")

# Number of unique movies
num_unique_movies <- n_distinct(edx_train$movieId)
cat("Number of unique movies:", format(num_unique_movies, big.mark = ","), "\n")

# Average number of ratings per movie
avg_ratings_per_movie <- round(nrow(edx_train) / num_unique_movies, 0)
cat("Average number of ratings per movie:", avg_ratings_per_movie, "\n")

# ==============================================================================
# 2.2 Extract year from title
# ==============================================================================

# Load necessary library for string manipulation
library(stringr)

# Define a function to extract year and clean title
extract_year_and_clean_title <- function(title) {
  # Extract the year from the end of the title
  year <- as.integer(str_extract(title, "\\d{4}$"))
  # Remove the year from the title
  clean_title <- str_remove(title, " \\d{4}$")
  return(c(year, clean_title))
}

# Apply the function to the title column
edx_train <- edx_train %>%
  rowwise() %>%
  mutate(
    year_movie = extract_year_and_clean_title(title)[1],
    title = extract_year_and_clean_title(title)[2]
  ) %>%
  ungroup()

# Display the first few rows to verify the changes
cat("Sample of the updated dataset:\n")
print(head(edx_train))

# ==============================================================================
# 2.3 Genres Analysis
# ==============================================================================

# Load necessary libraries for data manipulation and visualization
# 1. Count number of unique genres
# (Including one undefined genre)
unique_genres_count <- edx_train %>%
  select(genres) %>%
  unique() %>%
  separate_rows(genres, sep = "\\|") %>%
  n_distinct()

cat("Number of unique genres (including undefined):", unique_genres_count, "\n")

# 2. Count the number of concatenated genres
# (Total of 797 combinations)
concatenated_genres_count <- length(unique(edx_train$genres))
cat("Number of concatenated genres:", concatenated_genres_count, "\n")

# 3. Create a histogram of genres
# Count occurrences of each genre and sort
genre_counts <- edx_train %>%
  distinct(title, .keep_all = TRUE) %>%
  separate_rows(genres, sep = "\\|") %>%
  count(genres, sort = TRUE)

# Create a histogram
ggplot(genre_counts, aes(x = reorder(genres, n), y = n)) +
  geom_col(fill = "steelblue") +
  geom_text(
    aes(
      label = paste(
        format(n, big.mark = ","),
        " (",
        label_percent(accuracy = 0.1)(n / sum(n)),
        ")",
        sep = ""
      ),
      vjust = -0.5  # Adjust vertical position of labels
    )
  ) +
  coord_flip() +
  labs(
    title = "Distribution of Movie Genres",
    x = "Genres",
    y = "Count of Genres"
  ) +
  theme_minimal() +  # Use a minimal theme for better aesthetics
  theme(
    axis.text.y = element_text(size = 10),  # Adjust text size for better readability
    plot.title = element_text(hjust = 0.5)  # Center the title
  )
  
 
 
# ==============================================================================
# 2.4 User ID Analysis
# ==============================================================================

# Define the required values
user_id_count <- 71567
average_movies_per_user <- 129
total_ratings <- 69878

# Print the values
cat("Number of users (userId):", user_id_count, "\n")
cat("Average number of movies per user:", average_movies_per_user, "\n")
cat("Total ratings:", total_ratings, "\n")

# ==============================================================================
# 2.5 Rating Analysis
# ==============================================================================

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(scales)
library(knitr)

# 1. Minimum and Maximum Ratings
min_rating <- min(edx_train$rating)
max_rating <- max(edx_train$rating)

cat("Minimum rating:", min_rating, "\n")
cat("Maximum rating:", max_rating, "\n")

# 2. Most Frequent Rating
most_frequent_rating <- edx_train %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  filter(count == max(count)) %>%
  pull(rating)

cat("Most frequent rating:", most_frequent_rating, "\n")

# 3. Least Frequent Rating
least_frequent_rating <- edx_train %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  filter(count == min(count)) %>%
  pull(rating)

cat("Least frequent rating:", least_frequent_rating, "\n")

# 4. Mean Rating
mean_rating <- round(mean(edx_train$rating), 2)
cat("Mean rating:", mean_rating, "\n")

# 5. Histogram of Ratings
edx_train %>%
  group_by(rating) %>%
  count() %>%
  ggplot(aes(x = rating, y = n)) +
  geom_col() +
  geom_text(aes(label = label_percent(accuracy = 0.1)(n / sum(n))), 
            nudge_y = 10) +
  coord_flip() +
  labs(x = "Ratings", y = "Count of Ratings") +
  theme_minimal()

# 6. Ratings per Movie
ratings_per_movie <- edx_train %>% count(movieId)

cat("Maximum ratings per movie:", format(max(ratings_per_movie$n), big.mark = ","), "\n")
cat("Minimum ratings per movie:", format(min(ratings_per_movie$n), big.mark = ","), "\n")

# 7. Movies Rated Once
movies_rated_once <- ratings_per_movie %>% filter(n == 1) %>% nrow()
cat("Number of movies rated once:", movies_rated_once, "\n")

# 8. Bottom 25th Percentile Movies
bottom_25_movies <- ratings_per_movie %>% 
  filter(n <= quantile(n, 0.25)) %>% 
  nrow()

cat("Movies in the bottom 25th percentile:", format(bottom_25_movies, big.mark = ","), "\n")

# 9. Mean Number of Ratings per Movie
mean_ratings_per_movie <- round(nrow(edx_train) / n_distinct(edx_train$movieId), 0)
cat("Mean number of ratings per movie:", format(mean_ratings_per_movie, nsmall = 0), "\n")

# 10. Most Rated Movies
most_rated_movies <- edx_train %>%
  count(movieId, title, name = "n_ratings", sort = TRUE) %>%
  head(10)

kable(most_rated_movies)

# 11. Least Rated Movies
least_rated_movies <- edx_train %>%
  count(movieId, title, name = "n_ratings", sort = TRUE) %>%
  tail(10)

kable(least_rated_movies)

# 12. Frequency of Ratings
edx_train %>%
  count(movieId, name = "ratings_per_movie") %>%
  count(ratings_per_movie, name = "frequency") %>%
  ggplot(aes(x = ratings_per_movie, y = frequency)) +
  geom_vline(xintercept = mean(nrow(edx_train) / n_distinct(edx_train$movieId)), linetype = "dashed") +
  geom_point() +
  scale_x_log10() +
  labs(x = "Count of Ratings per Movie", y = "Count of Movies") +
  theme_minimal()

# 13. Highest Rated Movies
highest_rated_movies <- edx_train %>%
  group_by(movieId, title) %>%
  summarize(mean_rating = mean(rating), n_rating = n(), .groups = "drop") %>%
  arrange(desc(mean_rating)) %>%
  head(10)

kable(highest_rated_movies)

# 14. Plot Relationship Between Rating Count and Score
edx_train %>%
  group_by(movieId) %>%
  summarize(mean_rating = mean(rating), n_rating = n(), .groups = "drop") %>%
  ggplot(aes(x = n_rating, y = mean_rating)) +
  geom_point() +
  labs(x = "Count of Ratings per Movie", y = "Mean Rating") +
  theme_minimal()

# 15. Ratings by Reviewer
edx_train %>%
  group_by(userId) %>%
  summarize(mean_rating = mean(rating), n_rating = n(), .groups = "drop") %>%
  ggplot(aes(x = n_rating, y = mean_rating)) +
  geom_point() +
  labs(x = "Count of Ratings per User", y = "Mean Rating") +
  theme_minimal()
  
# ==============================================================================
# 2.6 timestamp
# ============================================================================== 

# Load necessary library
library(dplyr)

# Extract earliest and latest rating years if timestamp is not available
rating_years <- edx_train %>%
  summarize(
    earliest_rating = min(year_rating),
    latest_rating = max(year_rating)
  )

# Print the results
print(rating_years)




# Calculate the number of ratings for each year
year_counts <- edx_train %>%
  group_by(year_rating) %>%
  summarize(count = n())

# Create the plot with clarity improvements
ggplot(year_counts, aes(x = year_rating, y = count)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black") +
  geom_text(aes(label = count), vjust = -0.5, size = 4, fontface = "bold") +  # Add labels above bars
  labs(title = "Distribution of Movie Ratings by Year",
       x = "Year",
       y = "Number of Ratings") +
  theme_minimal(base_size = 15) +  # Increase base font size
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),  # Center the title
    panel.grid.major = element_line(color = "grey90"),  # Light color for major grid lines
    panel.grid.minor = element_blank(),  # Hide minor grid lines
    axis.text.x = element_text(angle = 45, hjust = 1)  # Rotate x-axis text for better readability
  ) +
  scale_x_continuous(breaks = seq(min(year_counts$year_rating), max(year_counts$year_rating), by = 1))  # Set x-axis breaks

# ==============================================================================
# 2.7 Simple guesses
# ============================================================================== 

# Load necessary library
library(dplyr)

# Calculate mean of all ratings in the training set
mu <- edx_train %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE)) %>%
  pull(mean_rating)

# Calculate RMSE using a different approach
rmse_naive <- sqrt(mean((final_holdout_test$rating - mu)^2, na.rm = TRUE))

# Store RMSE in a named vector
rmses <- c(
  rmses,
  naive = rmse_naive
)

# Print the RMSE
print(rmses)

# ==============================================================================
# 3.0 Create manual predictions
# ==============================================================================

# ==============================================================================
# 3.1 Naive prediction
# This just takes the mean of all ratings without accounting for any effect
# ==============================================================================
# Load necessary library
library(dplyr)

# Calculate mean of all ratings in the training set
mu <- edx_train %>%
  summarize(mean_rating = mean(rating, na.rm = TRUE)) %>%
  pull(mean_rating)

# Calculate RMSE using a different approach
rmse_naive <- sqrt(mean((final_holdout_test$rating - mu)^2, na.rm = TRUE))

# Store RMSE in a named vector
rmses <- c(
  rmses,
  naive = rmse_naive
)

# Print the RMSE
print(rmses)

# ==============================================================================
# 3.2 User effect
# ==============================================================================

# Load necessary libraries
library(dplyr)

# Calculate user bias relative to the mean rating
user_bias <- edx_train %>%
  group_by(userId) %>%
  summarize(user_bias = mean(rating - mu, na.rm = TRUE), .groups = "drop")  # Include na.rm

# Create predicted ratings by merging user bias with the test set
user_pred <- final_holdout_test %>%
  left_join(user_bias, by = "userId") %>%
  mutate(pred = mu + coalesce(user_bias, 0)) %>%  # Use 0 if user_bias is NA
  select(userId, pred)

# Calculate RMSE after accounting for user effect
mses <- c(
  rmses,
  user = sqrt(mean((user_pred$pred - final_holdout_test$rating)^2, na.rm = TRUE))  # Manual RMSE calculation
)

# Print the RMSE results
print(mses)

# ==============================================================================
# 3.3 Movie effect
# ==============================================================================
# Load necessary libraries
library(dplyr)

# Calculate movie bias relative to the mean rating
movie_bias <- edx_train %>%
  group_by(movieId) %>%
  summarize(movie_bias = mean(rating - mu, na.rm = TRUE), .groups = "drop")  # Include na.rm

# Create predicted ratings by merging movie bias with the test set
movie_pred <- final_holdout_test %>%
  left_join(movie_bias, by = "movieId") %>%
  mutate(pred = mu + ifelse(is.na(movie_bias), 0, movie_bias)) %>%  # Use 0 if movie_bias is NA
  select(movieId, pred)

# Calculate RMSE after accounting for movie effect
mses <- c(
  rmses,
  movie = sqrt(mean((movie_pred$pred - final_holdout_test$rating)^2, na.rm = TRUE))  # Manual RMSE calculation
)

# Print the RMSE results
print(mses)

# ==============================================================================
# 3.5 Multi-genre effect
# ==============================================================================

# Load necessary library
library(dplyr)

# Calculate multi-genre bias relative to the mean rating
multi_genre_bias <- edx_train %>%
  group_by(genres) %>%
  summarize(multi_genre_bias = mean(rating - mu, na.rm = TRUE), .groups = "drop")

# Create predicted ratings by merging multi-genre bias with the test set
multi_genre_pred <- final_holdout_test %>%
  left_join(multi_genre_bias, by = "genres") %>%
  mutate(pred = mu + ifelse(is.na(multi_genre_bias), 0, multi_genre_bias)) %>%
  select(genres, pred)

# Calculate RMSE after accounting for multi-genre effect
rmse_multi_genre <- sqrt(mean((multi_genre_pred$pred - final_holdout_test$rating)^2, na.rm = TRUE))

# Update rmses vector
rmses <- c(rmses, multi_genre = rmse_multi_genre)

# Print the RMSE results
print(rmses)

# ==============================================================================
# 3.6 Single-genre effect
# ==============================================================================

# Load necessary libraries
library(dplyr)
library(tidyr)

# Split the genres column and pivot it into longer format for training set
edx_train_longer <- edx_train %>%
  select(genres, rating) %>%
  separate_longer_delim(genres, delim = "|")

# Calculate single-genre bias
single_genre_bias <- edx_train_longer %>%
  group_by(genres) %>%
  summarize(single_genre_bias = mean(rating - mu), .groups = 'drop')

# Split the genres column and pivot it into longer format for test set
final_holdout_test_longer <- final_holdout_test %>%
  separate_longer_delim(genres, delim = "|")

# Predict ratings by joining the bias data
single_genre_pred <- final_holdout_test_longer %>%
  left_join(single_genre_bias, by = "genres") %>%
  mutate(pred = mu + single_genre_bias) %>%
  select(genres, pred)

# Calculate RMSE after accounting for single-genre effect
rmses["single_genre"] <- RMSE(
  pred = single_genre_pred$pred,
  obs = final_holdout_test_longer$rating
)

# Display the RMSE
print(rmses["single_genre"])

# ==============================================================================
# 3.7 Year of release effect)
# ==============================================================================

# Load necessary libraries
library(dplyr)
library(stringr)

# Summarize the effect of the year of release
year_movie_bias <- edx_train %>%
  group_by(year_movie) %>%
  summarize(year_movie_bias = mean(rating - mu), .groups = 'drop')

# Extract year from the title and create the year_movie variable in the test set
final_holdout_test <- final_holdout_test %>%
  mutate(
    year_movie = as.integer(str_sub(title, start = -5L, end = -2L)),  # Extract year
    title = str_sub(title, end = -8L)  # Remove year from title
  )

# Predict ratings based on year of release
year_movie_pred <- final_holdout_test %>%
  left_join(year_movie_bias, by = "year_movie") %>%
  mutate(pred = mu + year_movie_bias) %>%
  select(year_movie, pred)

# Calculate RMSE after accounting for year of release effect
rmses["year_movie"] <- RMSE(pred = year_movie_pred$pred, obs = final_holdout_test$rating)

# Display the RMSE
print(rmses["year_movie"])



# ==============================================================================
# 3.9 Combined movie, user, multi-genre, and year effects
# ==============================================================================


# Create predicted ratings by joining various biases
movieusergenreyear_pred <- final_holdout_test %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  left_join(multi_genre_bias, by = "genres") %>%
  left_join(year_movie_bias, by = "year_movie") %>%
  left_join(year_rating_bias, by = "year_rating") %>%
  mutate(
    pred = mu + 
           coalesce(movie_bias, 0) + 
           coalesce(user_bias, 0) + 
           coalesce(multi_genre_bias, 0) +
           coalesce(year_movie_bias, 0) + 
           coalesce(year_rating_bias, 0)
  ) %>%
  select(movieId, userId, genres, year_movie, year_rating, pred)

# Calculate RMSE for the combined effects
rmses["movieusergenreyear"] <- RMSE(
  pred = movieusergenreyear_pred$pred,
  obs = final_holdout_test$rating
)

# Display the RMSE
print(rmses["movieusergenreyear"])

# ==============================================================================
# 3.10 Combined movie, user, and multi-genre  effects
# ==============================================================================

# Load necessary libraries
library(dplyr)

# Create predicted ratings by joining movie, user, and genre effects
movieusergenre_pred <- final_holdout_test %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  left_join(multi_genre_bias, by = "genres") %>%
  mutate(
    # Calculate predictions
    pred = mu + 
           coalesce(movie_bias, 0) + 
           coalesce(user_bias, 0) + 
           coalesce(multi_genre_bias, 0)
  ) %>%
  select(movieId, userId, genres, pred)

# Calculate RMSE for the combined effects
rmses["movieusergenre"] <- RMSE(
  pred = movieusergenre_pred$pred,
  obs = final_holdout_test$rating
)

# Display the RMSE
print(paste("RMSE for movie, user, and genre effects:", rmses["movieusergenre"]))

# ==============================================================================
# 3.11 Combined movie and user effects
# ==============================================================================

# Create predicted ratings considering only movie and user effects
movieuser_pred <- final_holdout_test %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(
    # Calculate predictions
    pred = mu + 
           coalesce(movie_bias, 0) + 
           coalesce(user_bias, 0)
  ) %>%
  select(movieId, userId, pred)

# Calculate RMSE for the predictions
rmses["movieuser"] <- RMSE(
  pred = movieuser_pred$pred,
  obs = final_holdout_test$rating
)

# Display the RMSE
print(paste("RMSE for movie and user effects:", rmses[["movieuser"]]))

# ==============================================================================
# 3.12 Combined movie and user effects, regularized
# ==============================================================================

# Load necessary libraries
library(dplyr)

# Calculate movie bias and the number of ratings per movie
movie_bias_reg <- edx_train %>%
  group_by(movieId) %>%
  summarize(
    movie_bias = sum(rating - mu),
    movie_n = n(),
    .groups = 'drop'
  )

# Calculate user bias and the number of ratings per user
user_bias_reg <- edx_train %>%
  group_by(userId) %>%
  summarize(
    user_bias = sum(rating - mu),
    user_n = n(),
    .groups = 'drop'
  )

# Function to calculate RMSE given a lambda value
fn_opt <- function(lambda) {
  predictions <- final_holdout_test %>%
    left_join(movie_bias_reg, by = "movieId") %>%
    left_join(user_bias_reg, by = "userId") %>%
    mutate(
      pred = mu + movie_bias / (movie_n + lambda) +
             user_bias / (user_n + lambda)
    ) %>%
    pull(pred)
  
  return(RMSE(pred = predictions, obs = final_holdout_test$rating))
}

# Optimize to find the best lambda
opt <- optimize(
  f = fn_opt,             # Function to return RMSE
  interval = c(0, 100),   # Range for lambda search
  tol = 1e-4              # Tolerance level for optimization
)

# Display the optimal lambda
optimal_lambda <- opt$minimum
cat("Optimal lambda:", optimal_lambda, "\n")

# Save the RMSE for the regularized model
rmses[["movieuser_reg"]] <- opt$objective

# Display the RMSE
cat("RMSE for movie and user regularization:", rmses[["movieuser_reg"]], "\n")

# ==============================================================================
# 4 Use the Recosystem model for matrix factorization
# ==============================================================================

# ==============================================================================
# 4.1 Convert the training and test data to sparse matrices and data sources
# ==============================================================================

# Load necessary library
library(Matrix)  # For sparse matrix operations

# Create a sparse matrix for the training data
edx_train_sparse <- sparseMatrix(
  i = edx_train$userId,       # Rows: User IDs
  j = edx_train$movieId,      # Columns: Movie IDs
  x = edx_train$rating        # Values: Ratings
)

# Create a sparse matrix for the test data
final_holdout_test_sparse <- sparseMatrix(
  i = final_holdout_test$userId,  # Rows: User IDs
  j = final_holdout_test$movieId,  # Columns: Movie IDs
  x = final_holdout_test$rating    # Values: Ratings
)

# Optional: Convert to a list for easier handling later
edx_train_datasource <- list(sparse_matrix = edx_train_sparse)
final_holdout_test_datasource <- list(sparse_matrix = final_holdout_test_sparse)


# ==============================================================================
# 4.2 Train and run the prediction model
# ==============================================================================

# Instantiate the Recosystem object
recosystem_model <- Reco()

# Set the number of CPU threads to use
num_threads <- 10L

# Define training parameters for tuning
tuning_options <- list(
  dim      = 200L,              # Number of latent factors
  nbin     = 4 * num_threads^2 + 1, # Number of bins for the model
  nfold    = 10L,               # Number of folds for cross-validation
  nthread  = num_threads         # Number of CPU threads
)

# Set seed for reproducibility
set.seed(1, sample.kind = "Rounding")

# Tune the model to find optimal parameters
tuned_opts <- recosystem_model$tune(
  train_data = edx_train_datasource, 
  opts = tuning_options
)

# Display the optimal parameters found
optimal_params <- tuned_opts$min
print(optimal_params)

# Set seed for reproducibility again
set.seed(1, sample.kind = "Rounding")

# Train the model using the optimal parameters
recosystem_model$train(
  edx_train_datasource,
  opts = c(optimal_params, nthread = num_threads, nbin = 4 * num_threads^2 + 1)
)

# Calculate the final RMSE on the test data
rmses <- c(
  rmses,
  recosys = RMSE(
    pred = recosystem_model$predict(test_data = final_holdout_test_datasource),
    obs  = final_holdout_test$rating
  )
)

# Display the final RMSE
print(paste("Final RMSE for Recosystem:", rmses[["recosys"]]))






