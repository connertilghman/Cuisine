library(jsonlite)
library(readr)
library(tidytext)
library(tidyr)
library(tidyverse)
library(tidymodels)
library(textrecipes)

trainSet <- fromJSON("train.json") 
testSet <- fromJSON("test.json") 

trainSet$cuisine <- as.factor(trainSet$cuisine)


trainSet$num_ingredients <- sapply(trainSet$ingredients, length)
trainSet$avg_ingredient_length <- sapply(trainSet$ingredients, function(x) mean(nchar(x)))
trainSet$num_long_ingredients <- sapply(trainSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) > 1))

testSet$num_ingredients <- sapply(testSet$ingredients, length)
testSet$avg_ingredient_length <- sapply(testSet$ingredients, function(x) mean(nchar(x)))
testSet$num_long_ingredients <- sapply(testSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) > 1))

trainSet$num_unique_words <- sapply(trainSet$ingredients, function(x) length(unique(unlist(strsplit(x, " ")))))
testSet$num_unique_words  <- sapply(testSet$ingredients, function(x) length(unique(unlist(strsplit(x, " ")))))

trainSet$num_single_word <- sapply(trainSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) == 1))
testSet$num_single_word  <- sapply(testSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) == 1))


rec <- recipe(cuisine ~ ingredients + num_ingredients + avg_ingredient_length + num_long_ingredients + num_unique_words + num_single_word,
              data = trainSet) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens = 2500) %>%   
  step_tfidf(ingredients)


data_split <- initial_split(trainSet, prop = 0.8, strata = cuisine)
train_data <- training(data_split)
valid_data <- testing(data_split)


xgb_model <- boost_tree(
  trees = 250,        
  tree_depth = 10,     
  learn_rate = 0.05,  
  loss_reduction = 0, 
  sample_size = 0.8   
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")


wf <- workflow() %>%
  add_model(xgb_model) %>%
  add_recipe(rec)

xgb_fit <- wf %>% fit(data = train_data)

xgb_preds <- predict(xgb_fit, valid_data) %>%
  bind_cols(valid_data %>% select(cuisine))

class_metrics <- metric_set(accuracy, kap, f_meas)

class_metrics(
  xgb_preds,
  truth = cuisine,
  estimate = .pred_class
)

xgb_final <- wf %>% fit(data = trainSet)

test_predictions <- predict(xgb_final, testSet)

submission <- tibble(
  id = testSet$id,
  cuisine = test_predictions$.pred_class
)

     
write_csv(submission, "submission.csv")
