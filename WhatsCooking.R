library(jsonlite)
library(readr)
library(tidytext)
library(tidyr)
library(tidyverse)
library(tidymodels)
library(textrecipes)

trainSet <- read_file("train.json") |>
  fromJSON()
testSet <- read_file("test.json") |>
  fromJSON()

trainSet <- trainSet |>
  unnest(ingredients)
testSet <- testSet |>
  unnest(ingredients)


trainSet <- trainSet |> 
  mutate(
    # 1. Ingredient count
    n_ingredients = map_int(ingredients, length),
    
    # 3. Category indicators
    has_turmeric = map_lgl(ingredients, ~ any(str_detect(.x, "turmeric"))),
    has_soy      = map_lgl(ingredients, ~ any(str_detect(.x, "soy"))),
    has_cumin    = map_lgl(ingredients, ~ any(str_detect(.x, "cumin")))
  )

# --- Full RECIPE ---

recipe <- recipe(cuisine ~ ingredients + n_ingredients + 
         has_turmeric + has_soy + has_cumin,
       data = trainSet) |>
  
  ## Unnest ingredient list into tokens
  step_tokenize(ingredients) |> 
  
  ## Remove stopwords like "fresh", "ground", etc.
  step_stopwords(ingredients) |>
  
  ## Filter vocabulary to top words (helps speed)
  step_tokenfilter(ingredients, max_tokens = 2000) |>
  
  ## Create TF-IDF features
  step_tfidf(ingredients)

rf_spec <- rand_forest(
  mtry = 10,        # number of variables randomly sampled at each split
  trees = 500,      # number of trees
  min_n = 5         # minimum samples per leaf
) |>
  set_engine("ranger") |>
  set_mode("classification")

rf_workflow <- workflow() |>
  add_model(rf_spec) |>
  add_recipe(recipe)

rf_fit <- rf_workflow |> fit(data = trainSet)
