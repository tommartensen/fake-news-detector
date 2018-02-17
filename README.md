# Fake News Detection
Implementing a fake news detector. Comparing different ML algorithms and NLP strategies.

# Requirements
* Python 3, at least **Python 3.5.2**
* Python 3 package manager **pip3**
* Tested on Ubuntu 16.04

# Running experiment with submission data set
In `hyperparameter_optimization/randomized_search.py`, line 44 change the value of the variable `n_iter_search` which 
determines the number of combinations that are tested in the parameter search to a value of your choice. The lower 
this value, the faster the script, but the less values are tested.
Then, execute the script `./run-project.sh`.

# Running experiment step by step
## Setup Environment
* Python 3 is required for this project. Version used in development is **Python 3.6.4.**
* Run `pip3 install -r requirements.txt` in the `setup` folder. This installs the required Python libraries.
## Get the data
Download data. First link provides 13,000 fake news articles, second 1 million genuine articles of which 
a random sample of 52,000 articles will be used in the experiment. The share of fake news articles in the whole dataset
is therefore 20%. 
1. [Fake News dataset](https://www.kaggle.com/mrisdal/fake-news/data) (the file must be unzipped)
1. [The Signal Media One-Million News Articles Dataset](http://research.signalmedia.co/newsir16/signal-dataset.html) 
(the file must be unzipped)
## Preprocessing
Both downloaded files must be placed under their original names in the `preprocessing/raw_data` directory. 
* For the experiment, run `python3 generate_dataset.py`. This generates a dataset with the above mentioned 
characteristics.
* To reproduce the submission dataset, run `python3 generate_submission_dataset.py`. This generates a dataset with the 
above mentioned characteristics and 1,500 lines.
## Feature Generation
We distinguish two vectorizers: 
* hash vectorizer: `python3 hashing_vectorizer.py -l <lower bound> -u <upper bound>` 
* ngram vectorizer: `python3 ngram_vectorizer.py -l <lower bound> -u <upper bound> [-t]`

| Parameter|Values|
| :------------- |:-------------|
| `lower_bound` | tested with `1..3` |
| `upper_bound` | tested with `1..3` |
| `-t` | use term frequency–inverse document frequency (only for ngrams)|

In the experiment, `lower_bound` and `upper_bound` were set equal. The implementations are based on sklearn's 
implementation of the `HashingVectorizer` and `CountVectorizer`.

If the training set should be vectorized and the vectorizers stowed away for the regeneration 
of the feature vectors with the test set, use `hashing_vectorizer_training.py` and `ngram_vectorizer_training.py` 
instead, with the same parameters.

## Hyperparameter Optimization
To optimize the parameters for the machine learning algorithms in the experiment, run the hyperparameter optimization
 script. Therefore, all features for the validation set must already be generated as explained above! In line 44 of 
 the script, you can set the number of configurations that should be tested. A higher number may give better results,
  but also take longer.   
Then, run `python3 randomized_search.py`. 

## Feature Regeneration
This step is to create the vectors for the test set that is used in the final assessment of the algorithms. This can 
only work, when the training set was already vectorized.

Then, run both `python3 hashing_revectorizer.py` and `python3 ngram_revectorizer.py` with the same parameter settings
 that are mentioned in [this step](#feature-generation).
 
## Classifier Performance
The classifier performance is evaluated with the f1-score. The optimized parameters that were obtained in [this step](#hyperparameter-optimization) 
 must be configured in code. The output is printed directly on the console.
 Run `experiment.py` and have the vectorized training and test set ready in the `data` folders in 
 `feature_generation` and `feature_regeneration` respectively, this should have happened automatically.