# Fake News Detection
Implementing a fake news detector. Comparing different ML algorithms and NLP strategies.


# Steps to recreate experiment
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
* hash vectorizer: `python3 (hashing_vectorizer.py -l <lower bound> -u <upper bound>` 
* ngram vectorizer: `python3 (ngram_vectorizer.py -l <lower bound> -u <upper bound> [-t]`

| Parameter|Values|
| :------------- |:-------------|
| `lower_bound` | tested with `1..3` |
| `upper_bound` | tested with `1..3` |
| `-t` | use term frequency–inverse document frequency (only for ngrams)|

In the experiment, `lower_bound` and `upper_bound` were set equal. The implementations are based on sklearn's 
implementation of the `HashingVectorizer` and `CountVectorizer`. 
As is these programs generate the features based on the `validation_set.json`, which must be supplied in the 
`raw_data` directory. If the training set should be vectorized and the vectorizers stowed away for the regeneration 
of the feature vectors with the test set, uncomment the lines marked in the files.
## Hyperparameter Optimization
To optimize the parameters for the machine learning algorithms in the experiment, run the hyperparameter optimization
 script. This requires the output of [this step](#feature-generation) placed in the `raw_data` directory relative to 
 `randomized_search.py`.
  
Then, run `python3 randomized_search.py`. 
## Feature Regeneration
This step is to create the vectors for the test set that is used in the final assessment of the algorithms. 
Therefore, place the `test_set.json` file in the `raw_data` directory and the vectorizers that were produced in 
[this-step](#feature-generation) in `vectorizers`.

Then, run both `python3 hashing_revectorizer.py` and `python3 ngram_revectorizer.py`.
## Classifier Performance
The classifier performance is evaluated with the f1-score. The optimized parameters that were obtained in [this step](#hyperparameter-optimization) 
 are configured in code. The output is printed directly on the console.
 Run `experiment.py` and have the vectorized training and test set ready in the `data` folders in 
 `feature_generation` and `feature_regeneration` respectively. 