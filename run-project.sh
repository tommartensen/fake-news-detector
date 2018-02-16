#!/usr/bin/env bash
pip3 install -r setup/requirements.txt
echo "1. Skipping the generation of the dataset, as it is already given, will place it in the right folders."
echo "Prepared step 2."
echo "Starting feature generation for the validation set."
mkdir feature_generation/data
python3 feature_generation/hashing_vectorizer.py -u 1 -l 1
python3 feature_generation/hashing_vectorizer.py -u 2 -l 2
python3 feature_generation/hashing_vectorizer.py -u 3 -l 3
python3 feature_generation/ngram_vectorizer.py -u 1 -l 1
python3 feature_generation/ngram_vectorizer.py -u 2 -l 2
python3 feature_generation/ngram_vectorizer.py -u 3 -l 3
python3 feature_generation/ngram_vectorizer.py -u 1 -l 1 -t
python3 feature_generation/ngram_vectorizer.py -u 2 -l 2 -t
python3 feature_generation/ngram_vectorizer.py -u 3 -l 3 -t

echo "Done feature generation for the validation set."
echo "Starting hyperparameter optimization."
python3 hyperparameter_optimization/randomized_search.py

echo "Done with hyperparameter optimization and edit them in 'classifier/experiment.py'."
echo "Will generate features for training and test set now."
python3 feature_generation/hashing_vectorizer_training.py -u 1 -l 1
python3 feature_generation/hashing_vectorizer_training.py -u 2 -l 2
python3 feature_generation/hashing_vectorizer_training.py -u 3 -l 3
python3 feature_generation/ngram_vectorizer_training.py -u 1 -l 1
python3 feature_generation/ngram_vectorizer_training.py -u 2 -l 2
python3 feature_generation/ngram_vectorizer_training.py -u 3 -l 3
python3 feature_generation/ngram_vectorizer_training.py -u 1 -l 1 -t
python3 feature_generation/ngram_vectorizer_training.py -u 2 -l 2 -t
python3 feature_generation/ngram_vectorizer_training.py -u 3 -l 3 -t

echo "Done. Copying vectorizers into right directory".
echo "Starting regeneration now."
mkdir feature_regeneration/data
python3 feature_generation/hashing_revectorizer.py -u 1 -l 1
python3 feature_generation/hashing_revectorizer.py -u 2 -l 2
python3 feature_generation/hashing_revectorizer.py -u 3 -l 3
python3 feature_generation/ngram_revectorizer.py -u 1 -l 1
python3 feature_generation/ngram_revectorizer.py -u 2 -l 2
python3 feature_generation/ngram_revectorizer.py -u 3 -l 3
python3 feature_generation/ngram_revectorizer.py -u 1 -l 1 -t
python3 feature_generation/ngram_revectorizer.py -u 2 -l 2 -t
python3 feature_generation/ngram_revectorizer.py -u 3 -l 3 -t

echo "Done. Will start experiment now with values that were calculated in the project. If you want to use the
parameters that you have optimized, stop the script, edit 'classification/experiment.py' and run 'python3
classification/experiment.py'. The results won't be the same as in my setup anyway!"
python3 classification/experiment.py