#!/usr/bin/env bash
pip3 install -r setup/requirements.txt
echo "1. Skipping the generation of the dataset, as it is already given, will place it in the right folders."
echo "Prepared step 2."
echo "Starting feature generation for the validation set."
mkdir -p feature_generation/data
python3 feature_generation/hashing_vectorizer.py -l 1 -u 1
python3 feature_generation/hashing_vectorizer.py -l 2 -u 2
python3 feature_generation/hashing_vectorizer.py -l 3 -u 3
python3 feature_generation/ngram_vectorizer.py -l 1 -u 1
python3 feature_generation/ngram_vectorizer.py -l 2 -u 2
python3 feature_generation/ngram_vectorizer.py -l 3 -u 3
python3 feature_generation/ngram_vectorizer.py -l 1 -u 1 -t
python3 feature_generation/ngram_vectorizer.py -l 2 -u 2 -t
python3 feature_generation/ngram_vectorizer.py -l 3 -u 3 -t

echo "Done feature generation for the validation set."
echo "Starting hyperparameter optimization."
python3 hyperparameter_optimization/randomized_search.py

echo "Done with hyperparameter optimization and edit them in 'classifier/experiment.py'."
echo "Will generate features for training and test set now."
mkdir -p feature_generation/data/trained
mkdir -p feature_generation/vectorizers
python3 feature_generation/hashing_vectorizer_training.py -l 1 -u 1
python3 feature_generation/hashing_vectorizer_training.py -l 2 -u 2
python3 feature_generation/hashing_vectorizer_training.py -l 3 -u 3
python3 feature_generation/ngram_vectorizer_training.py -l 1 -u 1
python3 feature_generation/ngram_vectorizer_training.py -l 2 -u 2
python3 feature_generation/ngram_vectorizer_training.py -l 3 -u 3
python3 feature_generation/ngram_vectorizer_training.py -l 1 -u 1 -t
python3 feature_generation/ngram_vectorizer_training.py -l 2 -u 2 -t
python3 feature_generation/ngram_vectorizer_training.py -l 3 -u 3 -t

echo "Starting regeneration now."
mkdir -p feature_regeneration/data
python3 feature_regeneration/hashing_revectorizer.py -l 1 -u 1
python3 feature_regeneration/hashing_revectorizer.py -l 2 -u 2
python3 feature_regeneration/hashing_revectorizer.py -l 3 -u 3
python3 feature_regeneration/ngram_revectorizer.py -l 1 -u 1
python3 feature_regeneration/ngram_revectorizer.py -l 2 -u 2
python3 feature_regeneration/ngram_revectorizer.py -l 3 -u 3
python3 feature_regeneration/ngram_revectorizer.py -l 1 -u 1 -t
python3 feature_regeneration/ngram_revectorizer.py -l 2 -u 2 -t
python3 feature_regeneration/ngram_revectorizer.py -l 3 -u 3 -t

echo "Done. Will start experiment now with values that were calculated in the project. If you want to use the
parameters that you have optimized, stop the script, edit 'classification/experiment.py' and run 'python3
classification/experiment.py'. The results won't be the same as in my setup anyway!"
python3 classification/experiment.py