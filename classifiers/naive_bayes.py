import json
import os

from numpy import mean, std
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
	print("Preparing data...")
	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/data/ngram_train_l1_u1.json"), "r") as f:
		X = json.load(f)
	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/data/labels.json"), "r") as f:
		Y = json.load(f)

	print("Starting cross validation score test...")
	scores = cross_val_score(verbose=True, n_jobs=-1, cv=5, estimator=RandomForestClassifier(verbose=True), X=X, y=Y)
	print(scores)
	print(mean(scores), std(scores))
