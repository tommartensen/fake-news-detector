import getopt
import json
import os
import pickle

import sys
from sklearn.feature_extraction.text import HashingVectorizer


def dump_vectorizer(vectorizer, filename):
	print("Dumping vectorizer...")
	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/vectorizers/" + filename + ".vec"), "wb") as f:
		pickle.dump(vectorizer, f)


def main(argv):
	if len(argv) < 3:
		print('hashing_vectorizer.py -l <lower bound> -u <upper bound>')
		sys.exit(2)
	lower_bound = 1
	upper_bound = 1
	try:
		opts, args = getopt.getopt(argv, "htl:u:", ["lower_bound=", "upper_bound="])
	except getopt.GetoptError:
		print('hashing_vectorizer.py -l <lower bound> -u <upper bound>')
		sys.exit(2)
	filename = "hashed_10000"
	for opt, arg in opts:
		if opt == '-h':
			print('hashing_vectorizer.py -l <lower bound> -u <upper bound>')
			sys.exit()
		elif opt in ("-l", "--lower_bound"):
			filename += "_l" + arg
			lower_bound = int(arg)
		elif opt in ("-u", "--upper_bound"):
			filename += "_u" + arg
			upper_bound = int(arg)

	articles = []
	labels = []

	print("Preparing data...")
	with open(os.path.join(os.path.dirname(__file__), "../preprocessing/data/training_set.json"), "r") as f:
		data = json.load(f)
		for article in data:
			articles.append(article[0])
			labels.append(article[1])

	print("Vectorizing ngrams...")
	hv = HashingVectorizer(ngram_range=(lower_bound, upper_bound), n_features=10000, stop_words="english")
	features = hv.transform(articles).toarray()

	print("Dumping tokenized features...")
	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/data/trained/" + filename + ".json"), "w") as f:
		json.dump(features.tolist(), f)
	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/data/trained/labels_training.json"), "w") as f:
		json.dump(labels, f)

	dump_vectorizer(hv, filename)


if __name__ == "__main__":
	main(sys.argv[1:])
