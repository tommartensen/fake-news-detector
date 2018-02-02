import getopt
import json
import os
import pickle

import sys
from sklearn.feature_extraction.text import HashingVectorizer


def dump_vectorizer(vectorizer, filename):
	print("Dumping vectorizer...")
	if not os.path.exists("vectorizers"):
		os.mkdir("vectorizers")
	with open("vectorizers/" + filename + ".vec", "w") as f:
		pickle.dump(vectorizer, f)


def main(argv):
	if len(argv) < 2:
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
	with open(os.path.join(os.path.dirname(__file__), "../preprocessing/data/validation_set.json"), "r") as f:
		data = json.load(f)
		for article in data:
			articles.append(article[0])
			labels.append(article[1])

	print("Vectorizing ngrams...")
	hv = HashingVectorizer(ngram_range=(lower_bound, upper_bound), n_features=10000, stop_words="english")
	features = hv.transform(articles).toarray()

	print("Dumping tokenized features...")
	if not os.path.exists("data"):
		os.mkdir("data")
	with open("data/" + filename + ".json", "w") as f:
		json.dump(features.tolist(), f)
	with open("data/labels.json", "w") as f:
		json.dump(labels, f)

	# Use when vectorizing training set to reuse for test set!
	# dump_vectorizer(hv, filename)


if __name__ == "__main__":
	main(sys.argv[1:])
