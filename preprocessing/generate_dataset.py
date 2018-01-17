import csv
import json
import sys
from time import time

import json_lines
import os
import re
import random


def increase_field_limit():
	"""
	Increases the field limit for csv files.
	Source: https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
	:return:
	"""
	max_int = sys.maxsize
	decrement = True

	while decrement:
		decrement = False
		try:
			csv.field_size_limit(max_int)
		except OverflowError:
			max_int = int(max_int / 10)
			decrement = True


def read_fake_news():
	"""
	Reads fake news articles from file.
	:return: Array of fake news with label.
	"""
	print("Read fake news articles...")
	data = []
	with open("raw_data/fake.csv", mode="r", encoding="utf-8") as f:
		for article in csv.DictReader(f):
			data.append([clean_text(article["text"]), 1])
	return data


def generate_sample():
	"""
	Generates a sample of genuine articles.
	:return: Array of genuine articles with label.
	"""
	articles = read_articles()
	print("Generate sample...")
	sample = random.sample(articles, 52001)
	data = []
	for article in sample:
		data.append([article, 0])
	return data


def read_articles():
	"""
	Reads sample article file.
	:return: Array of news articles.
	"""
	print("Read genuine articles...")
	articles = []
	with open("raw_data/sample-1M.jsonl", mode="rb") as f:
		for article in json_lines.reader(f):
			if article["media-type"] == "News":						# filters blog entries
				articles.append(clean_text(article["content"]))
	return articles


def clean_text(text):
	"""
	Cleans a string from upper and special characters.
	:param text: string to clean
	:return: cleaned string
	"""
	cleaned_text = text.lower()
	return re.sub(r'[^\w\s]', '', cleaned_text)


def split_fake_base(data_set):
	"""
	Splits the data set into training, test and validation set.
	:param data_set: data set to split
	:return: three data sets according to specification
	"""
	data_set = random.sample(data_set, len(data_set))
	return data_set[0:39000], data_set[39000:52000], data_set[52000:65000]


def save_data_set(data_set):
	"""
	Saves an array.
	:param data_set: array to save
	:return:
	"""
	print("Save files...")
	if not os.path.exists("data"):
		os.mkdir("data")
	training_set, validation_set, test_set = split_fake_base(data_set)

	json.dump(data_set, open("data/fake_base_complete.json", "w"))
	json.dump(training_set, open("data/training_set.json", "w"))
	json.dump(validation_set, open("data/validation_set.json", "w"))
	json.dump(test_set, open("data/test_set.json", "w"))


if __name__ == "__main__":
	if not (os.path.exists("raw_data/fake.csv") and os.path.exists("raw_data/sample-1M.jsonl")):
		sys.exit("Critical Error!\nPlease place the raw data in the corresponding directory first.\nSee the readme "
		         "for more information.")
	start = time()
	increase_field_limit()
	fake_news_articles = read_fake_news()
	genuine_articles = generate_sample()
	save_data_set(fake_news_articles + genuine_articles)
	print("Operation successfully finished after %2.f seconds." % (time() - start))
