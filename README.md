# fake-news-detector
Implementing a fake news detector. Comparing different ML algorithms and NLP strategies.

# Steps to recreate experiment setup
## Setup environment
* Python 3 is required for this project. Version used in development is **Python 3.6.4.**
* Run `pip3 install -r requirements.txt` in the `setup` folder. This installs the required Python libraries.
## Get the data
Download data. First link provides 13,000 fake news articles, second 1 million genuine articles of which 
a random sample of [TODO: insert number] articles will be used in the experiment.
1. [Fake News dataset](https://www.kaggle.com/mrisdal/fake-news/data) (the file must be unzipped)
1. [The Signal Media One-Million News Articles Dataset](http://research.signalmedia.co/newsir16/signal-dataset.html) 
(the file must be unzipped)
## Preprocessing
Both files are placed under their original names in the `preprocessing/raw_data` directory.