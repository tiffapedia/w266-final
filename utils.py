import csv
import emoji
import json
import nltk
import numpy as np
import re
from nltk.corpus import stopwords

def get_tweets_with_location():
	with open('tweets.json') as f_tweets:
		json_tweets = json.load(f_tweets)
		#print(json_tweets[1]['text'])

		tweets_text = [] # should we change this to numpy
		tweets_place = []
		for tweet in json_tweets:
			if (tweet['user']['geo_enabled'] == True):
				tweets_text.append(tweet['text'])
				tweets_place.append(tweet['place']['full_name'])
	return tweets_text,tweets_place


def get_tweets_from_csv():
	with open('tweets_labelled.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')

		tweets_text = [] # should we change this to numpy
		tweets_place = []
		for row in csv_reader:
			tweets_text.append(row[0])
			tweets_place.append(row[1])
	return tweets_text,tweets_place


def preprocessor(text):
	# convert to lower case
	text = text.lower()
	# remove urls
	text = re.sub(r'http\S+', '', text)
	# remove stop words and emojis
	stop = set(stopwords.words('english'))
	text = ' '.join(word for word in nltk.word_tokenize(text) if word not in stop) #and word not in emoji.UNICODE_EMOJI
	#print(text)
	# remove special characters and numbers
	text = re.sub(r'\W+', ' ', text)
	#s = re.sub('(_|\(|\))','',s)
	# removing all digits ????????????????? think about this
	text = re.sub(r'(\d+)', '', text)
	# remove emoji
	#text = ''.join(word for word in text if word not in emoji.UNICODE_EMOJI)
	# perform stemming on words
	#s = re.sub(r'(\b\w+\b)',stem,s)
	# remove two letter words
	text = re.sub(r'\b[a-z][a-z]\b','',text)
	# remove single letter 
	text = re.sub(r'\b[a-z]\b','',text)
	# remove underscores
	text = re.sub(r'_','',text)

	return text


def tokenizer(text):
	tokens = nltk.word_tokenize(text)
	return tokens


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
