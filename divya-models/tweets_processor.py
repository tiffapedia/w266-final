import json
import re
import nltk
import emoji
import csv
from nltk.corpus import stopwords

#mapping of regions to integers
regions_mapping = {
    "albuquerque":0,
    "billings":1,
    "calgary":2,
    "charlotte":3,
    "chicago":4,
    "cincinnati":5,
    "denver":6,
    "houston":7,
    "kansas city":8,
    "las vegas":9,
    "los angeles":10,
    "minneapolis":11,
    "montreal":12,
    "nashville":13,
    "new york":14,
    "oklahoma city":15,
    "phoenix":16,
    "pittsburgh":17,
    "san francisco":18,
    "seattle":19,
    "tampa":20,
    "toronto":21,
    "washington":22
}


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
	#text = re.sub(r'\b[a-z][a-z]\b','',text)
	# remove single letter 
	text = re.sub(r'\b[a-z]\b','',text)
	# remove underscores
	text = re.sub(r'_','',text)

	return text

def tokenizer(text):
	tokens = nltk.word_tokenize(text)

	return tokens

def encode_labels(labels):
	encoded_labels = []
	for label in labels:
		encoded_labels.append(regions_mapping[label])

	return encoded_labels
