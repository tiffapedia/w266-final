
# coding: utf-8

# In[56]:


#import tweets_processor
import mlflow
import keras
import numpy as np
import talos as ta
import mlflow.keras
import importlib
import os
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda
from keras.layers import Embedding
from keras.layers import LSTM, RepeatVector
from sklearn.model_selection import train_test_split
from collections import Counter
from keras import backend as K
from sklearn.neighbors import NearestNeighbors
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b
from keras.optimizers import Adam, Nadam, RMSprop, Adadelta
from keras.activations import softmax, relu, tanh
from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy
from keras.initializers import Constant
from keras.models import Model
from keras.layers import TimeDistributed
from keras import objectives
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers import Dense, Dropout, Flatten
from scipy.spatial.distance import cdist
from keras.callbacks import ModelCheckpoint
from keras.layers import Cropping1D
from keras.layers import BatchNormalization


# In[2]:


#importlib.reload(tweets_processor)
# load the data
# get the tweets and the region labels from csv file
#tweets_text, tweets_regions = tweets_processor.get_tweets_from_csv('../data/tweets_labelled_40k.csv') # need to pass tweets of only one region or all?


# In[3]:


# another way to process the csv data
df = pd.read_csv('../data/tweets_labelled_balanced.csv', nrows=300000)
df.dropna(inplace=True)
df.region = df.region.astype(int)
tweets_text = df.text.tolist()
tweets_regions = df.region.tolist()


# In[4]:


# tokenize
# create the tokenizer at word level
t = Tokenizer(lower = True, filters ='')
t.fit_on_texts(tweets_text)


# In[5]:


# get the vocab size
vocab = list(t.word_counts.keys())
vocab_size = len(t.word_counts) + 1
vocab_ids = list(t.word_index.values())
word_index = t.word_index


# In[6]:


len(word_index)


# In[7]:


# convert the tweets to sequence of id's
encoded_tweets = t.texts_to_sequences(tweets_text)

# maximum input sequence length
# set it to max length tweet in the dataset
max_len = len(max(encoded_tweets, key=len))

# make inputs of same length(50 words, tweets won't be more than that) by using pad_sequences
padded_tweets = pad_sequences(encoded_tweets,padding='post',maxlen=max_len)


# In[8]:


# convert labels to categorical labels, we have 23 regions
categorical_labels = keras.utils.to_categorical(tweets_regions, num_classes=23)


# In[9]:


# split the data into train and test
train_data, test_data, train_labels, test_labels = train_test_split(padded_tweets, categorical_labels, test_size=0.1, random_state=5)


# In[10]:


# convert the test_data  into categorical of vocab_size
#vocab_size_test_data = keras.utils.to_categorical(test_data, num_classes=vocab_size)
# this is resulting in memory error so created a generator to generate categorical data of the input on the fly


# In[10]:


# first, build index mapping words in the embeddings set
# to their embedding vector
embeddings_index = {}
with open(os.path.join('../data', 'glove.twitter.27B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


# In[11]:


# prepare embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[29]:


sequence_length = max_len # padded length of words
input_dim = 100 #embedding feature dimensions since we used 100dimensional glove
batch_size = 10


# In[15]:


# input
inputs = Input(shape=(sequence_length,))

# embedding
encoder_input = Embedding(vocab_size,
                            input_dim,
                            weights=[embedding_matrix],
                            input_length=sequence_length,
                            trainable=False)(inputs)

# In[90]:


x = Conv1D(128, (3), activation='relu', padding='same', name='conv1')(encoder_input)
encoded = BatchNormalization(axis=-1)(x)
encoded = Conv1D(64, (3), activation='relu', padding='same', name='conv2')(encoded)
encoded = BatchNormalization(axis=-1)(encoded)
encoded = Dropout(0.2)(encoded)

flattened = Flatten(name='flatten')(encoded)
flattened = Dense(256, activation='relu', name='dense1')(flattened)
flattened = Dropout(0.2)(flattened)
flattened = Dense(256, activation='relu', name='dense2')(flattened)
flattened = Dropout(0.2)(flattened)
flattened = Dense(256, activation='relu', name='dense3')(flattened)
regions = Dense(23, activation='softmax', name='dense4')(flattened)

decoded = Conv1D(64, (3), activation='relu', padding='same', name='conv5')(encoded)
decoded = BatchNormalization(axis=-1)(decoded)
decoded = Conv1D(128, (3), activation='relu', padding='same', name='conv6')(decoded)
decoded = BatchNormalization(axis=-1)(decoded)
decoded = Dropout(0.2)(decoded)

preds = Dense(vocab_size, activation='softmax', name='dense5')(decoded)
model = Model(inputs, [regions,preds])


# In[91]:


model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy'])


# In[21]:


# creating this class for out of memeory issues when converting input data to categorical of vocab size
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_data, region_labels, batch_size, vocab_size):
        'Initialization'
        self.input_data = input_data
        self.region_labels = region_labels
        self.batch_size = batch_size
        self.vocab_size = vocab_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.input_data) / self.batch_size))

    def __getitem__(self, idx):
        begin = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        x = self.input_data[begin:end]
        y1 = self.region_labels[begin:end]
        y2 = keras.utils.to_categorical(self.input_data[begin:end], num_classes=self.vocab_size)
    
        return [x], [y1, y2]


# In[92]:


training_generator = DataGenerator(train_data, train_labels, batch_size, vocab_size)
validation_generator = DataGenerator(test_data, test_labels, batch_size, vocab_size)

callback = [keras.callbacks.EarlyStopping(monitor='dense4_loss'), ModelCheckpoint(filepath='cnn_300k.h5', 
                             monitor='dense4_loss', save_best_only=True)]

#with tf.Session( config = tf.ConfigProto( log_device_placement = True ) ): # to see the device information
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=10,
                    callbacks = callback)

# model.fit(test_data,
#         [test_labels,vocab_size_test_data], #,vocab_size_test_data
#         epochs=25, 
#         batch_size=1)


# In[93]:


model.summary()


# In[71]:


actual_test = test_data[:100]
actual_labels = test_labels[:100]
# convert the test_data into categorical of vocab_size
vocab_size_actual_test_data = keras.utils.to_categorical(actual_test, num_classes=vocab_size)
#print(vocab_size_actual_test_data)
score = model.evaluate(actual_test, [actual_labels,vocab_size_actual_test_data])


# In[72]:


score


# In[94]:

