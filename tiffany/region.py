# general libraries
from pathlib import Path 
from nltk.corpus import stopwords

import tensorflow as tf

# model definition
from gensim.models import Word2Vec

# reduce feature sets
from sklearn.decomposition import PCA

# data visualization
from matplotlib import pyplot

class Region:
    def __init__(self, df, region_idx):
        self.region_idx = region_idx
        self.df = df.loc[df.region == region_idx]
        # word embedding
        self._set_embedding_model()
        # vocab's lookup table
        self._set_vocab()
    
    def _get_word_list_per_tweet(self):
        """ get the list of words for each tweet [[used in _set_embedding_model()]] """
        word_list = list()
        for sentence in self.df.text.tolist():
            word_list.append(re.sub("[^\w]", " ", sentence).split())
        return word_list

    def _set_embedding_model(self, load_existing=True):
        """ create a word embedding for the region and save result in /var/models/ """
        self.embedding_model_filepath = '/var/models/region_{}_embedding'.format(self.region_idx)
        
        if Path(self.embedding_model_filepath).exists() and load_existing:
            # load existing embedding model
            self.embedding_model = Word2Vec.load(self.embedding_model_filepath)
        else:
            # create a new word embedding model
            self.embedding_model = Word2Vec(self._get_word_list_per_tweet(), min_count=1)
            self.embedding_model.save(self.embedding_model_filepath)
        
    def _get_unique_words(self):
        """ get the unique words for the region and save result in /var/data/ """
        self.unique_words_filepath = '/var/data/region_{}_unique_words'.format(self.region_idx)        
        
        # create a dictionary with frequency of words
        unique_words = set(self.embedding_model.wv.vocab)
        unique_words_count = dict()
        for word in unique_words:
            unique_words_count[word] = self.embedding_model.wv.vocab[word].count
        unique_words_count = dict(sorted(unique_words_count.items(), key=lambda v: v[1], reverse=True))

        # create a copy of that dictionary
        unique_words_count_without_stopwords = unique_words_count.copy()

        # filter out stopwords from the dictionary
        stop_words = set(stopwords.words('english')) 
        for key in unique_words_count.keys():
            if key.lower() in stop_words:
                unique_words_count_without_stopwords.pop(key, None)

        # save result in /var/data/
        with open(self.unique_words_filepath, 'w') as f:
            for word in unique_words_count_without_stopwords.keys():
                f.write('%s\n' % word)
                    
        return unique_words_count_without_stopwords
    
    def _set_vocab(self):
        """ create a vocab lookup table """
        # +1 to include <unk>
        self.vocab_size = len(self._get_unique_words())
        self.vocab = tf.contrib.lookup.index_table_from_file(
            self.unique_words_filepath, 
            vocab_size = self.vocab_size - 1,
            num_oov_buckets = 1
        )
               
    def plot_word_embeddings(self):
        """ plot the word embedding in a 2D space and store it in /var/data/img/ """
        
        unique_words = self._get_unique_words()

        # retrieve all of the vectors from the trained model
        top10 = list(unique_words)[:10]
        top10_x  = self.embedding_model[top10]

        # fit a 2D PCA model to the vectors 
        pca = PCA(n_components=2)
        result = pca.fit_transform(top10_x)
        pyplot.scatter(result[:, 0], result[:, 1])
        for i, word in enumerate(top10):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()
        pyplot.savefig('/var/data/img/region{}_pca_top10'.format(self.region_idx))
        