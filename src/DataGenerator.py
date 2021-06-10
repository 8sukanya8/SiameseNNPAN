# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity


class DataGenerator(Sequence):
    """
    Data Generator for the Koppel Blog Corpus for the Baseline_using_MGBD notebook.
    """
    'Generates data for keras'
    def __init__(self, df, tokenizer, batch_size= 32, num_clases = 2, shuffle = False):
        #self.file = file
        self.batch_size = batch_size
        self.df = df #pd.read_csv(open(file))
        self.indices = self.df.index.to_list()
        self.num_classes = num_clases
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        #self.on_epoch_end()

    def __len__(self):
        return len(self.indices)//self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]  # Select the list of IDs to return as batch
        batch = [self.indices[k] for k in index]  # Generate data
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        """
        This function gets a list of selected indices to form selected batch. A transformation is performed to create X and Y.
        X is the cosine similarity between the embedded paragraphs P1 and P2.
        :param batch: selected indices for df
        :return: batch of data
        """
        # == signifies, same author, != signifies diff author. flipping to observe the difference if any
        y = ((self.df.loc[batch]['author_1'] != self.df.loc[batch]['author_2']).astype(int)).to_numpy() # labels
        p1_column = self.df.loc[batch]['para1_text']
        p2_column = self.df.loc[batch]['para2_text']
        p1_embed = self.tokenizer.texts_to_matrix(p1_column, mode='freq')
        p2_embed = self.tokenizer.texts_to_matrix(p2_column, mode='freq')
        #print(p1_embed[0].shape, p2_embed[0].shape)
        X = np.array([np.squeeze(cosine_similarity([p1_embed[i]], [p2_embed[i]])) for i in range(len(p1_embed))]) # cosine_similarity(p1_embed, p2_embed)
        #print(X.shape)
        return X, to_categorical(y, num_classes=self.num_classes)