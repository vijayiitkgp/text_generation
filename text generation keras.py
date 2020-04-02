# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:05:46 2019

@author: vp999274
"""

# keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import pickle

# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os 

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


curr_dir = 'input_data_headlines/'
all_headlines = []
#reading all files from the directory
for filename in os.listdir(curr_dir):
    #reading articles only
    if 'Articles' in filename:
        article_df = pd.read_csv(curr_dir + filename)
        #getting hedlines in article
        all_headlines.extend(list(article_df.headline.values))
        break

all_headlines = [h for h in all_headlines if h != "Unknown"]
print(len(all_headlines))

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

corpus = [clean_text(x) for x in all_headlines]
print(corpus[:10])

tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    print(total_words)
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
print(inp_sequences[:10])

def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    print(input_sequences[:,:-1])
    print(input_sequences[:,-1])
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    print(label)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    lstm_out=200
    model.add(Bidirectional(LSTM(lstm_out,activation='tanh',recurrent_activation='hard_sigmoid',use_bias=True, dropout=0.0,return_sequences=True), merge_mode='concat'))
    model.add(Bidirectional(LSTM(lstm_out,activation='tanh',recurrent_activation='hard_sigmoid',use_bias=True, dropout=0.0), merge_mode='concat'))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

#model = create_model(max_sequence_len, total_words)
#model.summary()
#
#model.fit(predictors, label, epochs=100, verbose=5)
#model.save("text_gerneration_model.h5")
#tokemizer_maxlen = {
#		"maxlen" : max_sequence_len,
#		"tokenizer" : tokenizer
#                  }
#with open('tokenizer.pickle', 'wb') as handle:
#    pickle.dump(tokemizer_maxlen, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


