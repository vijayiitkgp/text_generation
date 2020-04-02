# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:39:50 2019

@author: vp999274
"""
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer_maxlen = pickle.load(handle)
  
    
#tokemizer_maxlen = {
#		"maxlen" : X_test.shape[1],
#		"tokenizer" : tokenizer
#	}    
max_sequence_len=tokenizer_maxlen.get("maxlen", "none")    
tokenizer=tokenizer_maxlen.get("tokenizer", "none")    

loaded_model = load_model('text_gerneration_model.h5')

seed_text="taking things"
#seed_text="what was lenin"
next_words=2


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

print(generate_text(seed_text, next_words, loaded_model, max_sequence_len ))