#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:14:17 2018

@author: JeremyTien
"""
import numpy
import os
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import model_from_json
import keras.preprocessing.text as prepro

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words) # data at /Users/JeremyTien/anaconda3/lib/python3.6/site-packages/keras/datasets 

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# be sure to re-compile loaded model
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# predict sentiment for a string of text
text = "This is a very bad movie. I hated every minute of it."
text = prepro.one_hot(text, top_words, lower=True, split=" ")
text = [text]
text = sequence.pad_sequences(text, maxlen=max_review_length)
predictions = loaded_model.predict(text)
print(predictions)

# evaluation of the loaded model
#scores = loaded_model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))