#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:18:30 2018

Sites referenced:
    https://keras.io/models/sequential/
    http://web.stanford.edu/class/cs224n/index.html
    https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
    https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/
    https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
    
@author: JeremyTien
"""

import numpy
import os
import tensorflow as tf
import string
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import model_from_json
from pickle import load
import keras.preprocessing.text as prepro


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

#load model from file
#can also try model = load_model('lstm_model.h5')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# be sure to re-compile loaded model
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#preprocess input - remove space and punctuation
# load document
in_filename = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/cell-culture-single-test.txt'
doc = load_doc(in_filename)
#clean document
tokens = clean_doc(doc)
cleaned = ' '.join(tokens)

#load trained tokenizer from MaterialsMethods-Classification.py and train on input data
x_in = []
x_in.append(doc)
tokenizer = load(open('tokenizer.pkl', 'rb'))
in_sequence = tokenizer.texts_to_sequences(x_in)
# truncate and pad input sequences
max_section_length = 500
in_sequence = sequence.pad_sequences(in_sequence, maxlen=max_section_length)
#print(doc)
#print(x_in)
#print(in_sequence)

#make prediction on tokenized sequence (on integers)
prediction = loaded_model.predict_classes(in_sequence, verbose=0) #0
print(prediction)
#convert prediction back to label - 0 for cell-culture, 1 for cell-imaging
if prediction == 0:
    method_material_type = 'cell-culture'
else:
    method_material_type = 'cell-imaging'
print('\nThe input Methods and Materials section is classified as: ' + method_material_type)