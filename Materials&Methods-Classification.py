#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 00:37:03 2018

Training and testing data obtainted from Molecular and Cellular Biology PMC Collection at https://www.ncbi.nlm.nih.gov/pmc/journals/91/

@author: JeremyTien
"""

import numpy
import os
import tensorflow as tf
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
import keras.preprocessing.text as prepro

DATASET_SIZE = 1000;

x_train = [];
y_train = [];
x_test = [];
y_test = [];

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

#for training dataset
for x in range(0, DATASET_SIZE):
    # label 0 for cell-culture, 1 for cell-imaging
    if x % 2 == 0:
        file = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/train/cell-culture/' + str(int(x/2)) + '.txt' #alternate between culture and imaging
        y_train.append(0)
    else:
        file = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/train/cell-imaging/' + str(int(x/2)) + '.txt'
        y_train.append(1)
    x_train.append(load_doc(file))
    
#for testing dataset
for x in range(0, DATASET_SIZE):
    # label 0 for cell-culture, 1 for cell-imaging
    if x % 2 == 0:
        file = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/test/cell-culture/' + str(int(x/2)) + '.txt' #alternate between culture and imaging
        y_test.append(0)
    else:
        file = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/test/cell-imaging/' + str(int(x/2)) + '.txt'
        y_test.append(1)
    x_test.append(load_doc(file))
    
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train + x_test) # train tokenizer on test and train sets
x_train_sequences = tokenizer.texts_to_sequences(x_train)
x_test_sequences = tokenizer.texts_to_sequences(x_test)
#vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# truncate and pad input sequences
max_section_length = 500
x_train_sequences = sequence.pad_sequences(x_train_sequences, maxlen=max_section_length)
x_test_sequences = sequence.pad_sequences(x_test_sequences, maxlen=max_section_length)

# create the model
embedding_vector_length = 32 # experiment with embedding size later for optimal results
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_section_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')) # add convolutional neural net for 1D spatial structure in sentences
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) # input dropout and recurrent dropout to prevent overfitting
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

#compile model -- for nonbinary classification, use loss = 'categorical_crossentropy'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # API Documentation at https://keras.io/models/sequential/
#fit model
model.fit(x_train_sequences, y_train, validation_data=(x_test_sequences, y_test), epochs=10, batch_size=64) # can increase number of epochs to achieve better accuracy

# evaluation of the trained model
scores = model.evaluate(x_test_sequences, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
print("Saved model to disk")

