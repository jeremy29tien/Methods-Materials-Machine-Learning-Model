#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:28:06 2018

Training and testing data obtainted from Melecular and Cellular Biology PMC Collection at https://www.ncbi.nlm.nih.gov/pmc/journals/91/
Loads and preprocesses the data by removing punctuation and white space.
Creates individual tokens ready for neural network.

@author: JeremyTien
"""

import string

DATASET_SIZE = 1000;

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

#save tokens to file
def save_doc(clean_data, filename):
    data = ' '.join(clean_data)
    file = open(filename, 'w')
    file.write(data)
    file.close()

#cell-culture dataset
for x in range(0,DATASET_SIZE):
    # load document
    in_filename = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/cell-culture/' + str(x) + '.txt'
    doc = load_doc(in_filename)
    #clean document
    tokens = clean_doc(doc)
    #split data into test and train and save
    if x % 2 == 0:
        out_filename = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/train/cell-culture/' + str(int(x/2)) + '.txt' #place in TRAIN dataset
    else:
        out_filename = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/test/cell-culture/' + str(int(x/2)) + '.txt' #place in TEST dataset
    print(out_filename)
    save_doc(tokens, out_filename)
    
#cell-imaging dataset
for x in range(0,DATASET_SIZE):
    # load document
    in_filename = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/cell-imaging/' + str(x) + '.txt'
    doc = load_doc(in_filename)
    #clean document
    tokens = clean_doc(doc)
    #split data into test and train and save
    if x % 2 == 0:
        out_filename = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/train/cell-imaging/' + str(int(x/2)) + '.txt' #place in TRAIN dataset
    else:
        out_filename = '/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/test/cell-imaging/' + str(int(x/2)) + '.txt' #place in TEST dataset
    print(out_filename)
    save_doc(tokens, out_filename)
