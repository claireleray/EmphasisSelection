#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:57:09 2019

@author: claire
"""
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer
import numpy as np

def read_data(filename):
    """
    This function reads the data from .txt file.
    :param filename: reading directory
    :return: lists of word_ids, words, emphasis probabilities, POS tags
    """
    lines = read_lines(filename) + ['']
    word_id_lst, word_id_lsts =[], []
    post_lst, post_lsts = [], []
    bio_lst , bio_lsts = [], []
    freq_lst, freq_lsts = [], []
    e_freq_lst, e_freq_lsts = [], []
    pos_lst, pos_lsts =[], []
    for line in lines:
        if line:
            splitted = line.split("\t")
            word_id = splitted[0]
            words = splitted[1]
            bio= splitted[2]
            freq = splitted[3]
            e_freq = splitted[4]
            pos = splitted[5]

            word_id_lst.append(word_id)
            post_lst.append(words)
            bio_lst.append(bio)
            freq_lst.append(freq)
            e_freq_lst.append(e_freq)
            pos_lst.append(pos)

        elif post_lst:
            word_id_lsts.append(word_id_lst)
            post_lsts.append(post_lst)
            bio_lsts.append(bio_lst)
            freq_lsts.append(freq_lst)
            e_freq_lsts.append(e_freq_lst)
            pos_lsts.append(pos_lst)
            word_id_lst =[]
            post_lst =[]
            bio_lst =[]
            freq_lst =[]
            e_freq_lst =[]
            pos_lst =[]
    return word_id_lsts, post_lsts, bio_lsts, freq_lsts, e_freq_lsts, pos_lsts


def read_lines( filename):
    with open(filename, 'r') as fp:
        lines = [line.strip() for line in fp]
    return lines


def write_results(word_id_lsts, words_lsts, e_freq_lsts, write_to):
    """
    This function writes results in the format.
    :param word_id_lsts: list of word_ids
    :param words_lsts: list of words
    :param e_freq_lsts: lists of emphasis probabilities
    :param write_to: writing directory
    :return:
    """


    with open(write_to, 'w') as out:
        sentence_id=""
        # a loop on sentences:
        for i in range(len(words_lsts)):
            # a loop on words in a sentence:
            for j in range(len(words_lsts[i])):
                # writing:
                if sentence_id ==i:
                    to_write = "{}\t{}\t{}\t".format(word_id_lsts[i][j], words_lsts[i][j], e_freq_lsts[i][j])
                    out.write(to_write + "\n")
                else:
                    out.write("\n")
                    to_write = "{}\t{}\t{}\t".format(word_id_lsts[i][j], words_lsts[i][j], e_freq_lsts[i][j])
                    out.write(to_write + "\n")
                    sentence_id = i
        out.write("\n")
        out.close()
        
        
def tokenize_and_pad(sentences,y, weights) :
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print ("Sentences tokenized - ex: %s" % tokenized_texts[0])
    MAX_LEN = max([len(x) for x in tokenized_texts])
    
    # Pad input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    print("Sentences padded - ex: %s" % input_ids[0])
    
    # Pad y
    y_padded = pad_sequences(y, maxlen=MAX_LEN, dtype="float", truncating="post", padding="post")
    print("Labels padded - ex: %s" % y_padded[0])
    
    # Pad weights
    weights_padded = pad_sequences(weights, maxlen=MAX_LEN, dtype="object", truncating="post", padding="post", value=9.0)
    print("Weights padded - ex: %s" % weights_padded[0])
    return input_ids, y_padded, weights_padded, MAX_LEN

def create_dataset(filename):    
    word_ids, posts, bios, freqs, e_freqs, poss = read_data(filename)

    # Boundaries tokens
    posts_boundaries = ['[CLS] ' + " ".join(words) + ' [SEP]' for words in posts]
    print("Boundaries tokens added - ex: %s" % posts_boundaries[0])
    
    # Labels
    e_freqs_float = [list(map(float,e_freq_lst)) for e_freq_lst in e_freqs]
    labels = [[0] + list(map(round,e_freq_lst)) for e_freq_lst in e_freqs_float]
    
    # Weights
    weights = []
    for freq in freqs:
        weights_sentence = [9.0]
        for word_bio in freq:
            word_bio_lst = list(map(float,word_bio.split("|")))
            weight = max(word_bio_lst)
            weights_sentence.append(weight)
        weights.append(weights_sentence)
    
    input_ids, y_padded, weights_padded, maxlen = tokenize_and_pad(posts_boundaries, labels, weights)

    X = input_ids
    y = y_padded
    
    return np.array(X), np.array(y), np.array(weights_padded)  
    

    

